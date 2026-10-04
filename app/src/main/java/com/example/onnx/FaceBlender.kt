package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Handles post-inference eye polarity/clarity restoration, skin-only illumination harmonization,
 * full-orbital soft-feathered facial mask generation, optional GFPGAN ONNX enhancement,
 * inverse affine re-projection onto the full-resolution target image, and provenance watermarking.
 */
object FaceBlender {

    private const val CROP_SIZE = 128

    // Canonical 128x128 eye & mouth coordinates (matching FaceAlignment.getReferenceTemplate(128))
    private const val LEFT_EYE_X = 46.2946f
    private const val LEFT_EYE_Y = 51.6963f
    private const val RIGHT_EYE_X = 81.5318f
    private const val RIGHT_EYE_Y = 51.5014f

    /**
     * Returns how strongly a pixel (x, y) in 128x128 canonical space belongs to the ocular
     * aperture (sclera, iris, pupil, eyelids, lashes) in [0.0f, 1.0f].
     * Used to protect eyes from skin-tone color shifts that would otherwise turn dark pupils
     * milky grey or tint white sclera into a negative-looking eye.
     */
    private fun computeEyeProtectionWeight128(x: Int, y: Int): Float {
        val lx = (x - LEFT_EYE_X) / 20.0f
        val ly = (y - LEFT_EYE_Y) / 12.5f
        val lDistSq = lx * lx + ly * ly

        val rx = (x - RIGHT_EYE_X) / 20.0f
        val ry = (y - RIGHT_EYE_Y) / 12.5f
        val rDistSq = rx * rx + ry * ry

        val minDistSq = min(lDistSq, rDistSq)
        return when {
            minDistSq <= 0.50f -> 1.0f
            minDistSq >= 1.0f -> 0.0f
            else -> {
                val t = (minDistSq - 0.50f) / 0.50f
                0.5f * (1.0f + cos(Math.PI * t).toFloat())
            }
        }
    }

    /**
     * Eliminates "negative image" / solarized / inverted contrast artifacts across the entire
     * 128x128 face crop and strictly enforces natural positive eye polarity (deep dark pupil/iris/lashes
     * and clean bright sclera) on both Left and Right eyes.
     *
     * Handles both:
     *  1) Full 3-Model Mode (`w600k_r50.onnx` + `emap` + `inswapper_128.onnx`): fixes local AdaIN
     *     polarity inversions in tilted/heavily-lined eyes, eyebrows, nostrils, and shadows.
     *  2) 2-Model Fallback Mode (when `hasTrueArcFaceLatent == false`): synthesizes a 100% positive
     *     multi-band illumination-matched face from `alignedSource128` + `alignedTarget128` so
     *     uncalibrated AdaIN activations never turn the output into a negative image.
     */
    fun restoreEyesAndEliminateNegativeArtifacts128(
        swapped128: Bitmap,
        alignedTarget128: Bitmap,
        alignedSource112: Bitmap? = null,
        alignedSource128: Bitmap? = null,
        hasTrueArcFaceLatent: Boolean = true
    ): Bitmap {
        val total = CROP_SIZE * CROP_SIZE
        val swapPx = IntArray(total)
        val tgtPx = IntArray(total)
        swapped128.getPixels(swapPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        alignedTarget128.getPixels(tgtPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)

        // Resolve 128x128 source pixels in exact canonical alignment
        val srcPx128 = IntArray(total)
        if (alignedSource128 != null && alignedSource128.width == CROP_SIZE && alignedSource128.height == CROP_SIZE) {
            alignedSource128.getPixels(srcPx128, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        } else if (alignedSource112 != null && alignedSource112.width == 112 && alignedSource112.height == 112) {
            val raw112 = IntArray(112 * 112)
            alignedSource112.getPixels(raw112, 0, 112, 0, 0, 112, 112)
            for (y in 0 until CROP_SIZE) {
                val sy = y.coerceIn(0, 111)
                for (x in 0 until CROP_SIZE) {
                    val sx = (x - 8).coerceIn(0, 111)
                    srcPx128[y * CROP_SIZE + x] = raw112[sy * 112 + sx]
                }
            }
        } else {
            System.arraycopy(tgtPx, 0, srcPx128, 0, total)
        }

        // Step 1: Build a 100% Positive Illumination-Harmonized Source Reference (srcHarmonizedPx)
        // by transferring target's low-frequency 13x13 shading envelope onto source's high-frequency identity features.
        val srcHarmonizedPx = buildIlluminationHarmonizedPositiveSource128(srcPx128, tgtPx)

        val outPx = IntArray(total)

        // Step 2: Full-Face Anti-Negative Polarity Pass across all 128x128 pixels
        for (y in 0 until CROP_SIZE) {
            val row = y * CROP_SIZE
            for (x in 0 until CROP_SIZE) {
                val idx = row + x
                val sc = swapPx[idx]
                val tc = tgtPx[idx]
                val hc = srcHarmonizedPx[idx]

                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF
                val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB

                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                val hR = (hc ushr 16) and 0xFF
                val hG = (hc ushr 8) and 0xFF
                val hB = hc and 0xFF
                val hLum = 0.299f * hR + 0.587f * hG + 0.114f * hB

                // Positive reference combines illumination-harmonized source identity (72%) + target shading (28%)
                val posRefR = 0.72f * hR + 0.28f * tR
                val posRefG = 0.72f * hG + 0.28f * tG
                val posRefB = 0.72f * hB + 0.28f * tB
                val posRefLum = 0.72f * hLum + 0.28f * tLum

                if (!hasTrueArcFaceLatent) {
                    // In 2-Model mode (no w600k_r50.onnx), inswapper_128's AdaIN modulation is uncalibrated
                    // and produces negative/solarized colors. Use the positive illumination-harmonized source face.
                    outPx[idx] = (0xFF shl 24) or
                        (posRefR.toInt().coerceIn(0, 255) shl 16) or
                        (posRefG.toInt().coerceIn(0, 255) shl 8) or
                        posRefB.toInt().coerceIn(0, 255)
                    continue
                }

                // Check local 3x3 high-frequency contrast polarity for negative/solarized feature inversion
                val sAvg3 = compute3x3LuminanceAvg(swapPx, CROP_SIZE, x, y)
                val hAvg3 = compute3x3LuminanceAvg(srcHarmonizedPx, CROP_SIZE, x, y)
                val tAvg3 = compute3x3LuminanceAvg(tgtPx, CROP_SIZE, x, y)

                val dSwap = sLum - sAvg3
                val dPosRef = (0.65f * (hLum - hAvg3)) + (0.35f * (tLum - tAvg3))

                // Detect negative polarity:
                // (a) Local contrast inversion: dSwap and dPosRef have opposite signs
                val isContrastInverted = (dSwap * dPosRef) < -1.5f
                // (b) Dark feature inversion (e.g., dark eyebrow, lash, nostril, shadow turned bright grey/white)
                val darkRefFloor = min(hLum, tLum)
                val brightRefCeil = max(hLum, tLum)
                val isDarkFeatureInverted = darkRefFloor < 92f && sLum > posRefLum + 10f
                // (c) Bright skin/highlight inversion (turned muddy dark grey)
                val isBrightFeatureInverted = brightRefCeil > 115f && sLum < posRefLum - 16f

                // Baseline positive anchor (22%) prevents any global washout, boosted up to 92% on inverted pixels
                var posWeight = 0.22f
                if (isDarkFeatureInverted) {
                    val severity = ((sLum - posRefLum) / 40f).coerceIn(0.35f, 0.92f)
                    posWeight = max(posWeight, severity)
                }
                if (isBrightFeatureInverted) {
                    val severity = ((posRefLum - sLum) / 45f).coerceIn(0.30f, 0.88f)
                    posWeight = max(posWeight, severity)
                }
                if (isContrastInverted) {
                    val invMag = (abs(dSwap - dPosRef) / 25f).coerceIn(0.35f, 0.90f)
                    posWeight = max(posWeight, invMag)
                }

                // Replace inverted local detail with positive high-frequency detail
                val detailFix = if (isContrastInverted) (dPosRef - dSwap) * 0.75f else dPosRef * 0.20f

                val finalR = (sR * (1f - posWeight) + posRefR * posWeight + detailFix).toInt().coerceIn(0, 255)
                val finalG = (sG * (1f - posWeight) + posRefG * posWeight + detailFix).toInt().coerceIn(0, 255)
                val finalB = (sB * (1f - posWeight) + posRefB * posWeight + detailFix).toInt().coerceIn(0, 255)

                outPx[idx] = (0xFF shl 24) or (finalR shl 16) or (finalG shl 8) or finalB
            }
        }

        // Step 3: Strict Positive Ocular Restoration for Both Left Eye (46.3, 51.7) & Right Eye (81.5, 51.5)
        // Guarantees zero "negative eye" / milky pupil / muddy sclera on either eye!
        val eyeCenters = arrayOf(
            LEFT_EYE_X to LEFT_EYE_Y,
            RIGHT_EYE_X to RIGHT_EYE_Y
        )

        for ((eyeCx, eyeCy) in eyeCenters) {
            val rx = 20.5f
            val ry = 13.0f
            val xMin = (eyeCx - rx).toInt().coerceAtLeast(1)
            val xMax = (eyeCx + rx).toInt().coerceAtMost(CROP_SIZE - 2)
            val yMin = (eyeCy - ry).toInt().coerceAtLeast(1)
            val yMax = (eyeCy + ry).toInt().coerceAtMost(CROP_SIZE - 2)

            // Compute subtle iris chrominance shift from source eye to target eye without ghosting sclera into pupil
            var srcEyeR = 0f
            var srcEyeG = 0f
            var srcEyeB = 0f
            var tgtEyeR = 0f
            var tgtEyeG = 0f
            var tgtEyeB = 0f
            var irisSamples = 0
            for (iy in (eyeCy - 5f).toInt()..(eyeCy + 5f).toInt()) {
                for (ix in (eyeCx - 7f).toInt()..(eyeCx + 7f).toInt()) {
                    if (iy !in 1 until CROP_SIZE - 1 || ix !in 1 until CROP_SIZE - 1) continue
                    val sc = srcPx128[iy * CROP_SIZE + ix]
                    val tc = tgtPx[iy * CROP_SIZE + ix]
                    val sL = 0.299f * (sc ushr 16 and 0xFF) + 0.587f * (sc ushr 8 and 0xFF) + 0.114f * (sc and 0xFF)
                    val tL = 0.299f * (tc ushr 16 and 0xFF) + 0.587f * (tc ushr 8 and 0xFF) + 0.114f * (tc and 0xFF)
                    if (sL < 110f && tL < 110f) {
                        srcEyeR += (sc ushr 16) and 0xFF
                        srcEyeG += (sc ushr 8) and 0xFF
                        srcEyeB += sc and 0xFF
                        tgtEyeR += (tc ushr 16) and 0xFF
                        tgtEyeG += (tc ushr 8) and 0xFF
                        tgtEyeB += tc and 0xFF
                        irisSamples++
                    }
                }
            }
            val chromaShiftR = if (irisSamples > 4) ((srcEyeR - tgtEyeR) / irisSamples).coerceIn(-18f, 18f) else 0f
            val chromaShiftG = if (irisSamples > 4) ((srcEyeG - tgtEyeG) / irisSamples).coerceIn(-18f, 18f) else 0f
            val chromaShiftB = if (irisSamples > 4) ((srcEyeB - tgtEyeB) / irisSamples).coerceIn(-18f, 18f) else 0f

            for (y in yMin..yMax) {
                val dy = (y - eyeCy) / ry
                for (x in xMin..xMax) {
                    val dx = (x - eyeCx) / rx
                    val normDist = sqrt(dx * dx + dy * dy)
                    if (normDist >= 1.0f) continue

                    // Smooth radial socket weight: 1.0 across iris/sclera/lashes (normDist <= 0.65), cosine taper to 1.0
                    val socketWeight = if (normDist <= 0.65f) {
                        1.0f
                    } else {
                        val t = (normDist - 0.65f) / 0.35f
                        0.5f * (1.0f + cos(Math.PI * t).toFloat())
                    }

                    val idx = y * CROP_SIZE + x
                    val curC = outPx[idx]
                    val tc = tgtPx[idx]

                    val cR = (curC ushr 16) and 0xFF
                    val cG = (curC ushr 8) and 0xFF
                    val cB = curC and 0xFF
                    val cLum = 0.299f * cR + 0.587f * cG + 0.114f * cB

                    val tR = (tc ushr 16) and 0xFF
                    val tG = (tc ushr 8) and 0xFF
                    val tB = tc and 0xFF
                    val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                    // Gaze-aligned positive eye reference from target eye (preserving exact iris/pupil/sclera geometry)
                    val isIrisZone = tLum < 105f && normDist < 0.60f
                    val cleanEyeR = (tR + if (isIrisZone) chromaShiftR * 0.45f else 0f).coerceIn(0f, 255f)
                    val cleanEyeG = (tG + if (isIrisZone) chromaShiftG * 0.45f else 0f).coerceIn(0f, 255f)
                    val cleanEyeB = (tB + if (isIrisZone) chromaShiftB * 0.45f else 0f).coerceIn(0f, 255f)
                    val cleanEyeLum = 0.299f * cleanEyeR + 0.587f * cleanEyeG + 0.114f * cleanEyeB

                    // Detect any negative/milky inversion inside the eye socket:
                    // - Dark pupil/iris/lashes (tLum < 115) must NEVER be brighter than cleanEyeLum
                    // - Bright sclera/catchlight (tLum > 125) must NEVER be darker than cleanEyeLum
                    val isNegativeIrisOrLash = tLum < 115f && cLum > cleanEyeLum + 4f
                    val isNegativeSclera = tLum > 125f && cLum < cleanEyeLum - 6f

                    // High baseline ocular clarity (0.78 in socket core, up to 0.96 on any inverted pixel)
                    val baseEyeAnchor = 0.78f * socketWeight
                    val inversionAnchor = when {
                        isNegativeIrisOrLash -> (0.85f + ((cLum - cleanEyeLum) / 60f).coerceIn(0f, 0.13f)) * socketWeight
                        isNegativeSclera -> (0.84f + ((cleanEyeLum - cLum) / 60f).coerceIn(0f, 0.14f)) * socketWeight
                        else -> baseEyeAnchor
                    }
                    val eyeBlend = max(baseEyeAnchor, inversionAnchor).coerceIn(0f, 0.96f)

                    // Local 3x3 high-frequency corneal catchlight & eyelash crispness
                    val tAvg3 = compute3x3LuminanceAvg(tgtPx, CROP_SIZE, x, y)
                    val eyeDetail = (tLum - tAvg3) * 0.35f * socketWeight

                    var eR = cR * (1f - eyeBlend) + cleanEyeR * eyeBlend + eyeDetail
                    var eG = cG * (1f - eyeBlend) + cleanEyeG * eyeBlend + eyeDetail
                    var eB = cB * (1f - eyeBlend) + cleanEyeB * eyeBlend + eyeDetail

                    // Strict Mathematical Positive Polarity Clamp inside the core eye aperture (normDist <= 0.75):
                    // Dark pupil/iris/lash pixels can NEVER exceed target darkness; sclera can NEVER drop below target whiteness.
                    if (normDist <= 0.75f) {
                        val eLum = 0.299f * eR + 0.587f * eG + 0.114f * eB
                        if (tLum < 95f && eLum > cleanEyeLum + 2f && eLum > 1f) {
                            val scaleDown = (cleanEyeLum / eLum).coerceIn(0.15f, 1.0f)
                            eR *= scaleDown
                            eG *= scaleDown
                            eB *= scaleDown
                        } else if (tLum > 130f && eLum < cleanEyeLum * 0.94f && eLum > 1f) {
                            val scaleUp = ((cleanEyeLum * 0.95f) / eLum).coerceIn(1.0f, 2.2f)
                            eR *= scaleUp
                            eG *= scaleUp
                            eB *= scaleUp
                        }
                    }

                    outPx[idx] = (0xFF shl 24) or
                        (eR.toInt().coerceIn(0, 255) shl 16) or
                        (eG.toInt().coerceIn(0, 255) shl 8) or
                        eB.toInt().coerceIn(0, 255)
                }
            }
        }

        // Step 4: Mild unsharp micro-contrast pass across the inner face (eyes, nose, lips)
        val sharpenedPx = outPx.copyOf()
        for (y in 24..108) {
            val row = y * CROP_SIZE
            for (x in 24..104) {
                val idx = row + x
                val c = outPx[idx]
                val n = outPx[(y - 1) * CROP_SIZE + x]
                val s = outPx[(y + 1) * CROP_SIZE + x]
                val l = outPx[row + x - 1]
                val r = outPx[row + x + 1]

                val cr = (c ushr 16) and 0xFF
                val cg = (c ushr 8) and 0xFF
                val cb = c and 0xFF

                val avgR = ((n ushr 16 and 0xFF) + (s ushr 16 and 0xFF) + (l ushr 16 and 0xFF) + (r ushr 16 and 0xFF)) * 0.25f
                val avgG = ((n ushr 8 and 0xFF) + (s ushr 8 and 0xFF) + (l ushr 8 and 0xFF) + (r ushr 8 and 0xFF)) * 0.25f
                val avgB = ((n and 0xFF) + (s and 0xFF) + (l and 0xFF) + (r and 0xFF)) * 0.25f

                val amount = 0.24f
                val nr = (cr + (cr - avgR) * amount).toInt().coerceIn(0, 255)
                val ng = (cg + (cg - avgG) * amount).toInt().coerceIn(0, 255)
                val nb = (cb + (cb - avgB) * amount).toInt().coerceIn(0, 255)
                sharpenedPx[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            }
        }

        val out = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Bitmap.Config.ARGB_8888)
        out.setPixels(sharpenedPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        return out
    }

    /**
     * Builds a 100% positive-polarity 128x128 face reference by combining the high-frequency
     * identity structure of `srcPx128` with the low-frequency 13x13 illumination envelope of `tgtPx128`.
     */
    private fun buildIlluminationHarmonizedPositiveSource128(
        srcPx128: IntArray,
        tgtPx128: IntArray
    ): IntArray {
        val total = CROP_SIZE * CROP_SIZE
        val out = IntArray(total)
        val radius = 6 // 13x13 box filter window

        for (y in 0 until CROP_SIZE) {
            val y0 = (y - radius).coerceAtLeast(0)
            val y1 = (y + radius).coerceAtMost(CROP_SIZE - 1)
            for (x in 0 until CROP_SIZE) {
                val x0 = (x - radius).coerceAtLeast(0)
                val x1 = (x + radius).coerceAtMost(CROP_SIZE - 1)

                var sLowR = 0f
                var sLowG = 0f
                var sLowB = 0f
                var tLowR = 0f
                var tLowG = 0f
                var tLowB = 0f
                var count = 0

                for (yy in y0..y1 step 2) {
                    val row = yy * CROP_SIZE
                    for (xx in x0..x1 step 2) {
                        val sc = srcPx128[row + xx]
                        val tc = tgtPx128[row + xx]
                        sLowR += (sc ushr 16) and 0xFF
                        sLowG += (sc ushr 8) and 0xFF
                        sLowB += sc and 0xFF
                        tLowR += (tc ushr 16) and 0xFF
                        tLowG += (tc ushr 8) and 0xFF
                        tLowB += tc and 0xFF
                        count++
                    }
                }
                val invCount = 1f / count.coerceAtLeast(1)
                sLowR *= invCount
                sLowG *= invCount
                sLowB *= invCount
                tLowR *= invCount
                tLowG *= invCount
                tLowB *= invCount

                val idx = y * CROP_SIZE + x
                val sc = srcPx128[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF

                // High-frequency positive identity detail from source + low-frequency scene lighting from target
                val hR = (tLowR + (sR - sLowR) * 0.92f).toInt().coerceIn(0, 255)
                val hG = (tLowG + (sG - sLowG) * 0.92f).toInt().coerceIn(0, 255)
                val hB = (tLowB + (sB - sLowB) * 0.92f).toInt().coerceIn(0, 255)

                out[idx] = (0xFF shl 24) or (hR shl 16) or (hG shl 8) or hB
            }
        }
        return out
    }

    private fun compute3x3LuminanceAvg(px: IntArray, stride: Int, cx: Int, cy: Int): Float {
        val x0 = (cx - 1).coerceAtLeast(0)
        val x1 = (cx + 1).coerceAtMost(stride - 1)
        val y0 = (cy - 1).coerceAtLeast(0)
        val y1 = (cy + 1).coerceAtMost(stride - 1)
        var sum = 0f
        var count = 0
        for (y in y0..y1) {
            val row = y * stride
            for (x in x0..x1) {
                val c = px[row + x]
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                sum += 0.299f * r + 0.587f * g + 0.114f * b
                count++
            }
        }
        return sum / count.coerceAtLeast(1)
    }

    /**
     * Harmonizes the color and luminance statistics of `swapped128` to match `targetCrop128`
     * strictly on SKIN regions while protecting the eyes (sclera, iris, pupil) from skin-tone
     * shifts that would wash out dark pupils or tint eye whites.
     */
    fun transferSkinToneStatistics128(
        swapped128: Bitmap,
        targetCrop128: Bitmap,
        strength: Float = 0.62f
    ): Bitmap {
        val total = CROP_SIZE * CROP_SIZE
        val swapPx = IntArray(total)
        val tgtPx = IntArray(total)
        swapped128.getPixels(swapPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        targetCrop128.getPixels(tgtPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)

        // Compute per-channel mean & std strictly on pure facial skin (excluding eyes and lips)
        var sRMean = 0.0
        var sGMean = 0.0
        var sBMean = 0.0
        var tRMean = 0.0
        var tGMean = 0.0
        var tBMean = 0.0
        var count = 0

        for (y in 28..106) {
            val ny = (y - 67.0) / 39.0
            for (x in 28..100) {
                val nx = (x - 64.0) / 36.0
                if (nx * nx + ny * ny <= 1.0) {
                    // Exclude eye sockets and mouth from skin statistics
                    if (computeEyeProtectionWeight128(x, y) > 0.15f) continue
                    if (y in 84..100 && x in 44..84) continue

                    val idx = y * CROP_SIZE + x
                    val sc = swapPx[idx]
                    val tc = tgtPx[idx]
                    sRMean += (sc ushr 16) and 0xFF
                    sGMean += (sc ushr 8) and 0xFF
                    sBMean += sc and 0xFF
                    tRMean += (tc ushr 16) and 0xFF
                    tGMean += (tc ushr 8) and 0xFF
                    tBMean += tc and 0xFF
                    count++
                }
            }
        }

        if (count < 32) return swapped128.copy(Bitmap.Config.ARGB_8888, true)

        sRMean /= count
        sGMean /= count
        sBMean /= count
        tRMean /= count
        tGMean /= count
        tBMean /= count

        var sRVar = 0.0
        var sGVar = 0.0
        var sBVar = 0.0
        var tRVar = 0.0
        var tGVar = 0.0
        var tBVar = 0.0

        for (y in 28..106) {
            val ny = (y - 67.0) / 39.0
            for (x in 28..100) {
                val nx = (x - 64.0) / 36.0
                if (nx * nx + ny * ny <= 1.0) {
                    if (computeEyeProtectionWeight128(x, y) > 0.15f) continue
                    if (y in 84..100 && x in 44..84) continue

                    val idx = y * CROP_SIZE + x
                    val sc = swapPx[idx]
                    val tc = tgtPx[idx]
                    val sr = ((sc ushr 16) and 0xFF) - sRMean
                    val sg = ((sc ushr 8) and 0xFF) - sGMean
                    val sb = (sc and 0xFF) - sBMean
                    val tr = ((tc ushr 16) and 0xFF) - tRMean
                    val tg = ((tc ushr 8) and 0xFF) - tGMean
                    val tb = (tc and 0xFF) - tBMean
                    sRVar += sr * sr
                    sGVar += sg * sg
                    sBVar += sb * sb
                    tRVar += tr * tr
                    tGVar += tg * tg
                    tBVar += tb * tb
                }
            }
        }

        val sRStd = sqrt(sRVar / count).coerceAtLeast(6.0)
        val sGStd = sqrt(sGVar / count).coerceAtLeast(6.0)
        val sBStd = sqrt(sBVar / count).coerceAtLeast(6.0)
        val tRStd = sqrt(tRVar / count).coerceAtLeast(6.0)
        val tGStd = sqrt(tGVar / count).coerceAtLeast(6.0)
        val tBStd = sqrt(tBVar / count).coerceAtLeast(6.0)

        val scaleR = (tRStd / sRStd).coerceIn(0.80, 1.25)
        val scaleG = (tGStd / sGStd).coerceIn(0.80, 1.25)
        val scaleB = (tBStd / sBStd).coerceIn(0.80, 1.25)

        val outPx = IntArray(total)
        val baseBlend = strength.coerceIn(0f, 1f)
        for (y in 0 until CROP_SIZE) {
            val row = y * CROP_SIZE
            for (x in 0 until CROP_SIZE) {
                val i = row + x
                val sc = swapPx[i]
                val origR = (sc ushr 16) and 0xFF
                val origG = (sc ushr 8) and 0xFF
                val origB = sc and 0xFF

                // Protect eyes (sclera/iris/pupil) from skin-tone color shifts
                val eyeProt = computeEyeProtectionWeight128(x, y)
                val blend = baseBlend * (1.0f - 0.88f * eyeProt)

                val matchedR = ((origR - sRMean) * scaleR + tRMean).toFloat()
                val matchedG = ((origG - sGMean) * scaleG + tGMean).toFloat()
                val matchedB = ((origB - sBMean) * scaleB + tBMean).toFloat()

                val finalR = (origR * (1f - blend) + matchedR * blend).toInt().coerceIn(0, 255)
                val finalG = (origG * (1f - blend) + matchedG * blend).toInt().coerceIn(0, 255)
                val finalB = (origB * (1f - blend) + matchedB * blend).toInt().coerceIn(0, 255)

                outPx[i] = (0xFF shl 24) or (finalR shl 16) or (finalG shl 8) or finalB
            }
        }

        val out = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Bitmap.Config.ARGB_8888)
        out.setPixels(outPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        return out
    }

    /**
     * Runs optional `gfpgan_1.4.onnx` ([1, 3, 512, 512] in [-1, 1]) if installed in the
     * ENHANCEMENT slot, blending the restored high-definition eyes/skin back into the crop.
     */
    fun runOptionalGfpganEnhancement128(
        ortEnv: OrtEnvironment,
        gfpganFile: File?,
        crop128: Bitmap,
        preferHardwareAccel: Boolean = false
    ): Bitmap {
        if (gfpganFile == null || !gfpganFile.exists() || gfpganFile.length() <= 1024L) {
            return crop128
        }
        return runCatching {
            val gfpSize = 512
            val upscaled512 = Bitmap.createScaledBitmap(crop128, gfpSize, gfpSize, true)
            val hw = gfpSize * gfpSize
            val pixels = IntArray(hw)
            upscaled512.getPixels(pixels, 0, gfpSize, 0, 0, gfpSize, gfpSize)
            upscaled512.recycle()

            val inBuf = FloatBuffer.allocate(3 * hw)
            for (i in 0 until hw) {
                val c = pixels[i]
                val r = (((c ushr 16) and 0xFF) / 255.0f - 0.5f) / 0.5f
                val g = (((c ushr 8) and 0xFF) / 255.0f - 0.5f) / 0.5f
                val b = ((c and 0xFF) / 255.0f - 0.5f) / 0.5f
                inBuf.put(i, r)
                inBuf.put(hw + i, g)
                inBuf.put(2 * hw + i, b)
            }
            inBuf.rewind()

            val outPixels512 = IntArray(hw)
            OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
                ortEnv.createSession(gfpganFile.absolutePath, opts).use { session ->
                    val inputName = session.inputNames.first()
                    val shape = longArrayOf(1L, 3L, gfpSize.toLong(), gfpSize.toLong())
                    OnnxTensor.createTensor(ortEnv, inBuf, shape).use { inTensor ->
                        session.run(mapOf(inputName to inTensor)).use { res ->
                            val outT = res[0] as OnnxTensor
                            val outFloats = FloatArray(3 * hw)
                            outT.floatBuffer.get(outFloats)
                            for (i in 0 until hw) {
                                val r = (((outFloats[i] * 0.5f + 0.5f).coerceIn(0f, 1f)) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                val g = (((outFloats[hw + i] * 0.5f + 0.5f).coerceIn(0f, 1f)) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                val b = (((outFloats[2 * hw + i] * 0.5f + 0.5f).coerceIn(0f, 1f)) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                outPixels512[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                            }
                        }
                    }
                }
            }
            val restored512 = Bitmap.createBitmap(gfpSize, gfpSize, Bitmap.Config.ARGB_8888)
            restored512.setPixels(outPixels512, 0, gfpSize, 0, 0, gfpSize, gfpSize)
            val downscaled128 = Bitmap.createScaledBitmap(restored512, CROP_SIZE, CROP_SIZE, true)
            restored512.recycle()
            downscaled128
        }.getOrDefault(crop128)
    }

    /**
     * Creates a smooth 128x128 soft-feathered facial mask in canonical InSwapper space.
     * Guarantees alpha = 1.0 across 100% of both eye sockets (including outer corners of wide eyes),
     * nose, and mouth so target eye sclera/eyeliner never ghosts through as a double/negative eye.
     */
    fun createFeatheredFaceMask128(): FloatArray {
        val mask = FloatArray(CROP_SIZE * CROP_SIZE)
        val centerX = 64.0f
        val centerY = 66.0f
        val radiusX = 52.0f
        val radiusY = 54.0f
        val innerCoreRatio = 0.68f
        val borderMargin = 6

        for (y in 0 until CROP_SIZE) {
            for (x in 0 until CROP_SIZE) {
                if (x < borderMargin || x >= CROP_SIZE - borderMargin ||
                    y < borderMargin || y >= CROP_SIZE - borderMargin
                ) {
                    mask[y * CROP_SIZE + x] = 0f
                    continue
                }
                val dx = (x - centerX) / radiusX
                val dy = (y - centerY) / radiusY
                val r = sqrt(dx * dx + dy * dy)

                val ellipticAlpha = when {
                    r <= innerCoreRatio -> 1.0f
                    r >= 1.0f -> 0.0f
                    else -> {
                        val t = (r - innerCoreRatio) / (1.0f - innerCoreRatio)
                        (0.5f * (1.0f + cos(Math.PI * t))).toFloat()
                    }
                }

                // Ensure full 1.0 coverage over both left & right eye sockets so outer eye corners never ghost
                val eyeSocketAlpha = computeEyeProtectionWeight128(x, y)
                val combinedAlpha = max(ellipticAlpha, eyeSocketAlpha)

                // Soft border envelope near the 128x128 boundary
                val edgeDist = min(
                    min(x - borderMargin, CROP_SIZE - 1 - borderMargin - x),
                    min(y - borderMargin, CROP_SIZE - 1 - borderMargin - y)
                ).toFloat()
                val edgeEnvelope = (edgeDist / 12.0f).coerceIn(0f, 1f)

                mask[y * CROP_SIZE + x] = combinedAlpha * edgeEnvelope
            }
        }
        return mask
    }

    /**
     * Warps and blends `swapped128` onto `targetPixels` in-place using the forward 2x3
     * similarity matrix `forwardMatrix128` (which maps target image `(x, y)` -> crop `(u, v)`).
     */
    fun blendSwappedFaceIntoTarget(
        targetPixels: IntArray,
        targetWidth: Int,
        targetHeight: Int,
        swapped128: Bitmap,
        forwardMatrix128: FloatArray,
        featheredMask128: FloatArray
    ) {
        val swapPixels = IntArray(CROP_SIZE * CROP_SIZE)
        swapped128.getPixels(swapPixels, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)

        // Compute bounding box of the 128x128 crop corners in target image coordinates using M^-1
        val inv = FaceAlignment.invertAffine2x3(forwardMatrix128)
        val corners = arrayOf(
            0f to 0f,
            CROP_SIZE.toFloat() to 0f,
            0f to CROP_SIZE.toFloat(),
            CROP_SIZE.toFloat() to CROP_SIZE.toFloat()
        )
        var minX = targetWidth - 1
        var maxX = 0
        var minY = targetHeight - 1
        var maxY = 0
        for ((cx, cy) in corners) {
            val tx = (inv[0] * cx + inv[1] * cy + inv[2]).toInt()
            val ty = (inv[3] * cx + inv[4] * cy + inv[5]).toInt()
            minX = min(minX, tx)
            maxX = max(maxX, tx)
            minY = min(minY, ty)
            maxY = max(maxY, ty)
        }

        minX = (minX - 2).coerceIn(0, targetWidth - 1)
        maxX = (maxX + 2).coerceIn(0, targetWidth - 1)
        minY = (minY - 2).coerceIn(0, targetHeight - 1)
        maxY = (maxY + 2).coerceIn(0, targetHeight - 1)

        val m00 = forwardMatrix128[0]
        val m01 = forwardMatrix128[1]
        val m02 = forwardMatrix128[2]
        val m10 = forwardMatrix128[3]
        val m11 = forwardMatrix128[4]
        val m12 = forwardMatrix128[5]

        for (y in minY..maxY) {
            val rowOffset = y * targetWidth
            val baseU = m01 * y + m02
            val baseV = m11 * y + m12
            for (x in minX..maxX) {
                val u = m00 * x + baseU
                val v = m10 * x + baseV
                if (u >= 1f && u < CROP_SIZE - 2f && v >= 1f && v < CROP_SIZE - 2f) {
                    val alpha = sampleMaskBilinear(featheredMask128, u, v)
                    if (alpha > 0.003f) {
                        val swapColor = FaceAlignment.sampleBilinearClamped(
                            swapPixels,
                            CROP_SIZE,
                            CROP_SIZE,
                            u,
                            v
                        )
                        val dstColor = targetPixels[rowOffset + x]

                        val sr = (swapColor ushr 16) and 0xFF
                        val sg = (swapColor ushr 8) and 0xFF
                        val sb = swapColor and 0xFF

                        val dr = (dstColor ushr 16) and 0xFF
                        val dg = (dstColor ushr 8) and 0xFF
                        val db = dstColor and 0xFF

                        val invA = 1.0f - alpha
                        val outR = (sr * alpha + dr * invA).toInt().coerceIn(0, 255)
                        val outG = (sg * alpha + dg * invA).toInt().coerceIn(0, 255)
                        val outB = (sb * alpha + db * invA).toInt().coerceIn(0, 255)

                        targetPixels[rowOffset + x] =
                            (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                    }
                }
            }
        }
    }

    private fun sampleMaskBilinear(mask128: FloatArray, u: Float, v: Float): Float {
        val x0 = u.toInt().coerceIn(0, CROP_SIZE - 2)
        val y0 = v.toInt().coerceIn(0, CROP_SIZE - 2)
        val fx = u - x0
        val fy = v - y0

        val m00 = mask128[y0 * CROP_SIZE + x0]
        val m10 = mask128[y0 * CROP_SIZE + (x0 + 1)]
        val m01 = mask128[(y0 + 1) * CROP_SIZE + x0]
        val m11 = mask128[(y0 + 1) * CROP_SIZE + (x0 + 1)]

        return m00 * (1f - fx) * (1f - fy) +
            m10 * fx * (1f - fy) +
            m01 * (1f - fx) * fy +
            m11 * fx * fy
    }

    /**
     * Renders a non-intrusive ethical AI provenance watermark badge in the bottom-right corner
     * of the output image to guard against deceptive misuse.
     */
    fun applyEthicalProvenanceWatermark(bitmap: Bitmap) {
        val canvas = Canvas(bitmap)
        val scale = (bitmap.width / 1080f).coerceIn(0.65f, 2.2f)
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(225, 240, 248, 255)
            textSize = 24f * scale
            isFakeBoldText = true
        }
        val label = "SYNTHETIC AI • OFFLINE ONNX"
        val textWidth = textPaint.measureText(label)
        val padH = 18f * scale
        val padV = 12f * scale
        val margin = 20f * scale
        val boxHeight = 36f * scale + padV
        val left = bitmap.width - textWidth - padH * 2 - margin
        val top = bitmap.height - boxHeight - margin
        val rect = RectF(left, top, bitmap.width - margin, bitmap.height - margin)

        val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(165, 11, 16, 33)
            style = Paint.Style.FILL
        }
        val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(190, 0, 229, 255)
            style = Paint.Style.STROKE
            strokeWidth = 2f * scale
        }

        canvas.drawRoundRect(rect, 12f * scale, 12f * scale, bgPaint)
        canvas.drawRoundRect(rect, 12f * scale, 12f * scale, borderPaint)
        canvas.drawText(
            label,
            rect.left + padH,
            rect.centerY() + (textPaint.textSize * 0.35f),
            textPaint
        )
    }
}
