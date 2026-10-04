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
     * aperture (sclera, iris, pupil, eyelids) in [0.0f, 1.0f].
     * Used to protect eyes from skin-tone color shifts that would otherwise turn dark pupils
     * milky grey or tint white sclera into a negative-looking eye.
     */
    private fun computeEyeProtectionWeight128(x: Int, y: Int): Float {
        val lx = (x - LEFT_EYE_X) / 16.0f
        val ly = (y - LEFT_EYE_Y) / 9.5f
        val lDistSq = lx * lx + ly * ly

        val rx = (x - RIGHT_EYE_X) / 16.0f
        val ry = (y - RIGHT_EYE_Y) / 9.5f
        val rDistSq = rx * rx + ry * ry

        val minDistSq = min(lDistSq, rDistSq)
        return when {
            minDistSq <= 0.45f -> 1.0f
            minDistSq >= 1.0f -> 0.0f
            else -> {
                val t = (minDistSq - 0.45f) / 0.55f
                0.5f * (1.0f + cos(Math.PI * t).toFloat())
            }
        }
    }

    /**
     * Restores natural eye polarity (deep black pupil, rich dark iris, clean white sclera, and
     * crisp eyelashes) on `swapped128` using `alignedTarget128` and `alignedSource112` as
     * structural and radiometric references.
     *
     * Specifically eliminates the "negative eye" / hollow iris artifact where `inswapper_128`
     * outputs inverted grey/white values inside the iris/pupil or darkens the outer sclera on
     * tilted or heavily-lined eyes.
     */
    fun restoreEyesAndEliminateNegativeArtifacts128(
        swapped128: Bitmap,
        alignedTarget128: Bitmap,
        alignedSource112: Bitmap? = null
    ): Bitmap {
        val total = CROP_SIZE * CROP_SIZE
        val swapPx = IntArray(total)
        val tgtPx = IntArray(total)
        swapped128.getPixels(swapPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        alignedTarget128.getPixels(tgtPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)

        val srcPx112: IntArray? = alignedSource112?.let { bmp ->
            if (bmp.width == 112 && bmp.height == 112) {
                IntArray(112 * 112).also { bmp.getPixels(it, 0, 112, 0, 0, 112, 112) }
            } else {
                null
            }
        }

        val outPx = swapPx.copyOf()

        // Process both Left Eye (46.3, 51.7) and Right Eye (81.5, 51.5)
        val eyeCenters = arrayOf(
            LEFT_EYE_X to LEFT_EYE_Y,
            RIGHT_EYE_X to RIGHT_EYE_Y
        )

        for ((eyeCx, eyeCy) in eyeCenters) {
            val xMin = (eyeCx - 16f).toInt().coerceAtLeast(1)
            val xMax = (eyeCx + 16f).toInt().coerceAtMost(CROP_SIZE - 2)
            val yMin = (eyeCy - 10f).toInt().coerceAtLeast(1)
            val yMax = (eyeCy + 10f).toInt().coerceAtMost(CROP_SIZE - 2)

            for (y in yMin..yMax) {
                val dy = y - eyeCy
                val ny = dy / 9.0f
                for (x in xMin..xMax) {
                    val dx = x - eyeCx
                    val nx = dx / 15.5f
                    val normDist = sqrt(nx * nx + ny * ny)
                    if (normDist >= 1.0f) continue

                    // Smooth radial weight inside the eye socket
                    val socketWeight = if (normDist <= 0.55f) {
                        1.0f
                    } else {
                        val t = (normDist - 0.55f) / 0.45f
                        0.5f * (1.0f + cos(Math.PI * t).toFloat())
                    }

                    val idx = y * CROP_SIZE + x
                    val sc = swapPx[idx]
                    val tc = tgtPx[idx]

                    val sR = (sc ushr 16) and 0xFF
                    val sG = (sc ushr 8) and 0xFF
                    val sB = sc and 0xFF
                    val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB

                    val tR = (tc ushr 16) and 0xFF
                    val tG = (tc ushr 8) and 0xFF
                    val tB = tc and 0xFF
                    val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                    // Sample corresponding source eye pixel in 112x112 space (where x_112 = x_128 - 8, y_112 = y_128)
                    var srcLum = tLum
                    var srcR = tR
                    var srcG = tG
                    var srcB = tB
                    if (srcPx112 != null) {
                        val sx112 = (x - 8).coerceIn(0, 111)
                        val sy112 = y.coerceIn(0, 111)
                        val srcC = srcPx112[sy112 * 112 + sx112]
                        srcR = (srcC ushr 16) and 0xFF
                        srcG = (srcC ushr 8) and 0xFF
                        srcB = srcC and 0xFF
                        srcLum = 0.299f * srcR + 0.587f * srcG + 0.114f * srcB
                    }

                    // Reference eye luminance combining target gaze/eyeliner structure (65%) + source eye tone (35%)
                    val refLum = 0.65f * tLum + 0.35f * srcLum
                    val refR = 0.65f * tR + 0.35f * srcR
                    val refG = 0.65f * tG + 0.35f * srcG
                    val refB = 0.65f * tB + 0.35f * srcB

                    // 1. Detect negative/inverted eye artifacts:
                    //    (a) Hollow/milky iris or eyelid: swapped pixel is anomalously brighter than reference dark iris/lashes
                    //    (b) Muddy/darkened sclera: swapped pixel is anomalously darker than reference white sclera
                    val lumDiff = abs(sLum - refLum)
                    val isIrisOrLashInversion = (tLum < 95f || srcLum < 95f) && sLum > refLum + 10f
                    val isScleraDarkening = (tLum > 125f) && sLum < refLum - 12f

                    // Compute adaptive correction blend: stronger when polarity inversion is detected
                    val baseEyeAnchor = 0.42f * socketWeight
                    val anomalyBoost = when {
                        isIrisOrLashInversion -> ((sLum - refLum) / 55f).coerceIn(0.25f, 0.88f) * socketWeight
                        isScleraDarkening -> ((refLum - sLum) / 55f).coerceIn(0.25f, 0.85f) * socketWeight
                        lumDiff > 22f -> ((lumDiff - 22f) / 70f).coerceIn(0f, 0.65f) * socketWeight
                        else -> 0f
                    }
                    val corrWeight = max(baseEyeAnchor, anomalyBoost).coerceIn(0f, 0.90f)

                    // Local 3x3 high-frequency detail from target eye (preserves crisp iris rim, catchlight & lashes)
                    val tNeighborsAvgLum = compute3x3LuminanceAvg(tgtPx, CROP_SIZE, x, y)
                    val highFreqDetail = (tLum - tNeighborsAvgLum) * 0.55f * socketWeight

                    val correctedR = (sR * (1f - corrWeight) + refR * corrWeight + highFreqDetail).toInt().coerceIn(0, 255)
                    val correctedG = (sG * (1f - corrWeight) + refG * corrWeight + highFreqDetail).toInt().coerceIn(0, 255)
                    val correctedB = (sB * (1f - corrWeight) + refB * corrWeight + highFreqDetail).toInt().coerceIn(0, 255)

                    outPx[idx] = (0xFF shl 24) or (correctedR shl 16) or (correctedG shl 8) or correctedB
                }
            }
        }

        // Mild unsharp pass across the inner face (eyes, nose, lips) so 128x128 synthesis is crisp
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

                val amount = 0.26f
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

    private fun compute3x3LuminanceAvg(px: IntArray, stride: Int, cx: Int, cy: Int): Float {
        var sum = 0f
        for (dy in -1..1) {
            val row = (cy + dy) * stride
            for (dx in -1..1) {
                val c = px[row + cx + dx]
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                sum += 0.299f * r + 0.587f * g + 0.114f * b
            }
        }
        return sum / 9.0f
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
