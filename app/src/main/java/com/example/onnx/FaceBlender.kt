package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Controls whose skin tone is applied to the swapped face or head:
 *  - TARGET_SCENE: Matches the Target scene person's skin tone and scene lighting (best when blending onto target's neck/body).
 *  - SOURCE_IDENTITY: Preserves the Source identity person's authentic skin tone (chrominance + base skin tone from source photo)
 *    while adapting gentle lighting contrast so it blends cleanly without looking flat.
 *  - BALANCED_BLEND: 50% Source identity skin tone + 50% Target scene skin tone.
 */
enum class SkinToneSourceMode(
    val title: String,
    val tamilTitle: String,
    val tamilSubtitle: String
) {
    TARGET_SCENE(
        title = "Target Scene Skin Tone",
        tamilTitle = "Target படத்தின் தோல் நிறம்",
        tamilSubtitle = "Target போட்டோவில் உள்ளவரின் உடல்/கழுத்து நிறம் மற்றும் வெளிச்சத்திற்கு ஏற்ப பொருந்தும்"
    ),
    SOURCE_IDENTITY(
        title = "Source Identity Skin Tone",
        tamilTitle = "Source முகத்தின் தோல் நிறம்",
        tamilSubtitle = "Source (மாற்றும் நபரின்) சொந்த தோல் நிறத்தை அப்படியே தக்கவைக்கும்"
    ),
    BALANCED_BLEND(
        title = "50/50 Balanced Skin Tone",
        tamilTitle = "இரண்டும் கலந்த நிறம் (50/50)",
        tamilSubtitle = "Source மற்றும் Target இருவரின் தோல் நிறத்தையும் சமமாகக் கலந்து இயற்கையாக மாற்றும்"
    )
}

/**
 * Controls whose facial reaction (smile, visible teeth, visible tongue, mouth expression & eye reaction) is used:
 *  - TARGET_REACTION: Uses the Target scene person's facial reaction (smile, open mouth, visible teeth, visible tongue)
 *    and restores crisp 512x512 teeth/tongue detail from the target photo so teeth/tongue never look blurry.
 *  - SOURCE_REACTION: Transfers the Source identity person's own facial reaction (their smile, visible teeth, visible tongue,
 *    lip shape, and eye expression) directly from the Source photo onto the Target pose!
 */
enum class FaceReactionSourceMode(
    val title: String,
    val tamilTitle: String,
    val tamilSubtitle: String
) {
    TARGET_REACTION(
        title = "Target Photo Reaction",
        tamilTitle = "Target முகபாவனை (சிரிப்பு / பற்கள் / நாக்கு)",
        tamilSubtitle = "Target படத்தில் உள்ளவரின் சிரிப்பு, தெரியும் பற்கள் (Teeth), நாக்கு (Tongue) & முகபாவனையைப் பயன்படுத்தும்"
    ),
    SOURCE_REACTION(
        title = "Source Face Reaction",
        tamilTitle = "Source முகபாவனை (சிரிப்பு / பற்கள் / நாக்கு)",
        tamilSubtitle = "Source படத்தில் உள்ளவரின் சொந்த சிரிப்பு, பற்கள் (Teeth), நாக்கு (Tongue) & கண் பாவனையைக் கொண்டுவரும்"
    )
}

/**
 * Studio-Grade Artifact-Free & Online-HD Face Blending Engine:
 *  1. Zero Source-Pixel Ghosting (eliminates cheek hair-strand stamping and eyebrow gaps)
 *  2. Dynamic Landmark-Guided Right & Left Eye + Eyebrow Continuity Restoration (`M * landmarks5`)
 *  3. Cheek Scratch/Glitch Healer & Foreground Hair/Jewelry Occlusion Guard
 *  4. Direct 512x512 Online-Style HD Super-Resolution (via `gfpgan_1.4.onnx` or Built-in Guided Pore/Detail Transfer)
 *  5. Selectable Skin Tone Source (Target Scene vs Source Identity vs 50/50 Blend)
 *  6. Selectable Face Reaction Source (Target Smile/Teeth/Tongue vs Source Smile/Teeth/Tongue)
 */
object FaceBlender {

    private const val CROP_SIZE = 128
    private const val HD_SIZE = 512

    // Default canonical 128x128 coordinates (used when dynamic landmarks are not passed)
    private const val DEFAULT_LEFT_EYE_X = 46.2946f
    private const val DEFAULT_LEFT_EYE_Y = 51.6963f
    private const val DEFAULT_RIGHT_EYE_X = 81.5318f
    private const val DEFAULT_RIGHT_EYE_Y = 51.5014f

    private data class WarpedFaceGeometry(
        val leftEye: PointF,
        val rightEye: PointF,
        val nose: PointF,
        val leftMouth: PointF,
        val rightMouth: PointF,
        val eyeDist: Float,
        val cosA: Float,
        val sinA: Float
    )

    private fun computeWarpedGeometry(
        forwardMatrix128: FloatArray?,
        targetLandmarks5: List<PointF>?,
        scale: Float = 1.0f
    ): WarpedFaceGeometry {
        if (forwardMatrix128 != null && targetLandmarks5 != null && targetLandmarks5.size >= 5) {
            fun mapPt(pt: PointF): PointF {
                val u = (forwardMatrix128[0] * pt.x + forwardMatrix128[1] * pt.y + forwardMatrix128[2]) * scale
                val v = (forwardMatrix128[3] * pt.x + forwardMatrix128[4] * pt.y + forwardMatrix128[5]) * scale
                return PointF(u, v)
            }
            val lEye = mapPt(targetLandmarks5[0])
            val rEye = mapPt(targetLandmarks5[1])
            val nose = mapPt(targetLandmarks5[2])
            val lMouth = mapPt(targetLandmarks5[3])
            val rMouth = mapPt(targetLandmarks5[4])
            val dx = rEye.x - lEye.x
            val dy = rEye.y - lEye.y
            val dist = hypot(dx, dy).coerceAtLeast(24.0f * scale)
            return WarpedFaceGeometry(
                leftEye = lEye,
                rightEye = rEye,
                nose = nose,
                leftMouth = lMouth,
                rightMouth = rMouth,
                eyeDist = dist,
                cosA = dx / dist,
                sinA = dy / dist
            )
        }
        val lEye = PointF(DEFAULT_LEFT_EYE_X * scale, DEFAULT_LEFT_EYE_Y * scale)
        val rEye = PointF(DEFAULT_RIGHT_EYE_X * scale, DEFAULT_RIGHT_EYE_Y * scale)
        val dx = rEye.x - lEye.x
        val dy = rEye.y - lEye.y
        val dist = hypot(dx, dy).coerceAtLeast(24.0f * scale)
        return WarpedFaceGeometry(
            leftEye = lEye,
            rightEye = rEye,
            nose = PointF(64.0252f * scale, 71.7366f * scale),
            leftMouth = PointF(49.5493f * scale, 92.3655f * scale),
            rightMouth = PointF(78.7299f * scale, 92.2041f * scale),
            eyeDist = dist,
            cosA = dx / dist,
            sinA = dy / dist
        )
    }

    /**
     * Oriented distance squared from `(x, y)` to an ellipse centered at `(cx, cy)` rotated by `(cosA, sinA)`.
     */
    private fun orientedEllipseDistSq(
        x: Float,
        y: Float,
        cx: Float,
        cy: Float,
        rx: Float,
        ry: Float,
        cosA: Float,
        sinA: Float
    ): Float {
        val dx = x - cx
        val dy = y - cy
        val u = (dx * cosA + dy * sinA) / rx
        val v = (-dx * sinA + dy * cosA) / ry
        return u * u + v * v
    }

    /**
     * Returns how strongly a pixel (x, y) in 128x128 canonical space belongs to either ocular
     * socket or eyebrow arch in [0.0f, 1.0f], using dynamic warped landmark coordinates.
     */
    private fun computeEyeAndBrowProtectionWeight(
        x: Float,
        y: Float,
        geom: WarpedFaceGeometry
    ): Float {
        val eyeRx = geom.eyeDist * 0.58f
        val eyeRy = geom.eyeDist * 0.38f

        val lDistSq = orientedEllipseDistSq(
            x, y, geom.leftEye.x, geom.leftEye.y, eyeRx, eyeRy, geom.cosA, geom.sinA
        )
        val rDistSq = orientedEllipseDistSq(
            x, y, geom.rightEye.x, geom.rightEye.y, eyeRx, eyeRy, geom.cosA, geom.sinA
        )

        // Eyebrow centers lie just above the eyes along the perpendicular normal (-sinA, cosA)
        val browOffset = geom.eyeDist * 0.30f
        val lBrowX = geom.leftEye.x + geom.sinA * browOffset
        val lBrowY = geom.leftEye.y - geom.cosA * browOffset
        val rBrowX = geom.rightEye.x + geom.sinA * browOffset
        val rBrowY = geom.rightEye.y - geom.cosA * browOffset
        val browRx = geom.eyeDist * 0.60f
        val browRy = geom.eyeDist * 0.22f

        val lbDistSq = orientedEllipseDistSq(
            x, y, lBrowX, lBrowY, browRx, browRy, geom.cosA, geom.sinA
        )
        val rbDistSq = orientedEllipseDistSq(
            x, y, rBrowX, rBrowY, browRx, browRy, geom.cosA, geom.sinA
        )

        val minDistSq = min(min(lDistSq, rDistSq), min(lbDistSq, rbDistSq))
        return when {
            minDistSq <= 0.48f -> 1.0f
            minDistSq >= 1.0f -> 0.0f
            else -> {
                val t = (minDistSq - 0.48f) / 0.52f
                0.5f * (1.0f + cos(Math.PI * t).toFloat())
            }
        }
    }

    /**
     * Eliminates AI artifacts, cheek hair-strand lines, eyebrow gaps, and cloudy/negative eyes on
     * `swapped128` WITHOUT stamping 2D source background/hair pixels onto the target cheeks or forehead.
     */
    fun restoreEyesAndEliminateNegativeArtifacts128(
        swapped128: Bitmap,
        alignedTarget128: Bitmap,
        alignedSource112: Bitmap? = null,
        alignedSource128: Bitmap? = null,
        hasTrueArcFaceLatent: Boolean = true,
        forwardMatrix128: FloatArray? = null,
        targetLandmarks5: List<PointF>? = null
    ): Bitmap {
        val total = CROP_SIZE * CROP_SIZE
        val swapPx = IntArray(total)
        val tgtPx = IntArray(total)
        swapped128.getPixels(swapPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        alignedTarget128.getPixels(tgtPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)

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

        val geom = computeWarpedGeometry(forwardMatrix128, targetLandmarks5, scale = 1.0f)

        // Step 1: Zero Source-Pixel Ghosting Base Synthesis
        // - When hasTrueArcFaceLatent == true: use inswapper_128's neural pose-aligned synthesis directly,
        //   removing ONLY local cheek scratches/glitches or negative luminance inversions using target shading.
        //   (NEVER stamp raw 2D source cheek/hair pixels onto the target face!)
        // - When hasTrueArcFaceLatent == false (2-Model fallback): use occlusion-filtered inner feature transfer
        //   that strictly rejects any dark hair strands on cheeks or bright forehead gaps over eyebrows.
        val outPx = if (hasTrueArcFaceLatent) {
            healCheekScratchesAndNegativeAnomalies128(swapPx, tgtPx, geom)
        } else {
            buildGlitchFreeTwoModelFallback128(srcPx128, tgtPx, geom)
        }

        // Step 2: Dynamic Landmark Eyebrow Continuity Guard (Eliminates white/bright gaps across eyebrows)
        restoreDynamicEyebrowContinuity(
            outPx = outPx,
            tgtPx = tgtPx,
            size = CROP_SIZE,
            geom = geom
        )

        // Step 3: Dynamic Landmark Bilateral Eye Restoration (Eliminates cloudy/white/negative right or left eye)
        restoreDynamicOcularSockets(
            outPx = outPx,
            tgtPx = tgtPx,
            srcPx = srcPx128,
            size = CROP_SIZE,
            geom = geom
        )

        val out = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Bitmap.Config.ARGB_8888)
        out.setPixels(outPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        return out
    }

    /**
     * Removes any isolated dark vertical hair/scratch glitches on the cheeks and corrects
     * any negative luminance inversion on `swapPx` using ONLY `tgtPx` (the target face in exact pose),
     * guaranteeing ZERO ghosting of source hair strands onto the target cheeks!
     */
    private fun healCheekScratchesAndNegativeAnomalies128(
        swapPx: IntArray,
        tgtPx: IntArray,
        geom: WarpedFaceGeometry
    ): IntArray {
        val total = CROP_SIZE * CROP_SIZE
        val out = swapPx.copyOf()

        for (y in 4 until CROP_SIZE - 4) {
            val row = y * CROP_SIZE
            for (x in 4 until CROP_SIZE - 4) {
                val idx = row + x
                val eyeBrowWeight = computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom)
                if (eyeBrowWeight > 0.25f) continue

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

                // Compute smooth 7x7 skin neighborhood on swapped & target crops
                val sSmooth = computeWideSkinSmoothRGB(swapPx, CROP_SIZE, x, y, radius = 3)
                val tSmooth = computeWideSkinSmoothRGB(tgtPx, CROP_SIZE, x, y, radius = 3)

                val sSmoothLum = 0.299f * sSmooth[0] + 0.587f * sSmooth[1] + 0.114f * sSmooth[2]
                val tSmoothLum = 0.299f * tSmooth[0] + 0.587f * tSmooth[1] + 0.114f * tSmooth[2]

                // 1. Detect isolated dark line/scratch/hair artifact on smooth target cheek skin:
                //    Target skin is smooth (|tLum - tSmoothLum| < 12, tLum > 75),
                //    but swapped pixel has a sharp dark dip or high horizontal jump.
                val isTargetSmoothSkin = tLum > 75f && abs(tLum - tSmoothLum) < 12f
                val isSwappedDarkScratch = isTargetSmoothSkin && (sLum < sSmoothLum - 9f || sLum < tSmoothLum - 32f)

                // 2. Detect local negative contrast inversion between swapped and target shading
                val sAvg3 = compute3x3LuminanceAvg(swapPx, CROP_SIZE, x, y)
                val tAvg3 = compute3x3LuminanceAvg(tgtPx, CROP_SIZE, x, y)
                val dSwap = sLum - sAvg3
                val dTgt = tLum - tAvg3
                val isContrastInverted = (dSwap * dTgt) < -3.0f && isTargetSmoothSkin

                if (isSwappedDarkScratch || isContrastInverted) {
                    // Heal scratch with smooth local swapped skin + subtle target skin micro-gradient
                    val healR = (sSmooth[0] + (tR - tSmooth[0]) * 0.65f).toInt().coerceIn(0, 255)
                    val healG = (sSmooth[1] + (tG - tSmooth[1]) * 0.65f).toInt().coerceIn(0, 255)
                    val healB = (sSmooth[2] + (tB - tSmooth[2]) * 0.65f).toInt().coerceIn(0, 255)
                    out[idx] = (0xFF shl 24) or (healR shl 16) or (healG shl 8) or healB
                }
            }
        }
        return out
    }

    /**
     * Used ONLY when `w600k_r50.onnx` is not installed (2-Model fallback mode).
     * Transfers inner facial features from `srcPx128` onto `tgtPx128`'s illumination envelope while
     * strictly rejecting any dark hair strands on the cheeks or bright skin gaps over eyebrows.
     */
    private fun buildGlitchFreeTwoModelFallback128(
        srcPx128: IntArray,
        tgtPx128: IntArray,
        geom: WarpedFaceGeometry
    ): IntArray {
        val total = CROP_SIZE * CROP_SIZE
        val out = IntArray(total)
        val cx = (geom.leftEye.x + geom.rightEye.x) * 0.5f
        val cy = (geom.nose.y + (geom.leftEye.y + geom.rightEye.y) * 0.5f) * 0.5f
        val rx = geom.eyeDist * 0.92f
        val ry = geom.eyeDist * 1.12f

        for (y in 0 until CROP_SIZE) {
            val row = y * CROP_SIZE
            for (x in 0 until CROP_SIZE) {
                val idx = row + x
                val tc = tgtPx128[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                // Tight inner-features weight so outer cheeks and forehead stay 100% free of source hair strands
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(), cx, cy, rx, ry, geom.cosA, geom.sinA
                )
                val innerWeight = when {
                    dSq >= 1.0f -> 0.0f
                    dSq <= 0.45f -> 0.65f
                    else -> {
                        val t = (dSq - 0.45f) / 0.55f
                        0.65f * 0.5f * (1.0f + cos(Math.PI * t).toFloat())
                    }
                }

                if (innerWeight <= 0.01f) {
                    out[idx] = tc
                    continue
                }

                val sSmooth = computeWideSkinSmoothRGB(srcPx128, CROP_SIZE, x, y, radius = 5)
                val tSmooth = computeWideSkinSmoothRGB(tgtPx128, CROP_SIZE, x, y, radius = 5)
                val sc = srcPx128[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF
                val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB
                val sSmoothLum = 0.299f * sSmooth[0] + 0.587f * sSmooth[1] + 0.114f * sSmooth[2]

                // Reject source hair strands on cheeks: if source has a sharp dark strand on smooth target skin, ignore source detail
                val isSourceHairArtifact = (sLum < sSmoothLum - 10f || sLum < 65f) && tLum > 85f
                val detailScale = if (isSourceHairArtifact) 0.0f else innerWeight

                val hR = (tSmooth[0] + (sR - sSmooth[0]) * detailScale + (tR - tSmooth[0]) * (1f - detailScale)).toInt().coerceIn(0, 255)
                val hG = (tSmooth[1] + (sG - sSmooth[1]) * detailScale + (tG - tSmooth[1]) * (1f - detailScale)).toInt().coerceIn(0, 255)
                val hB = (tSmooth[2] + (sB - sSmooth[2]) * detailScale + (tB - tSmooth[2]) * (1f - detailScale)).toInt().coerceIn(0, 255)

                out[idx] = (0xFF shl 24) or (hR shl 16) or (hG shl 8) or hB
            }
        }
        return out
    }

    /**
     * Protects both Left and Right Eyebrows from artificial white gaps, cuts, or double-eyebrow ghosting
     * using dynamic landmark-guided eyebrow arches.
     */
    private fun restoreDynamicEyebrowContinuity(
        outPx: IntArray,
        tgtPx: IntArray,
        size: Int,
        geom: WarpedFaceGeometry
    ) {
        val browOffset = geom.eyeDist * 0.29f
        val browCenters = arrayOf(
            PointF(geom.leftEye.x + geom.sinA * browOffset, geom.leftEye.y - geom.cosA * browOffset),
            PointF(geom.rightEye.x + geom.sinA * browOffset, geom.rightEye.y - geom.cosA * browOffset)
        )
        val browRx = geom.eyeDist * 0.62f
        val browRy = geom.eyeDist * 0.24f
        val boundR = max(browRx, browRy) * 1.15f

        for (browCenter in browCenters) {
            val xMin = (browCenter.x - boundR).toInt().coerceAtLeast(1)
            val xMax = (browCenter.x + boundR).toInt().coerceAtMost(size - 2)
            val yMin = (browCenter.y - boundR).toInt().coerceAtLeast(1)
            val yMax = (browCenter.y + boundR).toInt().coerceAtMost(size - 2)

            for (y in yMin..yMax) {
                for (x in xMin..xMax) {
                    val dSq = orientedEllipseDistSq(
                        x.toFloat(), y.toFloat(),
                        browCenter.x, browCenter.y,
                        browRx, browRy,
                        geom.cosA, geom.sinA
                    )
                    if (dSq >= 1.0f) continue

                    val normDist = sqrt(dSq)
                    val archWeight = if (normDist <= 0.60f) {
                        1.0f
                    } else {
                        val t = (normDist - 0.60f) / 0.40f
                        0.5f * (1.0f + cos(Math.PI * t).toFloat())
                    }

                    val idx = y * size + x
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

                    // If target pixel is part of the dark eyebrow arch (tLum < 115) or swapped pixel has a bright gap (cLum > tLum + 5),
                    // anchor strongly to the continuous eyebrow arch so no white cut/gap can ever appear!
                    val isBrowHair = tLum < 115f
                    val hasBrightGapGlitch = cLum > tLum + 4f
                    val browBlend = when {
                        isBrowHair && hasBrightGapGlitch -> (0.88f * archWeight).coerceIn(0f, 0.94f)
                        isBrowHair -> (0.72f * archWeight).coerceIn(0f, 0.85f)
                        else -> (0.45f * archWeight).coerceIn(0f, 0.60f)
                    }

                    val finalR = (cR * (1f - browBlend) + tR * browBlend).toInt().coerceIn(0, 255)
                    val finalG = (cG * (1f - browBlend) + tG * browBlend).toInt().coerceIn(0, 255)
                    val finalB = (cB * (1f - browBlend) + tB * browBlend).toInt().coerceIn(0, 255)

                    outPx[idx] = (0xFF shl 24) or (finalR shl 16) or (finalG shl 8) or finalB
                }
            }
        }
    }

    /**
     * Restores crystal-clear, positive-polarity Left & Right Eyes centered at the exact dynamic
     * warped landmarks `geom.leftEye` and `geom.rightEye` (supporting both 128x128 and 512x512 HD).
     */
    private fun restoreDynamicOcularSockets(
        outPx: IntArray,
        tgtPx: IntArray,
        srcPx: IntArray?,
        size: Int,
        geom: WarpedFaceGeometry
    ) {
        val eyeCenters = arrayOf(geom.leftEye, geom.rightEye)
        val rx = geom.eyeDist * 0.58f
        val ry = geom.eyeDist * 0.38f
        val boundR = max(rx, ry) * 1.15f

        for (eyeCenter in eyeCenters) {
            val xMin = (eyeCenter.x - boundR).toInt().coerceAtLeast(1)
            val xMax = (eyeCenter.x + boundR).toInt().coerceAtMost(size - 2)
            val yMin = (eyeCenter.y - boundR).toInt().coerceAtLeast(1)
            val yMax = (eyeCenter.y + boundR).toInt().coerceAtMost(size - 2)

            // Subtle iris chrominance shift from source eye (only when source is provided at matching size)
            var chromaShiftR = 0f
            var chromaShiftG = 0f
            var chromaShiftB = 0f
            if (srcPx != null && srcPx.size == size * size) {
                var sEr = 0f
                var sEg = 0f
                var sEb = 0f
                var tEr = 0f
                var tEg = 0f
                var tEb = 0f
                var samples = 0
                val irRadius = (geom.eyeDist * 0.16f).toInt().coerceAtLeast(3)
                val cxI = eyeCenter.x.toInt()
                val cyI = eyeCenter.y.toInt()
                for (iy in (cyI - irRadius)..(cyI + irRadius)) {
                    for (ix in (cxI - irRadius)..(cxI + irRadius)) {
                        if (iy !in 1 until size - 1 || ix !in 1 until size - 1) continue
                        val sc = srcPx[iy * size + ix]
                        val tc = tgtPx[iy * size + ix]
                        val sL = 0.299f * (sc ushr 16 and 0xFF) + 0.587f * (sc ushr 8 and 0xFF) + 0.114f * (sc and 0xFF)
                        val tL = 0.299f * (tc ushr 16 and 0xFF) + 0.587f * (tc ushr 8 and 0xFF) + 0.114f * (tc and 0xFF)
                        if (sL < 95f && tL < 95f) {
                            sEr += (sc ushr 16) and 0xFF
                            sEg += (sc ushr 8) and 0xFF
                            sEb += sc and 0xFF
                            tEr += (tc ushr 16) and 0xFF
                            tEg += (tc ushr 8) and 0xFF
                            tEb += tc and 0xFF
                            samples++
                        }
                    }
                }
                if (samples > 4) {
                    chromaShiftR = ((sEr - tEr) / samples).coerceIn(-14f, 14f)
                    chromaShiftG = ((sEg - tEg) / samples).coerceIn(-14f, 14f)
                    chromaShiftB = ((sEb - tEb) / samples).coerceIn(-14f, 14f)
                }
            }

            for (y in yMin..yMax) {
                for (x in xMin..xMax) {
                    val dSq = orientedEllipseDistSq(
                        x.toFloat(), y.toFloat(),
                        eyeCenter.x, eyeCenter.y,
                        rx, ry,
                        geom.cosA, geom.sinA
                    )
                    if (dSq >= 1.0f) continue

                    val normDist = sqrt(dSq)
                    // 1.0 across the entire ocular aperture (pupil, iris, sclera, lash line: normDist <= 0.68)
                    val socketWeight = if (normDist <= 0.68f) {
                        1.0f
                    } else {
                        val t = (normDist - 0.68f) / 0.32f
                        0.5f * (1.0f + cos(Math.PI * t).toFloat())
                    }

                    val idx = y * size + x
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

                    val isDarkIrisOrLash = tLum < 105f && normDist < 0.65f
                    val cleanR = (tR + if (isDarkIrisOrLash) chromaShiftR * 0.35f else 0f).coerceIn(0f, 255f)
                    val cleanG = (tG + if (isDarkIrisOrLash) chromaShiftG * 0.35f else 0f).coerceIn(0f, 255f)
                    val cleanB = (tB + if (isDarkIrisOrLash) chromaShiftB * 0.35f else 0f).coerceIn(0f, 255f)
                    val cleanLum = 0.299f * cleanR + 0.587f * cleanG + 0.114f * cleanB

                    // Strong 0.90..0.97 ocular anchor inside the eye socket so neither eye EVER looks cloudy, white, or negative
                    val isNegativeIris = tLum < 120f && cLum > cleanLum + 3f
                    val isNegativeSclera = tLum > 125f && cLum < cleanLum - 5f
                    val baseAnchor = 0.88f * socketWeight
                    val eyeBlend = when {
                        isNegativeIris || isNegativeSclera -> (0.96f * socketWeight).coerceIn(0f, 0.98f)
                        else -> baseAnchor.coerceIn(0f, 0.95f)
                    }

                    var eR = cR * (1f - eyeBlend) + cleanR * eyeBlend
                    var eG = cG * (1f - eyeBlend) + cleanG * eyeBlend
                    var eB = cB * (1f - eyeBlend) + cleanB * eyeBlend

                    // Strict Positive Polarity Clamp inside core ocular aperture (normDist <= 0.78)
                    if (normDist <= 0.78f) {
                        val eLum = 0.299f * eR + 0.587f * eG + 0.114f * eB
                        if (tLum < 105f && eLum > cleanLum + 1.5f && eLum > 1f) {
                            val scaleDown = (cleanLum / eLum).coerceIn(0.10f, 1.0f)
                            eR *= scaleDown
                            eG *= scaleDown
                            eB *= scaleDown
                        } else if (tLum > 128f && eLum < cleanLum * 0.95f && eLum > 1f) {
                            val scaleUp = ((cleanLum * 0.96f) / eLum).coerceIn(1.0f, 2.4f)
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
    }

    private fun computeWideSkinSmoothRGB(
        px: IntArray,
        stride: Int,
        cx: Int,
        cy: Int,
        radius: Int
    ): FloatArray {
        val x0 = (cx - radius).coerceAtLeast(0)
        val x1 = (cx + radius).coerceAtMost(stride - 1)
        val y0 = (cy - radius).coerceAtLeast(0)
        val y1 = (cy + radius).coerceAtMost(stride - 1)
        var rSum = 0f
        var gSum = 0f
        var bSum = 0f
        var count = 0
        for (y in y0..y1) {
            val row = y * stride
            for (x in x0..x1) {
                val c = px[row + x]
                rSum += (c ushr 16) and 0xFF
                gSum += (c ushr 8) and 0xFF
                bSum += c and 0xFF
                count++
            }
        }
        val inv = 1f / count.coerceAtLeast(1)
        return floatArrayOf(rSum * inv, gSum * inv, bSum * inv)
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
     * Harmonizes the color and luminance statistics of `swapped128` according to [skinToneMode]
     * (`TARGET_SCENE`, `SOURCE_IDENTITY`, or `BALANCED_BLEND`) strictly on SKIN regions while
     * protecting both eyes and eyebrows from washout.
     */
    fun transferSkinToneStatistics128(
        swapped128: Bitmap,
        targetCrop128: Bitmap,
        sourceCrop128: Bitmap? = null,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.TARGET_SCENE,
        strength: Float = 0.72f,
        forwardMatrix128: FloatArray? = null,
        targetLandmarks5: List<PointF>? = null
    ): Bitmap {
        val total = CROP_SIZE * CROP_SIZE
        val swapPx = IntArray(total)
        val tgtPx = IntArray(total)
        val srcPx = if (sourceCrop128 != null) IntArray(total) else null
        swapped128.getPixels(swapPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        targetCrop128.getPixels(tgtPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        sourceCrop128?.getPixels(srcPx!!, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)

        val geom = computeWarpedGeometry(forwardMatrix128, targetLandmarks5, scale = 1.0f)

        var sRMean = 0.0
        var sGMean = 0.0
        var sBMean = 0.0
        var tRMean = 0.0
        var tGMean = 0.0
        var tBMean = 0.0
        var dRMean = 0.0
        var dGMean = 0.0
        var dBMean = 0.0
        var count = 0

        for (y in 30..104) {
            val ny = (y - 67.0) / 37.0
            for (x in 30..98) {
                val nx = (x - 64.0) / 34.0
                if (nx * nx + ny * ny <= 1.0) {
                    if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom) > 0.12f) continue
                    if (y in 82..102 && x in 42..86) continue

                    val idx = y * CROP_SIZE + x
                    val sc = swapPx[idx]
                    val tc = tgtPx[idx]
                    val dc = srcPx?.get(idx) ?: sc
                    val tLum = 0.299 * (tc ushr 16 and 0xFF) + 0.587 * (tc ushr 8 and 0xFF) + 0.114 * (tc and 0xFF)
                    val dLum = 0.299 * (dc ushr 16 and 0xFF) + 0.587 * (dc ushr 8 and 0xFF) + 0.114 * (dc and 0xFF)
                    if (tLum < 50.0 || dLum < 45.0) continue

                    sRMean += (sc ushr 16) and 0xFF
                    sGMean += (sc ushr 8) and 0xFF
                    sBMean += sc and 0xFF
                    tRMean += (tc ushr 16) and 0xFF
                    tGMean += (tc ushr 8) and 0xFF
                    tBMean += tc and 0xFF
                    dRMean += (dc ushr 16) and 0xFF
                    dGMean += (dc ushr 8) and 0xFF
                    dBMean += dc and 0xFF
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
        dRMean /= count
        dGMean /= count
        dBMean /= count

        var sRVar = 0.0
        var sGVar = 0.0
        var sBVar = 0.0
        var tRVar = 0.0
        var tGVar = 0.0
        var tBVar = 0.0
        var dRVar = 0.0
        var dGVar = 0.0
        var dBVar = 0.0

        for (y in 30..104) {
            val ny = (y - 67.0) / 37.0
            for (x in 30..98) {
                val nx = (x - 64.0) / 34.0
                if (nx * nx + ny * ny <= 1.0) {
                    if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom) > 0.12f) continue
                    if (y in 82..102 && x in 42..86) continue

                    val idx = y * CROP_SIZE + x
                    val sc = swapPx[idx]
                    val tc = tgtPx[idx]
                    val dc = srcPx?.get(idx) ?: sc
                    val tLum = 0.299 * (tc ushr 16 and 0xFF) + 0.587 * (tc ushr 8 and 0xFF) + 0.114 * (tc and 0xFF)
                    val dLum = 0.299 * (dc ushr 16 and 0xFF) + 0.587 * (dc ushr 8 and 0xFF) + 0.114 * (dc and 0xFF)
                    if (tLum < 50.0 || dLum < 45.0) continue

                    val sr = ((sc ushr 16) and 0xFF) - sRMean
                    val sg = ((sc ushr 8) and 0xFF) - sGMean
                    val sb = (sc and 0xFF) - sBMean
                    val tr = ((tc ushr 16) and 0xFF) - tRMean
                    val tg = ((tc ushr 8) and 0xFF) - tGMean
                    val tb = (tc and 0xFF) - tBMean
                    val dr = ((dc ushr 16) and 0xFF) - dRMean
                    val dg = ((dc ushr 8) and 0xFF) - dGMean
                    val db = (dc and 0xFF) - dBMean

                    sRVar += sr * sr
                    sGVar += sg * sg
                    sBVar += sb * sb
                    tRVar += tr * tr
                    tGVar += tg * tg
                    tBVar += tb * tb
                    dRVar += dr * dr
                    dGVar += dg * dg
                    dBVar += db * db
                }
            }
        }

        val sRStd = sqrt(sRVar / count).coerceAtLeast(6.0)
        val sGStd = sqrt(sGVar / count).coerceAtLeast(6.0)
        val sBStd = sqrt(sBVar / count).coerceAtLeast(6.0)
        val tRStd = sqrt(tRVar / count).coerceAtLeast(6.0)
        val tGStd = sqrt(tGVar / count).coerceAtLeast(6.0)
        val tBStd = sqrt(tBVar / count).coerceAtLeast(6.0)
        val dRStd = sqrt(dRVar / count).coerceAtLeast(6.0)
        val dGStd = sqrt(dGVar / count).coerceAtLeast(6.0)
        val dBStd = sqrt(dBVar / count).coerceAtLeast(6.0)

        // Compute goal skin statistics based on user-selected SkinToneSourceMode
        val (goalRMean, goalGMean, goalBMean) = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> Triple(tRMean, tGMean, tBMean)
            SkinToneSourceMode.SOURCE_IDENTITY -> Triple(
                dRMean * 0.85 + tRMean * 0.15,
                dGMean * 0.85 + tGMean * 0.15,
                dBMean * 0.85 + tBMean * 0.15
            )
            SkinToneSourceMode.BALANCED_BLEND -> Triple(
                dRMean * 0.50 + tRMean * 0.50,
                dGMean * 0.50 + tGMean * 0.50,
                dBMean * 0.50 + tBMean * 0.50
            )
        }
        val (goalRStd, goalGStd, goalBStd) = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> Triple(tRStd, tGStd, tBStd)
            SkinToneSourceMode.SOURCE_IDENTITY -> Triple(
                dRStd * 0.80 + tRStd * 0.20,
                dGStd * 0.80 + tGStd * 0.20,
                dBStd * 0.80 + tBStd * 0.20
            )
            SkinToneSourceMode.BALANCED_BLEND -> Triple(
                dRStd * 0.50 + tRStd * 0.50,
                dGStd * 0.50 + tGStd * 0.50,
                dBStd * 0.50 + tBStd * 0.50
            )
        }

        val scaleR = (goalRStd / sRStd).coerceIn(0.80, 1.25)
        val scaleG = (goalGStd / sGStd).coerceIn(0.80, 1.25)
        val scaleB = (goalBStd / sBStd).coerceIn(0.80, 1.25)

        val outPx = IntArray(total)
        val baseBlend = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> strength.coerceIn(0f, 1f)
            SkinToneSourceMode.SOURCE_IDENTITY -> 0.88f
            SkinToneSourceMode.BALANCED_BLEND -> 0.80f
        }

        for (y in 0 until CROP_SIZE) {
            val row = y * CROP_SIZE
            for (x in 0 until CROP_SIZE) {
                val i = row + x
                val sc = swapPx[i]
                val origR = (sc ushr 16) and 0xFF
                val origG = (sc ushr 8) and 0xFF
                val origB = sc and 0xFF

                val eyeBrowProt = computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom)
                val blend = baseBlend * (1.0f - 0.92f * eyeBrowProt)

                val matchedR = ((origR - sRMean) * scaleR + goalRMean).toFloat()
                val matchedG = ((origG - sGMean) * scaleG + goalGMean).toFloat()
                val matchedB = ((origB - sBMean) * scaleB + goalBMean).toFloat()

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
     * Runs `gfpgan_1.4.onnx` ([1, 3, 512, 512] in [-1, 1]) directly using an active in-memory [OrtSession]
     * and returns the full 512x512 HD restored Bitmap.
     */
    fun runGfpganSession512(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        crop128Or512: Bitmap
    ): Bitmap {
        val gfpSize = HD_SIZE
        val upscaled512 = if (crop128Or512.width == gfpSize && crop128Or512.height == gfpSize) {
            crop128Or512
        } else {
            Bitmap.createScaledBitmap(crop128Or512, gfpSize, gfpSize, true)
        }
        val hw = gfpSize * gfpSize
        val pixels = IntArray(hw)
        upscaled512.getPixels(pixels, 0, gfpSize, 0, 0, gfpSize, gfpSize)
        if (upscaled512 !== crop128Or512) upscaled512.recycle()

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
        val restored512 = Bitmap.createBitmap(gfpSize, gfpSize, Bitmap.Config.ARGB_8888)
        restored512.setPixels(outPixels512, 0, gfpSize, 0, 0, gfpSize, gfpSize)
        return restored512
    }

    /**
     * Runs `gfpgan_1.4.onnx` ([1, 3, 512, 512] in [-1, 1]) if installed (or preloaded in memory)
     * and returns the full 512x512 HD restored Bitmap directly (without downscaling to 128x128!).
     */
    fun runOptionalGfpganEnhancement512(
        ortEnv: OrtEnvironment,
        gfpganFile: File?,
        crop128Or512: Bitmap,
        preferHardwareAccel: Boolean = false,
        preloadedGfpganSession: OrtSession? = null
    ): Bitmap? {
        if (preloadedGfpganSession != null) {
            return runCatching {
                runGfpganSession512(ortEnv, preloadedGfpganSession, crop128Or512)
            }.getOrNull()
        }
        if (gfpganFile == null || !gfpganFile.exists() || gfpganFile.length() <= 1024L) {
            return null
        }
        return runCatching {
            OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
                ortEnv.createSession(gfpganFile.absolutePath, opts).use { session ->
                    runGfpganSession512(ortEnv, session, crop128Or512)
                }
            }
        }.getOrNull()
    }

    /**
     * Online AI Photo Style 512x512 HD Super-Resolution & Seamless Blending Pipeline:
     *  1. Upscales the cleaned 128x128 face crop to 512x512 HD (using `gfpgan_1.4.onnx` if installed
     *     or smooth bicubic/bilinear scaling).
     *  2. Warps the target face at full 512x512 HD resolution (`alignedTarget512`).
     *  3. Transfers natural high-frequency skin pore texture & studio highlight sheen from `alignedTarget512`
     *     onto the 512x512 swapped skin (with an occlusion gate that rejects hair strands/scratches).
     *  4. Restores 512x512 razor-sharp Right & Left Eyes and Eyebrows using dynamic 512x512 landmarks.
     *  5. Blends the 512x512 HD face directly onto `targetPixels` using a tight Biometric Inner-Face Mask
     *     and foreground occlusion guard so cheek/jawline seams never appear.
     */
    fun enhanceAndBlendOnlineHdFace512(
        ortEnv: OrtEnvironment,
        gfpganFile: File?,
        targetBitmap: Bitmap,
        targetPixels: IntArray,
        targetWidth: Int,
        targetHeight: Int,
        colorCorrected128: Bitmap,
        forwardMatrix128: FloatArray,
        targetLandmarks5: List<PointF>,
        sourceBitmap: Bitmap? = null,
        sourceLandmarks5: List<PointF>? = null,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.TARGET_SCENE,
        faceReactionMode: FaceReactionSourceMode = FaceReactionSourceMode.TARGET_REACTION,
        enableColorTransfer: Boolean = true,
        enableOcclusionProtection: Boolean = true,
        blendStrength: Float = 1.0f,
        enhancementStrength: Float = 0.85f,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f,
        preferHardwareAccel: Boolean = false,
        segformerFile: File? = null,
        preloadedGfpganSession: OrtSession? = null,
        preloadedSegformerSession: OrtSession? = null
    ): Bitmap {
        val m512 = FloatArray(6) { i -> forwardMatrix128[i] * 4.0f }
        val alignedTarget512 = FaceAlignment.warpAffineCrop(targetBitmap, m512, HD_SIZE)
        val geom512 = computeWarpedGeometry(forwardMatrix128, targetLandmarks5, scale = 4.0f)

        // Optional Neural Occlusion Gate via segformer_B5_ce.onnx (protects eyeglasses, hair bangs, hats & earrings)
        val neuralOcclusionGate512 = if (enableOcclusionProtection) {
            HeadSegmentationAndInpainting.computeFaceOcclusionGate512(
                ortEnv = ortEnv,
                alignedTarget512 = alignedTarget512,
                segModelFile = segformerFile,
                preferHardwareAccel = preferHardwareAccel,
                preloadedSegformerSession = preloadedSegformerSession
            )
        } else null

        // Prepare 512x512 aligned Source crop if needed for Source Skin Tone or Source Reaction (Smile/Teeth/Tongue)
        val mSrc128 = if (sourceBitmap != null && sourceLandmarks5 != null && sourceLandmarks5.size >= 5) {
            FaceAlignment.estimateNorm(sourceLandmarks5, 128)
        } else null
        val alignedSource512 = if (sourceBitmap != null && mSrc128 != null) {
            val mSrc512 = FloatArray(6) { i -> mSrc128[i] * 4.0f }
            FaceAlignment.warpAffineCrop(sourceBitmap, mSrc512, HD_SIZE)
        } else null
        val srcGeom512 = if (mSrc128 != null && sourceLandmarks5 != null) {
            computeWarpedGeometry(mSrc128, sourceLandmarks5, scale = 4.0f)
        } else null

        val clampedEnhance = enhancementStrength.coerceIn(0f, 1f)
        val gfp512 = if (clampedEnhance > 0.05f) {
            runOptionalGfpganEnhancement512(
                ortEnv = ortEnv,
                gfpganFile = gfpganFile,
                crop128Or512 = colorCorrected128,
                preferHardwareAccel = preferHardwareAccel,
                preloadedGfpganSession = preloadedGfpganSession
            )
        } else null
        val baseUpscaled512 = Bitmap.createScaledBitmap(colorCorrected128, HD_SIZE, HD_SIZE, true)

        val total512 = HD_SIZE * HD_SIZE
        val swapPx512 = IntArray(total512)
        val tgtPx512 = IntArray(total512)
        val srcPx512 = if (alignedSource512 != null) IntArray(total512) else null

        baseUpscaled512.getPixels(swapPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        baseUpscaled512.recycle()

        if (gfp512 != null) {
            val gfpPx = IntArray(total512)
            gfp512.getPixels(gfpPx, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            gfp512.recycle()
            val invEnh = 1.0f - clampedEnhance
            for (i in 0 until total512) {
                val bc = swapPx512[i]
                val gc = gfpPx[i]
                val r = (((gc ushr 16) and 0xFF) * clampedEnhance + ((bc ushr 16) and 0xFF) * invEnh).toInt().coerceIn(0, 255)
                val g = (((gc ushr 8) and 0xFF) * clampedEnhance + ((bc ushr 8) and 0xFF) * invEnh).toInt().coerceIn(0, 255)
                val b = ((gc and 0xFF) * clampedEnhance + (bc and 0xFF) * invEnh).toInt().coerceIn(0, 255)
                swapPx512[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        alignedTarget512.getPixels(tgtPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        alignedSource512?.getPixels(srcPx512!!, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        alignedTarget512.recycle()
        alignedSource512?.recycle()

        // 0a. Post-GFPGAN 512x512 Skin Tone Lock so GFPGAN never overrides user's chosen SkinToneSourceMode
        if (enableColorTransfer) {
            harmonizeSkinToneByMode512(
                swapPx512 = swapPx512,
                tgtPx512 = tgtPx512,
                srcPx512 = srcPx512,
                geom512 = geom512,
                skinToneMode = skinToneMode
            )
        }

        // 0b. Face Reaction Synthesis (Smile, Visible Teeth, Visible Tongue, Mouth & Eye Expression)
        when (faceReactionMode) {
            FaceReactionSourceMode.TARGET_REACTION -> {
                // Preserve crisp 512x512 Target smile creases, visible white teeth, and visible tongue
                preserveTargetMouthTeethAndTongue512(
                    swapPx512 = swapPx512,
                    tgtPx512 = tgtPx512,
                    geom512 = geom512
                )
            }
            FaceReactionSourceMode.SOURCE_REACTION -> {
                // Transfer Source Identity's own smile, visible teeth, visible tongue & eye expression
                if (srcPx512 != null && srcGeom512 != null) {
                    transferSourceReactionSmileTeethTongue512(
                        swapPx512 = swapPx512,
                        srcPx512 = srcPx512,
                        tgtGeom512 = geom512,
                        srcGeom512 = srcGeom512
                    )
                }
            }
        }

        // 1. Transfer natural 512x512 DSLR skin pore micro-texture & studio lighting sheen
        //    (strictly gated so hair strands or dark lines on target are never transferred as texture)
        val poreScale = (if (skinToneMode == SkinToneSourceMode.SOURCE_IDENTITY) 0.35f else 0.55f) *
            (0.4f + 0.6f * clampedEnhance)
        for (y in 16 until HD_SIZE - 16) {
            val row = y * HD_SIZE
            for (x in 16 until HD_SIZE - 16) {
                val idx = row + x
                val tc = tgtPx512[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                val n = tgtPx512[(y - 2) * HD_SIZE + x]
                val s = tgtPx512[(y + 2) * HD_SIZE + x]
                val l = tgtPx512[row + x - 2]
                val r = tgtPx512[row + x + 2]
                val tAvgLum = (
                    (0.299f * (n ushr 16 and 0xFF) + 0.587f * (n ushr 8 and 0xFF) + 0.114f * (n and 0xFF)) +
                    (0.299f * (s ushr 16 and 0xFF) + 0.587f * (s ushr 8 and 0xFF) + 0.114f * (s and 0xFF)) +
                    (0.299f * (l ushr 16 and 0xFF) + 0.587f * (l ushr 8 and 0xFF) + 0.114f * (l and 0xFF)) +
                    (0.299f * (r ushr 16 and 0xFF) + 0.587f * (r ushr 8 and 0xFF) + 0.114f * (r and 0xFF))
                ) * 0.25f

                val poreDiff = tLum - tAvgLum
                if (tLum > 70f && abs(poreDiff) < 12f) {
                    val sc = swapPx512[idx]
                    val sr = (sc ushr 16) and 0xFF
                    val sg = (sc ushr 8) and 0xFF
                    val sb = sc and 0xFF
                    val poreBoost = poreDiff * poreScale
                    val nr = (sr + poreBoost).toInt().coerceIn(0, 255)
                    val ng = (sg + poreBoost).toInt().coerceIn(0, 255)
                    val nb = (sb + poreBoost).toInt().coerceIn(0, 255)
                    swapPx512[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
                }
            }
        }

        // 2. Restore 512x512 HD Eyebrow Continuity & 512x512 HD Left/Right Ocular Clarity
        restoreDynamicEyebrowContinuity(
            outPx = swapPx512,
            tgtPx = tgtPx512,
            size = HD_SIZE,
            geom = geom512
        )
        restoreDynamicOcularSockets(
            outPx = swapPx512,
            tgtPx = tgtPx512,
            srcPx = if (faceReactionMode == FaceReactionSourceMode.SOURCE_REACTION) srcPx512 else null,
            size = HD_SIZE,
            geom = geom512
        )

        // 3. Build tight 512x512 Biometric Landmark-Fitted Cosine Feather Mask with Foreground Occlusion Guard
        val mask512 = createBiometricFaceMask512(
            geom512 = geom512,
            tgtPx512 = tgtPx512,
            swapPx512 = swapPx512,
            skinToneMode = skinToneMode,
            enableOcclusionProtection = enableOcclusionProtection,
            neuralOcclusionGate512 = neuralOcclusionGate512
        )

        // 3b. Multi-Band (Low-Frequency Illumination + High-Frequency Facial Detail) Boundary & Jawline Shadow Fusion
        applyMultiBandBoundaryAndJawlineFusion512(
            swapPx512 = swapPx512,
            tgtPx512 = tgtPx512,
            mask512 = mask512
        )

        // 4. Inverse Affine Warp of 512x512 HD Face directly onto full-resolution targetPixels
        val inv = FaceAlignment.invertAffine2x3(m512)
        val corners = arrayOf(
            0f to 0f,
            HD_SIZE.toFloat() to 0f,
            0f to HD_SIZE.toFloat(),
            HD_SIZE.toFloat() to HD_SIZE.toFloat()
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

        minX = (minX - 16).coerceIn(0, targetWidth - 1)
        maxX = (maxX + 16).coerceIn(0, targetWidth - 1)
        minY = (minY - 16).coerceIn(0, targetHeight - 1)
        maxY = (maxY + 16).coerceIn(0, targetHeight - 1)

        val m00 = m512[0]
        val m01 = m512[1]
        val m02 = m512[2]
        val m10 = m512[3]
        val m11 = m512[4]
        val m12 = m512[5]

        val clampedBlend = blendStrength.coerceIn(0.20f, 1.0f)
        val safeScale = scaleAdjust.coerceIn(0.85f, 1.15f)
        val shiftU = offsetX.coerceIn(-36f, 36f)
        val shiftV = offsetY.coerceIn(-36f, 36f)

        for (y in minY..maxY) {
            val rowOffset = y * targetWidth
            val baseU = m01 * y + m02
            val baseV = m11 * y + m12
            for (x in minX..maxX) {
                val rawU = m00 * x + baseU
                val rawV = m10 * x + baseV
                val u = (rawU - 256f - shiftU) / safeScale + 256f
                val v = (rawV - 256f - shiftV) / safeScale + 256f
                if (u >= 2f && u < HD_SIZE - 3f && v >= 2f && v < HD_SIZE - 3f) {
                    val alpha = sampleMaskBilinearGeneric(mask512, HD_SIZE, u, v) * clampedBlend
                    if (alpha > 0.003f) {
                        val swapColor = FaceAlignment.sampleBilinearClamped(
                            swapPx512,
                            HD_SIZE,
                            HD_SIZE,
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

        val restoredHd512 = Bitmap.createBitmap(HD_SIZE, HD_SIZE, Bitmap.Config.ARGB_8888)
        restoredHd512.setPixels(swapPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        return restoredHd512
    }

    /**
     * 2-Band (Low-Frequency Illumination + High-Frequency Detail) Multi-Band & Jawline Shadow Fusion
     * in 512x512 HD space: smoothly aligns the outer cheek, forehead, and chin/neck jawline shadow
     * envelope of `swapPx512` to `tgtPx512` across the transition ring (`0.05 < mask512 < 0.92`)
     * while preserving 100% of the swapped face's crisp high-frequency detail.
     */
    private fun applyMultiBandBoundaryAndJawlineFusion512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        mask512: FloatArray
    ) {
        val step = 12
        for (y in step until HD_SIZE - step) {
            val row = y * HD_SIZE
            // Stronger low-frequency shadow alignment along the lower chin/neck jawline (y > 340)
            val jawlineBoost = if (y > 340) 0.35f else 0.22f
            for (x in step until HD_SIZE - step) {
                val idx = row + x
                val m = mask512[idx]
                if (m <= 0.04f || m >= 0.94f) continue

                // Transition weight peaks around the boundary feather zone (m ~ 0.45..0.85)
                val boundaryWeight = ((1.0f - m) * jawlineBoost).coerceIn(0f, 0.42f)
                val sLow = compute5PointLowPassRGB512(swapPx512, x, y, step)
                val tLow = compute5PointLowPassRGB512(tgtPx512, x, y, step)

                val sc = swapPx512[idx]
                val sr = (sc ushr 16) and 0xFF
                val sg = (sc ushr 8) and 0xFF
                val sb = sc and 0xFF

                val nr = (sr + (tLow[0] - sLow[0]) * boundaryWeight).toInt().coerceIn(0, 255)
                val ng = (sg + (tLow[1] - sLow[1]) * boundaryWeight).toInt().coerceIn(0, 255)
                val nb = (sb + (tLow[2] - sLow[2]) * boundaryWeight).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            }
        }
    }

    private fun compute5PointLowPassRGB512(px: IntArray, x: Int, y: Int, d: Int): FloatArray {
        val c0 = px[y * HD_SIZE + x]
        val cN = px[(y - d) * HD_SIZE + x]
        val cS = px[(y + d) * HD_SIZE + x]
        val cW = px[y * HD_SIZE + (x - d)]
        val cE = px[y * HD_SIZE + (x + d)]
        val r = (((c0 ushr 16) and 0xFF) + ((cN ushr 16) and 0xFF) + ((cS ushr 16) and 0xFF) +
            ((cW ushr 16) and 0xFF) + ((cE ushr 16) and 0xFF)) * 0.2f
        val g = (((c0 ushr 8) and 0xFF) + ((cN ushr 8) and 0xFF) + ((cS ushr 8) and 0xFF) +
            ((cW ushr 8) and 0xFF) + ((cE ushr 8) and 0xFF)) * 0.2f
        val b = ((c0 and 0xFF) + (cN and 0xFF) + (cS and 0xFF) + (cW and 0xFF) + (cE and 0xFF)) * 0.2f
        return floatArrayOf(r, g, b)
    }

    /**
     * Re-harmonizes the 512x512 restored crop so that GFPGAN never overrides the user's chosen
     * [SkinToneSourceMode] (`TARGET_SCENE`, `SOURCE_IDENTITY`, or `BALANCED_BLEND`).
     */
    private fun harmonizeSkinToneByMode512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        srcPx512: IntArray?,
        geom512: WarpedFaceGeometry,
        skinToneMode: SkinToneSourceMode
    ) {
        var sRMean = 0.0
        var sGMean = 0.0
        var sBMean = 0.0
        var tRMean = 0.0
        var tGMean = 0.0
        var tBMean = 0.0
        var dRMean = 0.0
        var dGMean = 0.0
        var dBMean = 0.0
        var count = 0

        for (y in 120..410 step 2) {
            val ny = (y - 268.0) / 148.0
            for (x in 120..392 step 2) {
                val nx = (x - 256.0) / 136.0
                if (nx * nx + ny * ny <= 1.0) {
                    if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom512) > 0.12f) continue
                    if (y in 328..408 && x in 168..344) continue

                    val idx = y * HD_SIZE + x
                    val sc = swapPx512[idx]
                    val tc = tgtPx512[idx]
                    val dc = srcPx512?.get(idx) ?: sc
                    val tLum = 0.299 * (tc ushr 16 and 0xFF) + 0.587 * (tc ushr 8 and 0xFF) + 0.114 * (tc and 0xFF)
                    val dLum = 0.299 * (dc ushr 16 and 0xFF) + 0.587 * (dc ushr 8 and 0xFF) + 0.114 * (dc and 0xFF)
                    if (tLum < 50.0 || dLum < 45.0) continue

                    sRMean += (sc ushr 16) and 0xFF
                    sGMean += (sc ushr 8) and 0xFF
                    sBMean += sc and 0xFF
                    tRMean += (tc ushr 16) and 0xFF
                    tGMean += (tc ushr 8) and 0xFF
                    tBMean += tc and 0xFF
                    dRMean += (dc ushr 16) and 0xFF
                    dGMean += (dc ushr 8) and 0xFF
                    dBMean += dc and 0xFF
                    count++
                }
            }
        }
        if (count < 32) return

        sRMean /= count
        sGMean /= count
        sBMean /= count
        tRMean /= count
        tGMean /= count
        tBMean /= count
        dRMean /= count
        dGMean /= count
        dBMean /= count

        val (goalRMean, goalGMean, goalBMean) = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> Triple(tRMean, tGMean, tBMean)
            SkinToneSourceMode.SOURCE_IDENTITY -> Triple(
                dRMean * 0.85 + tRMean * 0.15,
                dGMean * 0.85 + tGMean * 0.15,
                dBMean * 0.85 + tBMean * 0.15
            )
            SkinToneSourceMode.BALANCED_BLEND -> Triple(
                dRMean * 0.50 + tRMean * 0.50,
                dGMean * 0.50 + tGMean * 0.50,
                dBMean * 0.50 + tBMean * 0.50
            )
        }

        val shiftR = (goalRMean - sRMean).toFloat()
        val shiftG = (goalGMean - sGMean).toFloat()
        val shiftB = (goalBMean - sBMean).toFloat()
        val strength = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> 0.65f
            SkinToneSourceMode.SOURCE_IDENTITY -> 0.86f
            SkinToneSourceMode.BALANCED_BLEND -> 0.75f
        }

        for (y in 0 until HD_SIZE) {
            val row = y * HD_SIZE
            for (x in 0 until HD_SIZE) {
                val idx = row + x
                val eyeProt = computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom512)
                val blend = strength * (1.0f - 0.90f * eyeProt)
                val c = swapPx512[idx]
                val r = ((c ushr 16 and 0xFF) + shiftR * blend).toInt().coerceIn(0, 255)
                val g = ((c ushr 8 and 0xFF) + shiftG * blend).toInt().coerceIn(0, 255)
                val b = ((c and 0xFF) + shiftB * blend).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }

    /**
     * Preserves 512x512 HD smile creases, visible white teeth (பற்கள்), and visible tongue (நாக்கு)
     * from the Target photo when `FaceReactionSourceMode.TARGET_REACTION` is active.
     */
    private fun preserveTargetMouthTeethAndTongue512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        geom512: WarpedFaceGeometry
    ) {
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val mouthWidth = hypot(
            (geom512.rightMouth.x - geom512.leftMouth.x).toDouble(),
            (geom512.rightMouth.y - geom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(72f, 220f)

        val rx = mouthWidth * 0.72f
        val ry = mouthWidth * 0.52f
        val minX = (mouthMidX - rx - 8f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxX = (mouthMidX + rx + 8f).toInt().coerceIn(8, HD_SIZE - 9)
        val minY = (mouthMidY - ry - 8f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxY = (mouthMidY + ry + 8f).toInt().coerceIn(8, HD_SIZE - 9)

        for (y in minY..maxY) {
            val row = y * HD_SIZE
            for (x in minX..maxX) {
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    mouthMidX, mouthMidY,
                    rx, ry,
                    geom512.cosA, geom512.sinA
                )
                if (dSq >= 1.0f) continue
                val r = sqrt(dSq)
                val env = (0.5f * (1.0f + cos(Math.PI * r.coerceIn(0f, 1f)))).toFloat()

                val idx = row + x
                val tc = tgtPx512[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB
                val maxCh = max(tR, max(tG, tB)).coerceAtLeast(1)
                val minCh = min(tR, min(tG, tB))
                val tSat = (maxCh - minCh).toFloat() / maxCh.toFloat()

                // Detect visible teeth (bright low-saturation enamel), pink/red tongue, or dark open oral cavity
                val isVisibleTeeth = r <= 0.78f && tLum > 130f && tSat < 0.28f
                val isVisibleTongue = r <= 0.80f && tR > 115 && tR > (tG * 1.24f) && tR > (tB * 1.25f) && tSat > 0.22f
                val isOpenMouthCavity = r <= 0.70f && tLum < 52f

                val sc = swapPx512[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF

                if (isVisibleTeeth || isVisibleTongue || isOpenMouthCavity) {
                    val directWeight = (0.76f * env).coerceIn(0f, 0.82f)
                    val invW = 1.0f - directWeight
                    val outR = (sR * invW + tR * directWeight).toInt().coerceIn(0, 255)
                    val outG = (sG * invW + tG * directWeight).toInt().coerceIn(0, 255)
                    val outB = (sB * invW + tB * directWeight).toInt().coerceIn(0, 255)
                    swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                } else {
                    // Enhance lip & smile crease high-frequency definition from Target 512x512
                    val n = tgtPx512[(y - 2) * HD_SIZE + x]
                    val s = tgtPx512[(y + 2) * HD_SIZE + x]
                    val l = tgtPx512[row + x - 2]
                    val rg = tgtPx512[row + x + 2]
                    val tLocalLum = (
                        (0.299f * (n ushr 16 and 0xFF) + 0.587f * (n ushr 8 and 0xFF) + 0.114f * (n and 0xFF)) +
                        (0.299f * (s ushr 16 and 0xFF) + 0.587f * (s ushr 8 and 0xFF) + 0.114f * (s and 0xFF)) +
                        (0.299f * (l ushr 16 and 0xFF) + 0.587f * (l ushr 8 and 0xFF) + 0.114f * (l and 0xFF)) +
                        (0.299f * (rg ushr 16 and 0xFF) + 0.587f * (rg ushr 8 and 0xFF) + 0.114f * (rg and 0xFF))
                    ) * 0.25f
                    val detail = (tLum - tLocalLum) * 0.65f * env
                    val outR = (sR + detail).toInt().coerceIn(0, 255)
                    val outG = (sG + detail).toInt().coerceIn(0, 255)
                    val outB = (sB + detail).toInt().coerceIn(0, 255)
                    swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                }
            }
        }
    }

    /**
     * Transfers the Source identity's own facial reaction (smile, visible teeth, visible tongue,
     * lip curvature, and expressive eye reaction) directly from `srcPx512` onto `swapPx512`
     * using landmark-registered local warping from `tgtGeom512` to `srcGeom512`.
     */
    private fun transferSourceReactionSmileTeethTongue512(
        swapPx512: IntArray,
        srcPx512: IntArray,
        tgtGeom512: WarpedFaceGeometry,
        srcGeom512: WarpedFaceGeometry
    ) {
        val tMouthMidX = (tgtGeom512.leftMouth.x + tgtGeom512.rightMouth.x) * 0.5f
        val tMouthMidY = (tgtGeom512.leftMouth.y + tgtGeom512.rightMouth.y) * 0.5f
        val sMouthMidX = (srcGeom512.leftMouth.x + srcGeom512.rightMouth.x) * 0.5f
        val sMouthMidY = (srcGeom512.leftMouth.y + srcGeom512.rightMouth.y) * 0.5f

        val tMouthW = hypot(
            (tgtGeom512.rightMouth.x - tgtGeom512.leftMouth.x).toDouble(),
            (tgtGeom512.rightMouth.y - tgtGeom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(72f, 220f)
        val sMouthW = hypot(
            (srcGeom512.rightMouth.x - srcGeom512.leftMouth.x).toDouble(),
            (srcGeom512.rightMouth.y - srcGeom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(72f, 220f)

        val scaleX = (sMouthW / tMouthW).coerceIn(0.75f, 1.35f)
        val scaleY = scaleX

        // Measure perioral skin tone difference between swapPx512 and srcPx512 so the transferred
        // Source smile/mouth blends seamlessly without any lip-border color ring
        var swapPerioralR = 0f
        var swapPerioralG = 0f
        var swapPerioralB = 0f
        var srcPerioralR = 0f
        var srcPerioralG = 0f
        var srcPerioralB = 0f
        var pCount = 0

        val rx = tMouthW * 0.88f
        val ry = tMouthW * 0.64f
        val minX = (tMouthMidX - rx - 8f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxX = (tMouthMidX + rx + 8f).toInt().coerceIn(8, HD_SIZE - 9)
        val minY = (tMouthMidY - ry - 8f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxY = (tMouthMidY + ry + 8f).toInt().coerceIn(8, HD_SIZE - 9)

        for (y in minY..maxY step 2) {
            for (x in minX..maxX step 2) {
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    tMouthMidX, tMouthMidY,
                    rx, ry,
                    tgtGeom512.cosA, tgtGeom512.sinA
                )
                val r = sqrt(dSq)
                if (r in 0.65f..0.98f) {
                    val uSrc = sMouthMidX + (x - tMouthMidX) * scaleX
                    val vSrc = sMouthMidY + (y - tMouthMidY) * scaleY
                    if (uSrc >= 4f && uSrc < HD_SIZE - 5f && vSrc >= 4f && vSrc < HD_SIZE - 5f) {
                        val sc = swapPx512[y * HD_SIZE + x]
                        val dc = FaceAlignment.sampleBilinearClamped(srcPx512, HD_SIZE, HD_SIZE, uSrc, vSrc)
                        swapPerioralR += (sc ushr 16) and 0xFF
                        swapPerioralG += (sc ushr 8) and 0xFF
                        swapPerioralB += sc and 0xFF
                        srcPerioralR += (dc ushr 16) and 0xFF
                        srcPerioralG += (dc ushr 8) and 0xFF
                        srcPerioralB += dc and 0xFF
                        pCount++
                    }
                }
            }
        }

        val deltaR = if (pCount > 8) (swapPerioralR - srcPerioralR) / pCount else 0f
        val deltaG = if (pCount > 8) (swapPerioralG - srcPerioralG) / pCount else 0f
        val deltaB = if (pCount > 8) (swapPerioralB - srcPerioralB) / pCount else 0f

        for (y in minY..maxY) {
            val row = y * HD_SIZE
            for (x in minX..maxX) {
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    tMouthMidX, tMouthMidY,
                    rx, ry,
                    tgtGeom512.cosA, tgtGeom512.sinA
                )
                if (dSq >= 1.0f) continue
                val r = sqrt(dSq)

                val uSrc = sMouthMidX + (x - tMouthMidX) * scaleX
                val vSrc = sMouthMidY + (y - tMouthMidY) * scaleY
                if (uSrc < 2f || uSrc >= HD_SIZE - 3f || vSrc < 2f || vSrc >= HD_SIZE - 3f) continue

                val dc = FaceAlignment.sampleBilinearClamped(srcPx512, HD_SIZE, HD_SIZE, uSrc, vSrc)
                val dR = (dc ushr 16) and 0xFF
                val dG = (dc ushr 8) and 0xFF
                val dB = dc and 0xFF
                val dLum = 0.299f * dR + 0.587f * dG + 0.114f * dB
                val maxCh = max(dR, max(dG, dB)).coerceAtLeast(1)
                val minCh = min(dR, min(dG, dB))
                val dSat = (maxCh - minCh).toFloat() / maxCh.toFloat()

                // Protect bright white teeth and vivid tongue in Source photo from skin-tone shifting
                val isSourceTeeth = r <= 0.72f && dLum > 135f && dSat < 0.26f
                val isSourceTongue = r <= 0.75f && dR > 115 && dR > (dG * 1.24f) && dR > (dB * 1.24f)
                val skinShiftScale = if (isSourceTeeth) 0.12f else if (isSourceTongue) 0.25f else 0.85f

                val adjSrcR = (dR + deltaR * skinShiftScale).toInt().coerceIn(0, 255)
                val adjSrcG = (dG + deltaG * skinShiftScale).toInt().coerceIn(0, 255)
                val adjSrcB = (dB + deltaB * skinShiftScale).toInt().coerceIn(0, 255)

                val coreRatio = 0.48f
                val alpha = when {
                    r <= coreRatio -> 0.90f
                    else -> {
                        val t = (r - coreRatio) / (1.0f - coreRatio)
                        (0.90f * 0.5f * (1.0f + cos(Math.PI * t))).toFloat()
                    }
                }

                val idx = row + x
                val sc = swapPx512[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF
                val invA = 1.0f - alpha

                val outR = (sR * invA + adjSrcR * alpha).toInt().coerceIn(0, 255)
                val outG = (sG * invA + adjSrcG * alpha).toInt().coerceIn(0, 255)
                val outB = (sB * invA + adjSrcB * alpha).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    /**
     * Creates a landmark-fitted 512x512 Biometric Inner-Face Mask that:
     *  - Covers the eyes, eyebrows, nose, lips, and inner cheeks smoothly
     *  - Tapers cleanly inside the outer cheek and jawline contour so paste seams never appear
     *  - Protects dark foreground hair strands or earrings near the outer cheek boundary
     */
    private fun createBiometricFaceMask512(
        geom512: WarpedFaceGeometry,
        tgtPx512: IntArray,
        swapPx512: IntArray,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.TARGET_SCENE,
        enableOcclusionProtection: Boolean = true,
        neuralOcclusionGate512: FloatArray? = null
    ): FloatArray {
        val mask = FloatArray(HD_SIZE * HD_SIZE)
        val eyeMidX = (geom512.leftEye.x + geom512.rightEye.x) * 0.5f
        val eyeMidY = (geom512.leftEye.y + geom512.rightEye.y) * 0.5f
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f

        val centerX = (eyeMidX * 0.45f + geom512.nose.x * 0.30f + mouthMidX * 0.25f)
        val centerY = (eyeMidY * 0.42f + geom512.nose.y * 0.28f + mouthMidY * 0.30f)

        // When SOURCE_IDENTITY skin tone is selected, use a wider cosine feather transition (innerCoreRatio = 0.48f)
        // so the Source skin tone blends smoothly across the cheeks and jawline
        val radiusX = (geom512.eyeDist * 1.28f).coerceIn(140f, 206f)
        val radiusY = (geom512.eyeDist * 1.46f).coerceIn(160f, 222f)
        val innerCoreRatio = if (skinToneMode == SkinToneSourceMode.SOURCE_IDENTITY) 0.48f else 0.56f
        val borderMargin = 28

        for (y in 0 until HD_SIZE) {
            val row = y * HD_SIZE
            for (x in 0 until HD_SIZE) {
                if (x < borderMargin || x >= HD_SIZE - borderMargin ||
                    y < borderMargin || y >= HD_SIZE - borderMargin
                ) {
                    mask[row + x] = 0f
                    continue
                }

                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    centerX, centerY,
                    radiusX, radiusY,
                    geom512.cosA, geom512.sinA
                )
                val r = sqrt(dSq)

                var alpha = when {
                    r <= innerCoreRatio -> 1.0f
                    r >= 1.0f -> 0.0f
                    else -> {
                        val t = (r - innerCoreRatio) / (1.0f - innerCoreRatio)
                        (0.5f * (1.0f + cos(Math.PI * t))).toFloat()
                    }
                }

                val idx = row + x
                if (enableOcclusionProtection && alpha > 0.01f) {
                    // 1. Neural segformer_B5_ce.onnx Face-Parsing Gate (protects eyeglasses, hair bangs, hats & earrings)
                    if (neuralOcclusionGate512 != null) {
                        val segGate = neuralOcclusionGate512[idx].coerceIn(0f, 1f)
                        alpha *= (0.18f + 0.82f * segGate)
                    }

                    // 2. Chromatic/Luminance Foreground Hair & Glasses Frame Occlusion Guard on outer periphery (r > 0.48)
                    if (r > 0.48f) {
                        val tc = tgtPx512[idx]
                        val sc = swapPx512[idx]
                        val tLum = 0.299f * (tc ushr 16 and 0xFF) + 0.587f * (tc ushr 8 and 0xFF) + 0.114f * (tc and 0xFF)
                        val sLum = 0.299f * (sc ushr 16 and 0xFF) + 0.587f * (sc ushr 8 and 0xFF) + 0.114f * (sc and 0xFF)
                        if (tLum < 58f && sLum > tLum + 24f) {
                            val hairKeep = ((58f - tLum) / 45f).coerceIn(0f, 0.88f)
                            alpha *= (1.0f - hairKeep)
                        }
                    }
                }

                val edgeDist = min(
                    min(x - borderMargin, HD_SIZE - 1 - borderMargin - x),
                    min(y - borderMargin, HD_SIZE - 1 - borderMargin - y)
                ).toFloat()
                val edgeEnvelope = (edgeDist / 48.0f).coerceIn(0f, 1f)

                mask[row + x] = alpha * edgeEnvelope
            }
        }
        return mask
    }

    /**
     * Creates a smooth 128x128 soft-feathered facial mask in canonical InSwapper space
     * (used by Head Replacement inner expression fusion and legacy callers).
     */
    fun createFeatheredFaceMask128(): FloatArray {
        val mask = FloatArray(CROP_SIZE * CROP_SIZE)
        val centerX = 64.0f
        val centerY = 67.0f
        val radiusX = 46.0f
        val radiusY = 50.0f
        val innerCoreRatio = 0.58f
        val borderMargin = 8

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

                val edgeDist = min(
                    min(x - borderMargin, CROP_SIZE - 1 - borderMargin - x),
                    min(y - borderMargin, CROP_SIZE - 1 - borderMargin - y)
                ).toFloat()
                val edgeEnvelope = (edgeDist / 12.0f).coerceIn(0f, 1f)

                mask[y * CROP_SIZE + x] = ellipticAlpha * edgeEnvelope
            }
        }
        return mask
    }

    /**
     * Warps and blends `swapped128` onto `targetPixels` in-place using the forward 2x3
     * similarity matrix `forwardMatrix128`.
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
                    val alpha = sampleMaskBilinearGeneric(featheredMask128, CROP_SIZE, u, v)
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

    private fun sampleMaskBilinearGeneric(mask: FloatArray, size: Int, u: Float, v: Float): Float {
        val x0 = u.toInt().coerceIn(0, size - 2)
        val y0 = v.toInt().coerceIn(0, size - 2)
        val fx = u - x0
        val fy = v - y0

        val m00 = mask[y0 * size + x0]
        val m10 = mask[y0 * size + (x0 + 1)]
        val m01 = mask[(y0 + 1) * size + x0]
        val m11 = mask[(y0 + 1) * size + (x0 + 1)]

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
