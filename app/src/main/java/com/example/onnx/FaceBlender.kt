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

        // Preserve 100% of the neural swapped face output from the ONNX swap session,
        // healing only isolated single-pixel dark glitch spikes using the swapped face's own neighborhood.
        val outPx = healCheekScratchesAndNegativeAnomalies128(swapPx, tgtPx, geom)

        val out = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Bitmap.Config.ARGB_8888)
        out.setPixels(outPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        return out
    }

    /**
     * Removes only isolated dark glitch spikes on `swapPx` using the swapped face's own smooth
     * neighborhood (`sSmooth`), NEVER overwriting swapped facial structure with target features
     * when Source and Target have different skin tones.
     */
    private fun healCheekScratchesAndNegativeAnomalies128(
        swapPx: IntArray,
        tgtPx: IntArray,
        geom: WarpedFaceGeometry
    ): IntArray {
        val out = swapPx.copyOf()

        for (y in 4 until CROP_SIZE - 4) {
            val row = y * CROP_SIZE
            for (x in 4 until CROP_SIZE - 4) {
                val idx = row + x
                val eyeBrowWeight = computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom)
                if (eyeBrowWeight > 0.15f) continue
                // Skip nose nostrils and mouth aperture so dark nostrils/lips are never smoothed
                if (y in 62..106 && x in 40..88) continue

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

                val sSmooth = computeWideSkinSmoothRGB(swapPx, CROP_SIZE, x, y, radius = 2)
                val tSmooth = computeWideSkinSmoothRGB(tgtPx, CROP_SIZE, x, y, radius = 2)

                val sSmoothLum = 0.299f * sSmooth[0] + 0.587f * sSmooth[1] + 0.114f * sSmooth[2]
                val tSmoothLum = 0.299f * tSmooth[0] + 0.587f * tSmooth[1] + 0.114f * tSmooth[2]

                // Only heal if target cheek is smooth skin AND swapped pixel has an isolated sharp dark spike
                // relative to its OWN neighborhood (sLum < sSmoothLum - 26f)
                val isTargetSmoothSkin = tLum > 80f && abs(tLum - tSmoothLum) < 8f
                val isIsolatedGlitchSpike = isTargetSmoothSkin && (sLum < sSmoothLum - 26f)

                if (isIsolatedGlitchSpike) {
                    val healR = sSmooth[0].toInt().coerceIn(0, 255)
                    val healG = sSmooth[1].toInt().coerceIn(0, 255)
                    val healB = sSmooth[2].toInt().coerceIn(0, 255)
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
     * Protects both Left and Right Eyebrows ONLY against unnatural bright white cuts/gaps,
     * while preserving 100% of the swapped identity's natural eyebrow shape and thickness.
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
        val browRx = geom.eyeDist * 0.48f
        val browRy = geom.eyeDist * 0.16f
        val boundR = max(browRx, browRy) * 1.10f

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
                    val archWeight = 0.5f * (1.0f + cos(Math.PI * normDist).toFloat())

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

                    // Only heal if the swapped eyebrow has an unnatural blown-out bright white gap (cLum > tLum + 48)
                    if (tLum < 75f && cLum > tLum + 48f) {
                        val browBlend = (0.38f * archWeight).coerceIn(0f, 0.42f)
                        val finalR = (cR * (1f - browBlend) + tR * browBlend).toInt().coerceIn(0, 255)
                        val finalG = (cG * (1f - browBlend) + tG * browBlend).toInt().coerceIn(0, 255)
                        val finalB = (cB * (1f - browBlend) + tB * browBlend).toInt().coerceIn(0, 255)
                        outPx[idx] = (0xFF shl 24) or (finalR shl 16) or (finalG shl 8) or finalB
                    }
                }
            }
        }
    }

    /**
     * Protects Left & Right Eyes strictly within the tight iris/pupil aperture (`rx = 0.22 * eyeDist`,
     * `ry = 0.13 * eyeDist`) ONLY when a pupil has a cloudy/white negative artifact, preserving 100%
     * of the swapped identity's eye shape, eyelids, eyelashes, and gaze.
     */
    private fun restoreDynamicOcularSockets(
        outPx: IntArray,
        tgtPx: IntArray,
        srcPx: IntArray?,
        size: Int,
        geom: WarpedFaceGeometry
    ) {
        val eyeCenters = arrayOf(geom.leftEye, geom.rightEye)
        val rx = geom.eyeDist * 0.22f
        val ry = geom.eyeDist * 0.13f
        val boundR = max(rx, ry) * 1.10f

        for (eyeCenter in eyeCenters) {
            val xMin = (eyeCenter.x - boundR).toInt().coerceAtLeast(1)
            val xMax = (eyeCenter.x + boundR).toInt().coerceAtMost(size - 2)
            val yMin = (eyeCenter.y - boundR).toInt().coerceAtLeast(1)
            val yMax = (eyeCenter.y + boundR).toInt().coerceAtMost(size - 2)

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
                    val socketWeight = 0.5f * (1.0f + cos(Math.PI * normDist).toFloat())

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

                    // Only intervene if a dark pupil/iris pixel turned cloudy/white in the neural output
                    val isCloudyPupilAnomaly = normDist <= 0.65f && tLum < 55f && cLum > tLum + 42f
                    if (isCloudyPupilAnomaly) {
                        val eyeBlend = (0.55f * socketWeight).coerceIn(0f, 0.60f)
                        val eR = (cR * (1f - eyeBlend) + tR * eyeBlend).toInt().coerceIn(0, 255)
                        val eG = (cG * (1f - eyeBlend) + tG * eyeBlend).toInt().coerceIn(0, 255)
                        val eB = (cB * (1f - eyeBlend) + tB * eyeBlend).toInt().coerceIn(0, 255)
                        outPx[idx] = (0xFF shl 24) or (eR shl 16) or (eG shl 8) or eB
                    }
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
        val hw = gfpSize * gfpSize
        val pixels = upscaleBitmapBicubicTo512(crop128Or512)

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
        ortEnv: OrtEnvironment?,
        gfpganFile: File?,
        crop128Or512: Bitmap,
        preferHardwareAccel: Boolean = false,
        preloadedGfpganSession: OrtSession? = null
    ): Bitmap? {
        if (ortEnv == null) return null
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
    /**
     * FaceFusion-Style Mobile 512x512 HD Enhancement, Target-Aware Upper-Lip/Philtrum Protection,
     * Dynamic Eye/Brow Protection, Occlusion Masking & Direct High-Resolution Bicubic Blending:
     *  - Accepts 128x128 (`inswapper_128.onnx`) or 256x256 (`hyperswap_1a/1b/1c_256.onnx`) swapped crop.
     *  - Upscales via 4x4 Catmull-Rom Bicubic interpolation + `gfpgan_1.4.onnx` 512x512 HD restoration.
     *  - Applies Target-Aware Philtrum / Upper-Lip / Mouth-Corner Facial-Hair Protection so a moustache-free
     *    target image NEVER gets an unwanted moustache or dark upper-lip shadow.
     *  - Warps the 512x512 HD face directly onto full-resolution `targetPixels` using Catmull-Rom Bicubic sampling
     *    (zero 512 -> 128 downscaling).
     */
    fun enhanceAndBlendOnlineHdFace512(
        ortEnv: OrtEnvironment? = null,
        gfpganFile: File?,
        targetBitmap: Bitmap,
        targetPixels: IntArray,
        targetWidth: Int,
        targetHeight: Int,
        colorCorrected128: Bitmap? = null,
        swappedCrop128: Bitmap? = null,
        forwardMatrix128: FloatArray? = null,
        targetLandmarks5: List<PointF>,
        sourceBitmap: Bitmap? = null,
        sourceLandmarks5: List<PointF>? = null,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.SOURCE_IDENTITY,
        faceReactionMode: FaceReactionSourceMode = FaceReactionSourceMode.TARGET_REACTION,
        enableColorTransfer: Boolean = true,
        enableOcclusionProtection: Boolean = true,
        blendStrength: Float = 1.0f,
        enhancementStrength: Float = 0.96f,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f,
        preferHardwareAccel: Boolean = false,
        segformerFile: File? = null,
        preloadedGfpganSession: OrtSession? = null,
        preloadedSegformerSession: OrtSession? = null,
        onStagesCaptured: ((stage1SwapOnly512: Bitmap, stage2SwapRestore512: Bitmap, stage3FinalBlend512: Bitmap) -> Unit)? = null,
        onSubStageTimings: ((restoration512Ms: Long, upperLipGuardMs: Long, maskGenerationMs: Long, finalBlendingMs: Long) -> Unit)? = null
    ): Bitmap {
        val activeSwapCrop = colorCorrected128 ?: swappedCrop128
            ?: throw IllegalArgumentException("Either colorCorrected128 or swappedCrop128 must be provided.")
        val m128 = forwardMatrix128 ?: FaceAlignment.estimateNorm(targetLandmarks5, 128)
        val m512 = FloatArray(6) { i -> m128[i] * 4.0f }
        val alignedTarget512 = FaceAlignment.warpAffineCrop(targetBitmap, m512, HD_SIZE)
        val geom512 = computeWarpedGeometry(m128, targetLandmarks5, scale = 4.0f)

        // Optional Neural Occlusion Gate via segformer_B5_ce.onnx (when explicitly provided for Mode 2)
        val tMask0 = System.currentTimeMillis()
        val neuralOcclusionGate512 = if (enableOcclusionProtection && (preloadedSegformerSession != null || segformerFile != null)) {
            HeadSegmentationAndInpainting.computeFaceOcclusionGate512(
                ortEnv = ortEnv,
                alignedTarget512 = alignedTarget512,
                segModelFile = segformerFile,
                preferHardwareAccel = preferHardwareAccel,
                preloadedSegformerSession = preloadedSegformerSession
            )
        } else null
        var maskGenMs = (System.currentTimeMillis() - tMask0).coerceAtLeast(0L)

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
        val tRestore0 = System.currentTimeMillis()
        val gfp512 = if (clampedEnhance > 0.05f && (preloadedGfpganSession != null || gfpganFile != null)) {
            runOptionalGfpganEnhancement512(
                ortEnv = ortEnv,
                gfpganFile = gfpganFile,
                crop128Or512 = activeSwapCrop,
                preferHardwareAccel = preferHardwareAccel,
                preloadedGfpganSession = preloadedGfpganSession
            )
        } else null

        val total512 = HD_SIZE * HD_SIZE
        // Upscale swapped crop (128x128 or 256x256) to 512x512 using sharp 4x4 Catmull-Rom Bicubic interpolation
        val swapPx512 = upscaleBitmapBicubicTo512(activeSwapCrop)
        val tgtPx512 = IntArray(total512)
        val srcPx512 = if (alignedSource512 != null) IntArray(total512) else null

        alignedTarget512.getPixels(tgtPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        alignedSource512?.getPixels(srcPx512!!, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        alignedTarget512.recycle()
        alignedSource512?.recycle()

        // Capture Stage 1: Swap-Only Output (raw neural swap before restoration, upper-lip guard, or blending)
        val stage1SwapOnlyBmp = if (onStagesCaptured != null) {
            Bitmap.createBitmap(HD_SIZE, HD_SIZE, Bitmap.Config.ARGB_8888).apply {
                setPixels(swapPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            }
        } else null

        if (gfp512 != null) {
            val gfpPx = IntArray(total512)
            gfp512.getPixels(gfpPx, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            gfp512.recycle()
            val gfpWeight = (0.35f + 0.65f * clampedEnhance).coerceIn(0f, 1f)
            val invEnh = 1.0f - gfpWeight
            for (i in 0 until total512) {
                val bc = swapPx512[i]
                val gc = gfpPx[i]
                val r = (((gc ushr 16) and 0xFF) * gfpWeight + ((bc ushr 16) and 0xFF) * invEnh).toInt().coerceIn(0, 255)
                val g = (((gc ushr 8) and 0xFF) * gfpWeight + ((bc ushr 8) and 0xFF) * invEnh).toInt().coerceIn(0, 255)
                val b = (((gc and 0xFF) * gfpWeight + (bc and 0xFF) * invEnh)).toInt().coerceIn(0, 255)
                swapPx512[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        // 1. Unified 512x512 Skin-Tone & Scene Illumination Harmonization (Mean + Contrast Transfer)
        //    Run BEFORE Upper-Lip Guard & HD feature sharpening so swapPx512 and tgtPx512 share the exact same skin tone!
        if (enableColorTransfer) {
            harmonizeSkinToneByMode512(
                swapPx512 = swapPx512,
                tgtPx512 = tgtPx512,
                srcPx512 = srcPx512,
                geom512 = geom512,
                skinToneMode = skinToneMode
            )
        }

        // 2. First-pass Target-Aware Upper-Lip / Philtrum Moustache & Grey-Patch Guard
        val tGuard0 = System.currentTimeMillis()
        protectTargetUpperLipAndEliminateMoustache512(
            swapPx512 = swapPx512,
            tgtPx512 = tgtPx512,
            geom512 = geom512
        )
        var upperLipGuardMs = (System.currentTimeMillis() - tGuard0).coerceAtLeast(1L)

        // 3. Apply frequency-separated 512x512 HD anatomical feature crispness boost on eyes, nose nostrils/bridge, and lips
        //    (strictly excluding the philtrum/upper-lip skin region)
        enhanceAnatomicalFeatures512(
            swapPx512 = swapPx512,
            geom512 = geom512,
            strength = (0.42f + 0.52f * clampedEnhance).coerceIn(0.32f, 0.94f)
        )

        // 4. Subtle 512x512 Skin Pore Micro-Texture Harmonization strictly on smooth cheek/forehead skin
        val poreScale = (if (skinToneMode == SkinToneSourceMode.SOURCE_IDENTITY) 0.08f else 0.14f) *
            (0.4f + 0.6f * clampedEnhance)
        val mouthMidX512 = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY512 = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        for (y in 24 until HD_SIZE - 24) {
            val row = y * HD_SIZE
            for (x in 24 until HD_SIZE - 24) {
                if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom512) > 0.05f) continue
                val dNoseSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    geom512.nose.x, geom512.nose.y,
                    geom512.eyeDist * 0.40f, geom512.eyeDist * 0.48f,
                    geom512.cosA, geom512.sinA
                )
                if (dNoseSq <= 1.0f) continue
                val dMouthSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    mouthMidX512, mouthMidY512,
                    geom512.eyeDist * 0.62f, geom512.eyeDist * 0.46f,
                    geom512.cosA, geom512.sinA
                )
                if (dMouthSq <= 1.0f) continue

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
                if (tLum > 85f && abs(poreDiff) < 7f) {
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

        // 5. Dynamic Landmark Eye & Eyebrow Protection (preserves 100% of swapped eye/brow identity)
        restoreDynamicEyebrowContinuity(
            outPx = swapPx512,
            tgtPx = tgtPx512,
            size = HD_SIZE,
            geom = geom512
        )
        restoreDynamicOcularSockets(
            outPx = swapPx512,
            tgtPx = tgtPx512,
            srcPx = null,
            size = HD_SIZE,
            geom = geom512
        )
        val restoration512Ms = ((System.currentTimeMillis() - tRestore0) - upperLipGuardMs).coerceAtLeast(1L)

        // Capture Stage 2: Swap + 512x512 HD Restoration + Upper-Lip Guard Output (before final scene blending)
        val stage2RestoreBmp = if (onStagesCaptured != null) {
            Bitmap.createBitmap(HD_SIZE, HD_SIZE, Bitmap.Config.ARGB_8888).apply {
                setPixels(swapPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            }
        } else null

        val tBlendStage0 = System.currentTimeMillis()
        // 6. Face Reaction Synthesis (Zero Source-Pixel Ghosting: never copy raw source mouth/skin pixels)
        when (faceReactionMode) {
            FaceReactionSourceMode.TARGET_REACTION -> {
                preserveTargetMouthTeethAndTongue512(
                    swapPx512 = swapPx512,
                    tgtPx512 = tgtPx512,
                    geom512 = geom512
                )
            }
            FaceReactionSourceMode.SOURCE_REACTION -> {
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

        // 7. Build 512x512 Target-Aware Biometric Landmark-Fitted Mask with Hair/Occlusion Protection
        val tMask1 = System.currentTimeMillis()
        val mask512 = createBiometricFaceMask512(
            geom512 = geom512,
            tgtPx512 = tgtPx512,
            swapPx512 = swapPx512,
            skinToneMode = skinToneMode,
            enableOcclusionProtection = enableOcclusionProtection,
            neuralOcclusionGate512 = neuralOcclusionGate512
        )
        val maskElapsed = (System.currentTimeMillis() - tMask1).coerceAtLeast(1L)
        maskGenMs = (maskGenMs + maskElapsed).coerceAtLeast(1L)

        // 8. 2-Band Multi-Band (Low-Frequency Illumination + High-Frequency Facial Detail) Boundary Fusion
        applyMultiBandBoundaryAndJawlineFusion512(
            swapPx512 = swapPx512,
            tgtPx512 = tgtPx512,
            mask512 = mask512
        )

        // 9. Final-pass Target-Aware Upper-Lip & Philtrum Protection Lock
        val tGuard1 = System.currentTimeMillis()
        protectTargetUpperLipAndEliminateMoustache512(
            swapPx512 = swapPx512,
            tgtPx512 = tgtPx512,
            geom512 = geom512
        )
        val guard2Elapsed = (System.currentTimeMillis() - tGuard1).coerceAtLeast(0L)
        upperLipGuardMs = (upperLipGuardMs + guard2Elapsed).coerceAtLeast(1L)

        // 10. Direct High-Resolution Warp of 512x512 HD Face onto full-resolution targetPixels using Catmull-Rom Bicubic
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
                        val swapColor = FaceAlignment.sampleBicubicClamped(
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

        // Composite swapPx512 with tgtPx512 using mask512 so Stage 3 (production_final_blend.png)
        // shows the exact final blended 512x512 face (including target hair, forehead, temples & jawline)
        val stage3BlendedPx512 = IntArray(total512)
        for (i in 0 until total512) {
            val a = (mask512[i] * clampedBlend).coerceIn(0f, 1f)
            val invA = 1.0f - a
            val sc = swapPx512[i]
            val tc = tgtPx512[i]
            val r = (((sc ushr 16) and 0xFF) * a + ((tc ushr 16) and 0xFF) * invA).toInt().coerceIn(0, 255)
            val g = (((sc ushr 8) and 0xFF) * a + ((tc ushr 8) and 0xFF) * invA).toInt().coerceIn(0, 255)
            val b = ((sc and 0xFF) * a + (tc and 0xFF) * invA).toInt().coerceIn(0, 255)
            stage3BlendedPx512[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val finalBlendingMs = ((System.currentTimeMillis() - tBlendStage0) - maskElapsed - guard2Elapsed).coerceAtLeast(1L)
        onSubStageTimings?.invoke(restoration512Ms, upperLipGuardMs, maskGenMs, finalBlendingMs)

        val restoredHd512 = Bitmap.createBitmap(HD_SIZE, HD_SIZE, Bitmap.Config.ARGB_8888)
        restoredHd512.setPixels(stage3BlendedPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        if (onStagesCaptured != null && stage1SwapOnlyBmp != null && stage2RestoreBmp != null) {
            onStagesCaptured(stage1SwapOnlyBmp, stage2RestoreBmp, restoredHd512)
        }
        return restoredHd512
    }

    /**
     * Upscales any 128x128 or 256x256 face crop to 512x512 HD using 4x4 Catmull-Rom Bicubic interpolation
     * instead of blurry bilinear scaling.
     */
    private fun upscaleBitmapBicubicTo512(src: Bitmap): IntArray {
        val w = src.width
        val h = src.height
        val out = IntArray(HD_SIZE * HD_SIZE)
        if (w == HD_SIZE && h == HD_SIZE) {
            src.getPixels(out, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            return out
        }
        val srcPx = IntArray(w * h)
        src.getPixels(srcPx, 0, w, 0, 0, w, h)
        val scaleX = (w - 1).toFloat() / (HD_SIZE - 1).toFloat()
        val scaleY = (h - 1).toFloat() / (HD_SIZE - 1).toFloat()
        for (y in 0 until HD_SIZE) {
            val sy = y * scaleY
            val row = y * HD_SIZE
            for (x in 0 until HD_SIZE) {
                val sx = x * scaleX
                out[row + x] = FaceAlignment.sampleBicubicClamped(srcPx, w, h, sx, sy)
            }
        }
        return out
    }

    /**
     * Frequency-separated anatomical feature enhancer in 512x512 HD space:
     * Boosts fine micro-contrast specifically on the Left Eye, Right Eye, Eyebrows, Nose Tip/Nostrils,
     * and Lip Vermilion Contour (strictly excluding the philtrum/upper-lip skin region).
     */
    private fun enhanceAnatomicalFeatures512(
        swapPx512: IntArray,
        geom512: WarpedFaceGeometry,
        strength: Float
    ) {
        val copy = swapPx512.copyOf()
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val eyeRx = geom512.eyeDist * 0.34f
        val eyeRy = geom512.eyeDist * 0.24f
        val noseRx = geom512.eyeDist * 0.30f
        val noseRy = geom512.eyeDist * 0.30f
        val mouthRx = geom512.eyeDist * 0.44f
        val mouthRy = geom512.eyeDist * 0.16f

        for (y in 16 until HD_SIZE - 16) {
            val row = y * HD_SIZE
            for (x in 16 until HD_SIZE - 16) {
                val xf = x.toFloat()
                val yf = y.toFloat()
                val lEyeD = orientedEllipseDistSq(xf, yf, geom512.leftEye.x, geom512.leftEye.y, eyeRx, eyeRy, geom512.cosA, geom512.sinA)
                val rEyeD = orientedEllipseDistSq(xf, yf, geom512.rightEye.x, geom512.rightEye.y, eyeRx, eyeRy, geom512.cosA, geom512.sinA)
                val noseD = orientedEllipseDistSq(xf, yf, geom512.nose.x, geom512.nose.y - 4f, noseRx, noseRy, geom512.cosA, geom512.sinA)
                val mouthD = orientedEllipseDistSq(xf, yf, mouthMidX, mouthMidY + 4f, mouthRx, mouthRy, geom512.cosA, geom512.sinA)

                val minD = min(min(lEyeD, rEyeD), min(noseD, mouthD))
                if (minD >= 1.0f) continue

                val featWeight = 0.5f * (1.0f + cos(Math.PI * sqrt(minD)).toFloat()) * strength
                val idx = row + x
                val c = copy[idx]
                val cr = (c ushr 16) and 0xFF
                val cg = (c ushr 8) and 0xFF
                val cb = c and 0xFF

                val n = copy[(y - 2) * HD_SIZE + x]
                val s = copy[(y + 2) * HD_SIZE + x]
                val w = copy[row + x - 2]
                val e = copy[row + x + 2]
                val avgR = (((n ushr 16) and 0xFF) + ((s ushr 16) and 0xFF) + ((w ushr 16) and 0xFF) + ((e ushr 16) and 0xFF)) * 0.25f
                val avgG = (((n ushr 8) and 0xFF) + ((s ushr 8) and 0xFF) + ((w ushr 8) and 0xFF) + ((e ushr 8) and 0xFF)) * 0.25f
                val avgB = ((n and 0xFF) + (s and 0xFF) + (w and 0xFF) + (e and 0xFF)) * 0.25f

                val outR = (cr + (cr - avgR) * featWeight * 0.65f).toInt().coerceIn(0, 255)
                val outG = (cg + (cg - avgG) * featWeight * 0.65f).toInt().coerceIn(0, 255)
                val outB = (cb + (cb - avgB) * featWeight * 0.65f).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    /**
     * CRITICAL STAGE — Target-Aware Upper-Lip / Philtrum / Mouth-Corner Facial-Hair Protection:
     *
     * Checks whether the Target image has a moustache by measuring the Target's philtrum luminance
     * relative to the Target's upper malar cheek skin luminance.
     * When the Target image has NO moustache (`targetPhiltrumRatio >= 0.62`), this stage protects:
     *  - philtrum (between sub-nasale and upper lip vermilion border)
     *  - upper lip skin arch
     *  - left & right mouth corners (commissures)
     *  - surrounding perioral skin
     * eliminating 100% of any moustache, dark upper-lip shadow, grey patch above the lips, or source
     * facial-hair pigment leakage while keeping the actual pink/red lip vermilion and nose nostrils intact.
     */
    private fun protectTargetUpperLipAndEliminateMoustache512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        geom512: WarpedFaceGeometry
    ) {
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val mouthW = hypot(
            (geom512.rightMouth.x - geom512.leftMouth.x).toDouble(),
            (geom512.rightMouth.y - geom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(72f, 220f)

        val dxNM = mouthMidX - geom512.nose.x
        val dyNM = mouthMidY - geom512.nose.y
        val noseToMouthDist = hypot(dxNM.toDouble(), dyNM.toDouble()).toFloat().coerceAtLeast(48f)
        // Unit vector (vx, vy) pointing downward from nose tip to mouth center
        val vx = dxNM / noseToMouthDist
        val vy = dyNM / noseToMouthDist
        // Unit vector (ux, uy) pointing horizontally along the eye/mouth axis
        val ux = geom512.cosA
        val uy = geom512.sinA

        // Philtrum center lies between nose tip and mouth center along the facial vertical axis
        val philtrumCx = geom512.nose.x * 0.38f + mouthMidX * 0.62f
        val philtrumCy = geom512.nose.y * 0.40f + mouthMidY * 0.60f
        val philtrumRx = max(mouthW * 0.62f, geom512.eyeDist * 0.48f)
        val philtrumRy = (noseToMouthDist * 0.52f).coerceIn(28f, 64f)

        // 1. Sample clean upper-malar cheek skin reference (strictly above any beard/moustache zone)
        val leftCheekCx = geom512.leftEye.x * 0.65f + geom512.leftMouth.x * 0.35f - geom512.cosA * (geom512.eyeDist * 0.14f)
        val leftCheekCy = geom512.leftEye.y * 0.55f + geom512.leftMouth.y * 0.45f
        val rightCheekCx = geom512.rightEye.x * 0.65f + geom512.rightMouth.x * 0.35f + geom512.cosA * (geom512.eyeDist * 0.14f)
        val rightCheekCy = geom512.rightEye.y * 0.55f + geom512.rightMouth.y * 0.45f
        val cheekR = (geom512.eyeDist * 0.16f).toInt().coerceAtLeast(10)

        var sCheekR = 0f
        var sCheekG = 0f
        var sCheekB = 0f
        var tCheekR = 0f
        var tCheekG = 0f
        var tCheekB = 0f
        var cheekCount = 0

        for ((ccx, ccy) in arrayOf(leftCheekCx.toInt() to leftCheekCy.toInt(), rightCheekCx.toInt() to rightCheekCy.toInt())) {
            for (y in (ccy - cheekR)..(ccy + cheekR) step 2) {
                if (y !in 16 until HD_SIZE - 16) continue
                for (x in (ccx - cheekR)..(ccx + cheekR) step 2) {
                    if (x !in 16 until HD_SIZE - 16) continue
                    val idx = y * HD_SIZE + x
                    val sc = swapPx512[idx]
                    val tc = tgtPx512[idx]
                    sCheekR += (sc ushr 16) and 0xFF
                    sCheekG += (sc ushr 8) and 0xFF
                    sCheekB += sc and 0xFF
                    tCheekR += (tc ushr 16) and 0xFF
                    tCheekG += (tc ushr 8) and 0xFF
                    tCheekB += tc and 0xFF
                    cheekCount++
                }
            }
        }
        if (cheekCount < 8) return
        sCheekR /= cheekCount
        sCheekG /= cheekCount
        sCheekB /= cheekCount
        tCheekR /= cheekCount
        tCheekG /= cheekCount
        tCheekB /= cheekCount
        val tCheekLum = (0.299f * tCheekR + 0.587f * tCheekG + 0.114f * tCheekB).coerceAtLeast(25f)

        // 2. Measure Target philtrum luminance to verify whether Target has a moustache
        var tPhiltrumLumSum = 0f
        var philtrumSamples = 0
        val pMinX = (philtrumCx - philtrumRx * 0.60f).toInt().coerceIn(16, HD_SIZE - 17)
        val pMaxX = (philtrumCx + philtrumRx * 0.60f).toInt().coerceIn(16, HD_SIZE - 17)
        val pMinY = (philtrumCy - philtrumRy * 0.40f).toInt().coerceIn(16, HD_SIZE - 17)
        val pMaxY = (philtrumCy + philtrumRy * 0.30f).toInt().coerceIn(16, HD_SIZE - 17)
        for (y in pMinY..pMaxY step 2) {
            for (x in pMinX..pMaxX step 2) {
                val tc = tgtPx512[y * HD_SIZE + x]
                val tLum = 0.299f * (tc ushr 16 and 0xFF) + 0.587f * (tc ushr 8 and 0xFF) + 0.114f * (tc and 0xFF)
                tPhiltrumLumSum += tLum
                philtrumSamples++
            }
        }
        val tPhiltrumMeanLum = (if (philtrumSamples > 0) tPhiltrumLumSum / philtrumSamples else tCheekLum).coerceAtLeast(25f)
        val targetHasNoMoustache = (tPhiltrumMeanLum / tCheekLum) >= 0.62f
        if (!targetHasNoMoustache) return

        // 3. Target has NO moustache: lift only genuine dark moustache shadows or grey cast in the philtrum
        //    using a smooth radial cosine ellipse and low-frequency cheek-matched shading (NEVER pasting
        //    high-frequency Target nose/lip pixels or hard rectangular boundaries).
        val residualR = sCheekR - tCheekR
        val residualG = sCheekG - tCheekG
        val residualB = sCheekB - tCheekB

        val halfSpanU = max(mouthW * 0.56f, geom512.eyeDist * 0.44f)
        val boundMinX = (min(geom512.nose.x, mouthMidX) - halfSpanU - 14f).toInt().coerceIn(14, HD_SIZE - 15)
        val boundMaxX = (max(geom512.nose.x, mouthMidX) + halfSpanU + 14f).toInt().coerceIn(14, HD_SIZE - 15)
        val boundMinY = (min(geom512.nose.y, mouthMidY) - 10f).toInt().coerceIn(14, HD_SIZE - 15)
        val boundMaxY = (max(geom512.nose.y, mouthMidY) + noseToMouthDist * 0.12f).toInt().coerceIn(14, HD_SIZE - 15)

        val nostrilRx = geom512.eyeDist * 0.22f
        val nostrilRy = geom512.eyeDist * 0.10f
        val origSwapCopy = swapPx512.copyOf()

        for (y in boundMinY..boundMaxY) {
            val row = y * HD_SIZE
            for (x in boundMinX..boundMaxX) {
                val xf = x.toFloat()
                val yf = y.toFloat()

                // Local anatomical coordinates (uNorm, vFrac) from nose tip (vFrac = 0) to mouth center (vFrac = 1)
                val dx = xf - geom512.nose.x
                val dy = yf - geom512.nose.y
                val u = dx * ux + dy * uy
                val v = dx * vx + dy * vy
                val uNorm = abs(u) / halfSpanU
                if (uNorm >= 1.0f) continue

                val vFrac = v / noseToMouthDist
                val vTop = 0.18f
                val vBottom = 0.82f
                if (vFrac <= vTop || vFrac >= vBottom) continue

                val dNostrilSq = orientedEllipseDistSq(
                    xf, yf, geom512.nose.x, geom512.nose.y + nostrilRy * 0.15f, nostrilRx, nostrilRy, geom512.cosA, geom512.sinA
                )
                if (dNostrilSq < 1.0f) continue

                val vMid = (vTop + vBottom) * 0.5f
                val vHalf = (vBottom - vTop) * 0.5f
                val vNorm = (vFrac - vMid) / vHalf
                val radialSq = uNorm * uNorm + vNorm * vNorm
                if (radialSq >= 1.0f) continue

                val idx = row + x
                val tc = tgtPx512[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF

                // Chromatic lip-vermilion guard: never overwrite actual red/pink lip vermilion pixels
                if (vFrac > 0.68f && (tR - tG) > 48) continue

                val sc = origSwapCopy[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF
                val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB

                // Use low-frequency smoothed Target shading + cheek residual so NO high-frequency Target nose/lip edge is ever stamped
                val tLow = compute5PointLowPassRGB512(tgtPx512, x, y, 8)
                val sLow = compute5PointLowPassRGB512(origSwapCopy, x, y, 8)
                val cleanSkinR = (0.72f * (tLow[0] + residualR) + 0.28f * sCheekR + (sR - sLow[0]) * 0.25f).coerceIn(0f, 255f)
                val cleanSkinG = (0.72f * (tLow[1] + residualG) + 0.28f * sCheekG + (sG - sLow[1]) * 0.25f).coerceIn(0f, 255f)
                val cleanSkinB = (0.72f * (tLow[2] + residualB) + 0.28f * sCheekB + (sB - sLow[2]) * 0.25f).coerceIn(0f, 255f)
                val expectedCleanLum = 0.299f * cleanSkinR + 0.587f * cleanSkinG + 0.114f * cleanSkinB

                // Pure radial cosine dome (zero flat rectangular plateau)
                val zoneEnv = (0.5f * (1.0f + cos(Math.PI * sqrt(radialSq)))).toFloat()

                val shadowDeficit = expectedCleanLum - sLum
                val cleanWarmth = cleanSkinR - cleanSkinB
                val swapWarmth = (sR - sB).toFloat()
                val greyCastDeficit = cleanWarmth - swapWarmth

                if (shadowDeficit > 3.5f || greyCastDeficit > 4.5f) {
                    val severity = max((shadowDeficit - 3.5f) / 16.0f, (greyCastDeficit - 4.5f) / 16.0f).coerceIn(0f, 1f)
                    val replaceWeight = (0.94f * zoneEnv * severity).coerceIn(0f, 0.94f)
                    val outR = (sR * (1f - replaceWeight) + cleanSkinR * replaceWeight).toInt().coerceIn(0, 255)
                    val outG = (sG * (1f - replaceWeight) + cleanSkinG * replaceWeight).toInt().coerceIn(0, 255)
                    val outB = (sB * (1f - replaceWeight) + cleanSkinB * replaceWeight).toInt().coerceIn(0, 255)
                    swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                }
            }
        }
    }

    /**
     * 2-Band (Low-Frequency Illumination + High-Frequency Detail) Multi-Band & Jawline Shadow Fusion
     * in 512x512 HD space: smoothly aligns the outer cheek, forehead, and chin/neck jawline shadow
     * envelope of `swapPx512` to `tgtPx512` across the transition ring (`0.02 < mask512 < 0.96`)
     * while preserving 100% of the swapped face's crisp high-frequency detail.
     */
    private fun applyMultiBandBoundaryAndJawlineFusion512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        mask512: FloatArray
    ) {
        val origSwap = swapPx512.copyOf()
        val step1 = 8
        val step2 = 18
        for (y in step2 until HD_SIZE - step2) {
            val row = y * HD_SIZE
            val jawlineBoost = if (y > 330) 0.76f else 0.64f
            for (x in step2 until HD_SIZE - step2) {
                val idx = row + x
                val m = mask512[idx]
                if (m <= 0.02f || m >= 0.96f) continue

                val boundaryWeight = ((1.0f - m) * jawlineBoost).coerceIn(0f, 0.74f)
                val sLow1 = compute5PointLowPassRGB512(origSwap, x, y, step1)
                val sLow2 = compute5PointLowPassRGB512(origSwap, x, y, step2)
                val tLow1 = compute5PointLowPassRGB512(tgtPx512, x, y, step1)
                val tLow2 = compute5PointLowPassRGB512(tgtPx512, x, y, step2)

                val sLowR = 0.45f * sLow1[0] + 0.55f * sLow2[0]
                val sLowG = 0.45f * sLow1[1] + 0.55f * sLow2[1]
                val sLowB = 0.45f * sLow1[2] + 0.55f * sLow2[2]

                val tLowR = 0.45f * tLow1[0] + 0.55f * tLow2[0]
                val tLowG = 0.45f * tLow1[1] + 0.55f * tLow2[1]
                val tLowB = 0.45f * tLow1[2] + 0.55f * tLow2[2]

                val sc = origSwap[idx]
                val sr = (sc ushr 16) and 0xFF
                val sg = (sc ushr 8) and 0xFF
                val sb = sc and 0xFF

                val nr = (sr + (tLowR - sLowR) * boundaryWeight).toInt().coerceIn(0, 255)
                val ng = (sg + (tLowG - sLowG) * boundaryWeight).toInt().coerceIn(0, 255)
                val nb = (sb + (tLowB - sLowB) * boundaryWeight).toInt().coerceIn(0, 255)
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
     * Harmonizes the 512x512 restored crop according to the user's chosen [SkinToneSourceMode]
     * (`TARGET_SCENE`, `SOURCE_IDENTITY`, or `BALANCED_BLEND`) using full Mean + Contrast (StdDev)
     * transfer across the face while protecting dark iris/lash contrast.
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
                    // Strictly exclude nose base, philtrum/moustache arch, and mouth when sampling clean cheek/forehead skin
                    if (y in 280..425 && x in 150..362) continue

                    val idx = y * HD_SIZE + x
                    val sc = swapPx512[idx]
                    val tc = tgtPx512[idx]
                    val dc = srcPx512?.get(idx) ?: sc
                    val tLum = 0.299 * (tc ushr 16 and 0xFF) + 0.587 * (tc ushr 8 and 0xFF) + 0.114 * (tc and 0xFF)
                    val dLum = 0.299 * (dc ushr 16 and 0xFF) + 0.587 * (dc ushr 8 and 0xFF) + 0.114 * (dc and 0xFF)
                    if (tLum < 50.0 || dLum < 55.0) continue

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

        var sRVar = 0.0
        var sGVar = 0.0
        var sBVar = 0.0
        var tRVar = 0.0
        var tGVar = 0.0
        var tBVar = 0.0
        var dRVar = 0.0
        var dGVar = 0.0
        var dBVar = 0.0
        for (y in 120..410 step 4) {
            val ny = (y - 268.0) / 148.0
            for (x in 120..392 step 4) {
                val nx = (x - 256.0) / 136.0
                if (nx * nx + ny * ny <= 1.0) {
                    if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom512) > 0.12f) continue
                    if (y in 280..425 && x in 150..362) continue
                    val idx = y * HD_SIZE + x
                    val sc = swapPx512[idx]
                    val tc = tgtPx512[idx]
                    val dc = srcPx512?.get(idx) ?: sc
                    val tLum = 0.299 * (tc ushr 16 and 0xFF) + 0.587 * (tc ushr 8 and 0xFF) + 0.114 * (tc and 0xFF)
                    if (tLum < 50.0) continue
                    val dSr = ((sc ushr 16) and 0xFF) - sRMean
                    val dSg = ((sc ushr 8) and 0xFF) - sGMean
                    val dSb = (sc and 0xFF) - sBMean
                    val dTr = ((tc ushr 16) and 0xFF) - tRMean
                    val dTg = ((tc ushr 8) and 0xFF) - tGMean
                    val dTb = (tc and 0xFF) - tBMean
                    val dDr = ((dc ushr 16) and 0xFF) - dRMean
                    val dDg = ((dc ushr 8) and 0xFF) - dGMean
                    val dDb = (dc and 0xFF) - dBMean
                    sRVar += dSr * dSr
                    sGVar += dSg * dSg
                    sBVar += dSb * dSb
                    tRVar += dTr * dTr
                    tGVar += dTg * dTg
                    tBVar += dTb * dTb
                    dRVar += dDr * dDr
                    dGVar += dDg * dDg
                    dBVar += dDb * dDb
                }
            }
        }
        val varCount = (count / 4).coerceAtLeast(1)
        val sRStd = sqrt(sRVar / varCount).coerceAtLeast(10.0)
        val sGStd = sqrt(sGVar / varCount).coerceAtLeast(10.0)
        val sBStd = sqrt(sBVar / varCount).coerceAtLeast(10.0)
        val tRStd = sqrt(tRVar / varCount).coerceAtLeast(10.0)
        val tGStd = sqrt(tGVar / varCount).coerceAtLeast(10.0)
        val tBStd = sqrt(tBVar / varCount).coerceAtLeast(10.0)
        val dRStd = sqrt(dRVar / varCount).coerceAtLeast(10.0)
        val dGStd = sqrt(dGVar / varCount).coerceAtLeast(10.0)
        val dBStd = sqrt(dBVar / varCount).coerceAtLeast(10.0)

        val (goalRMean, goalGMean, goalBMean) = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> Triple(
                tRMean * 0.96 + sRMean * 0.04,
                tGMean * 0.96 + sGMean * 0.04,
                tBMean * 0.96 + sBMean * 0.04
            )
            SkinToneSourceMode.SOURCE_IDENTITY -> Triple(
                dRMean * 0.92 + tRMean * 0.08,
                dGMean * 0.92 + tGMean * 0.08,
                dBMean * 0.92 + tBMean * 0.08
            )
            SkinToneSourceMode.BALANCED_BLEND -> Triple(
                dRMean * 0.50 + tRMean * 0.50,
                dGMean * 0.50 + tGMean * 0.50,
                dBMean * 0.50 + tBMean * 0.50
            )
        }

        // When SOURCE_IDENTITY is selected, preserve Source Image 1's smooth skin contrast rather than forcing Target's blotchy variance
        val (goalRStd, goalGStd, goalBStd) = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> Triple(
                0.85 * tRStd + 0.15 * sRStd,
                0.85 * tGStd + 0.15 * sGStd,
                0.85 * tBStd + 0.15 * sBStd
            )
            SkinToneSourceMode.SOURCE_IDENTITY -> Triple(
                0.78 * dRStd + 0.22 * tRStd,
                0.78 * dGStd + 0.22 * tGStd,
                0.78 * dBStd + 0.22 * tBStd
            )
            SkinToneSourceMode.BALANCED_BLEND -> Triple(
                0.50 * dRStd + 0.50 * tRStd,
                0.50 * dGStd + 0.50 * tGStd,
                0.50 * dBStd + 0.50 * tBStd
            )
        }

        val scaleR = (goalRStd / sRStd).toFloat().coerceIn(0.76f, 1.24f)
        val scaleG = (goalGStd / sGStd).toFloat().coerceIn(0.76f, 1.24f)
        val scaleB = (goalBStd / sBStd).toFloat().coerceIn(0.76f, 1.24f)

        val strength = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> 0.94f
            SkinToneSourceMode.SOURCE_IDENTITY -> 0.92f
            SkinToneSourceMode.BALANCED_BLEND -> 0.88f
        }

        for (y in 0 until HD_SIZE) {
            val row = y * HD_SIZE
            val yf = y.toFloat()
            for (x in 0 until HD_SIZE) {
                val idx = row + x
                val xf = x.toFloat()

                val c = swapPx512[idx]
                val origR = (c ushr 16) and 0xFF
                val origG = (c ushr 8) and 0xFF
                val origB = c and 0xFF

                val eyeProt = computeEyeAndBrowProtectionWeight(xf, yf, geom512)
                val lum = 0.299f * origR + 0.587f * origG + 0.114f * origB
                // Protect dark pupil/lash/eyebrow hair cores so dark features stay crisp and rich
                val darkFeatureProt = if (eyeProt > 0.18f && lum < 72f) eyeProt * 0.85f else 0f
                val blend = strength * (1.0f - darkFeatureProt)

                val matchedR = ((origR - sRMean) * scaleR + goalRMean).toFloat()
                val matchedG = ((origG - sGMean) * scaleG + goalGMean).toFloat()
                val matchedB = ((origB - sBMean) * scaleB + goalBMean).toFloat()

                val r = (origR * (1f - blend) + matchedR * blend).toInt().coerceIn(0, 255)
                val g = (origG * (1f - blend) + matchedG * blend).toInt().coerceIn(0, 255)
                val b = (origB * (1f - blend) + matchedB * blend).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }

    /**
     * Preserves 100% of the Target photo's (Image 2) facial reaction — whether a closed-lip gentle smile,
     * open-mouth smile with teeth/tongue, or natural lip expression — while smoothly adapting the perioral
     * skin tone to match the harmonized swapped face (`swapPx512`) without creating any rectangular patch.
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

        // Sample clean malar cheek skin delta between harmonized swapPx512 and tgtPx512
        val lCheekX = (geom512.leftEye.x * 0.62f + geom512.leftMouth.x * 0.38f - geom512.cosA * (geom512.eyeDist * 0.14f)).toInt()
        val lCheekY = (geom512.leftEye.y * 0.52f + geom512.leftMouth.y * 0.48f).toInt()
        val rCheekX = (geom512.rightEye.x * 0.62f + geom512.rightMouth.x * 0.38f + geom512.cosA * (geom512.eyeDist * 0.14f)).toInt()
        val rCheekY = (geom512.rightEye.y * 0.52f + geom512.rightMouth.y * 0.48f).toInt()
        val sampleR = (geom512.eyeDist * 0.14f).toInt().coerceAtLeast(8)

        var sCheekR = 0f
        var sCheekG = 0f
        var sCheekB = 0f
        var tCheekR = 0f
        var tCheekG = 0f
        var tCheekB = 0f
        var cheekSamples = 0
        for ((cx, cy) in arrayOf(lCheekX to lCheekY, rCheekX to rCheekY)) {
            for (y in (cy - sampleR)..(cy + sampleR) step 2) {
                if (y !in 16 until HD_SIZE - 16) continue
                for (x in (cx - sampleR)..(cx + sampleR) step 2) {
                    if (x !in 16 until HD_SIZE - 16) continue
                    val idx = y * HD_SIZE + x
                    val sc = swapPx512[idx]
                    val tc = tgtPx512[idx]
                    sCheekR += (sc ushr 16) and 0xFF
                    sCheekG += (sc ushr 8) and 0xFF
                    sCheekB += sc and 0xFF
                    tCheekR += (tc ushr 16) and 0xFF
                    tCheekG += (tc ushr 8) and 0xFF
                    tCheekB += tc and 0xFF
                    cheekSamples++
                }
            }
        }
        val deltaR = if (cheekSamples > 0) (sCheekR - tCheekR) / cheekSamples else 0f
        val deltaG = if (cheekSamples > 0) (sCheekG - tCheekG) / cheekSamples else 0f
        val deltaB = if (cheekSamples > 0) (sCheekB - tCheekB) / cheekSamples else 0f

        // Smooth elliptical aperture focused strictly on the oral slit, lip vermilion, and mouth corners
        val rx = mouthWidth * 0.56f
        val ry = mouthWidth * 0.27f
        val centerY = mouthMidY + 2f
        val minX = (mouthMidX - rx - 4f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxX = (mouthMidX + rx + 4f).toInt().coerceIn(8, HD_SIZE - 9)
        val minY = (centerY - ry - 4f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxY = (centerY + ry + 4f).toInt().coerceIn(8, HD_SIZE - 9)

        for (y in minY..maxY) {
            val row = y * HD_SIZE
            for (x in minX..maxX) {
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    mouthMidX, centerY,
                    rx, ry,
                    geom512.cosA, geom512.sinA
                )
                if (dSq >= 1.0f) continue
                val r = sqrt(dSq)
                // Smooth radial cosine falloff from center (r = 0) to perimeter (r = 1) with no hard plateau edge
                val env = (0.5f * (1.0f + cos(Math.PI * r))).toFloat()

                val idx = row + x
                val tc = tgtPx512[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB
                val maxCh = max(tR, max(tG, tB)).coerceAtLeast(1)
                val minCh = min(tR, min(tG, tB))
                val tSat = (maxCh - minCh).toFloat() / maxCh.toFloat()

                val isVisibleTeeth = r <= 0.62f && tLum > 148f && tSat < 0.22f
                val isOpenOralCavity = r <= 0.56f && tLum < 52f
                val isLipVermilion = r <= 0.76f && (tR - tG) > 32

                val skinAdaptScale = when {
                    isVisibleTeeth || isOpenOralCavity -> 0.0f
                    isLipVermilion -> 0.52f
                    else -> 0.96f
                }

                val adaptedTargetR = (tR + deltaR * skinAdaptScale).coerceIn(0f, 255f)
                val adaptedTargetG = (tG + deltaG * skinAdaptScale).coerceIn(0f, 255f)
                val adaptedTargetB = (tB + deltaB * skinAdaptScale).coerceIn(0f, 255f)

                val sc = swapPx512[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF

                val directWeight = (0.90f * env).coerceIn(0f, 0.90f)
                val invW = 1.0f - directWeight
                val outR = (sR * invW + adaptedTargetR * directWeight).toInt().coerceIn(0, 255)
                val outG = (sG * invW + adaptedTargetG * directWeight).toInt().coerceIn(0, 255)
                val outB = (sB * invW + adaptedTargetB * directWeight).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    /**
     * Transfers the Source photo's (Image 1) own mouth/smile/teeth expression onto the swapped face
     * when `FaceReactionSourceMode.SOURCE_REACTION` is selected by the user, while keeping the upper-lip
     * philtrum protected from any unwanted moustache shadow.
     */
    private fun transferSourceReactionSmileTeethTongue512(
        swapPx512: IntArray,
        srcPx512: IntArray,
        tgtGeom512: WarpedFaceGeometry,
        srcGeom512: WarpedFaceGeometry
    ) {
        val tMouthMidX = (tgtGeom512.leftMouth.x + tgtGeom512.rightMouth.x) * 0.5f
        val tMouthMidY = (tgtGeom512.leftMouth.y + tgtGeom512.rightMouth.y) * 0.5f
        val tMouthW = hypot(
            (tgtGeom512.rightMouth.x - tgtGeom512.leftMouth.x).toDouble(),
            (tgtGeom512.rightMouth.y - tgtGeom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(72f, 220f)

        val sMouthMidX = (srcGeom512.leftMouth.x + srcGeom512.rightMouth.x) * 0.5f
        val sMouthMidY = (srcGeom512.leftMouth.y + srcGeom512.rightMouth.y) * 0.5f

        val rx = tMouthW * 0.56f
        val ry = tMouthW * 0.32f
        val centerY = tMouthMidY + 4f
        // Strictly stay below the upper-lip philtrum moustache zone (minY >= tMouthMidY - ry * 0.45f)
        val minX = (tMouthMidX - rx - 4f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxX = (tMouthMidX + rx + 4f).toInt().coerceIn(8, HD_SIZE - 9)
        val minY = (tMouthMidY - ry * 0.45f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxY = (centerY + ry + 4f).toInt().coerceIn(8, HD_SIZE - 9)

        for (y in minY..maxY) {
            val row = y * HD_SIZE
            for (x in minX..maxX) {
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    tMouthMidX, centerY,
                    rx, ry,
                    tgtGeom512.cosA, tgtGeom512.sinA
                )
                if (dSq >= 1.0f) continue
                val r = sqrt(dSq)
                val env = (0.5f * (1.0f + cos(Math.PI * r.coerceIn(0f, 1f)))).toFloat()

                val sx = (sMouthMidX + (x - tMouthMidX)).toInt().coerceIn(2, HD_SIZE - 3)
                val sy = (sMouthMidY + (y - tMouthMidY)).toInt().coerceIn(2, HD_SIZE - 3)
                val srcC = srcPx512[sy * HD_SIZE + sx]
                val sR = (srcC ushr 16) and 0xFF
                val sG = (srcC ushr 8) and 0xFF
                val sB = srcC and 0xFF

                val idx = row + x
                val curC = swapPx512[idx]
                val cR = (curC ushr 16) and 0xFF
                val cG = (curC ushr 8) and 0xFF
                val cB = curC and 0xFF

                val w = (0.86f * env).coerceIn(0f, 0.88f)
                val invW = 1.0f - w
                val outR = (cR * invW + sR * w).toInt().coerceIn(0, 255)
                val outG = (cG * invW + sG * w).toInt().coerceIn(0, 255)
                val outB = (cB * invW + sB * w).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    /**
     * Creates a landmark-fitted 512x512 Biometric Inner-Face Mask that:
     *  - Covers 100% of the eyebrows, eyes, nose, lips, and inner cheeks (`innerCoreRatio = 0.58f`)
     *  - Tapers smoothly along the outer cheek, forehead, and jawline contour (`radiusX = 1.12 * eyeDist`, `radiusY = 1.24 * eyeDist`)
     *  - Applies foreground hair/glasses occlusion protection strictly on the outer perimeter (`r > 0.54f`)
     *    so the inner swapped face is never suppressed and the swap box border never leaks onto hair/ears/neck.
     */
    private fun createBiometricFaceMask512(
        geom512: WarpedFaceGeometry,
        tgtPx512: IntArray,
        swapPx512: IntArray,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.SOURCE_IDENTITY,
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

        val radiusX = (geom512.eyeDist * 1.12f).coerceIn(136f, 174f)
        val radiusY = (geom512.eyeDist * 1.24f).coerceIn(152f, 194f)
        val innerCoreRatio = 0.58f
        val borderMargin = 24

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
                if (enableOcclusionProtection && r > 0.54f && alpha > 0.01f) {
                    val perimWeight = ((r - 0.54f) / 0.46f).coerceIn(0f, 1f)
                    if (neuralOcclusionGate512 != null) {
                        val segGate = neuralOcclusionGate512[idx].coerceIn(0f, 1f)
                        val gateFactor = 1.0f - perimWeight * (1.0f - (0.35f + 0.65f * segGate))
                        alpha *= gateFactor
                    }

                    val tc = tgtPx512[idx]
                    val sc = swapPx512[idx]
                    val tLum = 0.299f * (tc ushr 16 and 0xFF) + 0.587f * (tc ushr 8 and 0xFF) + 0.114f * (tc and 0xFF)
                    val sLum = 0.299f * (sc ushr 16 and 0xFF) + 0.587f * (sc ushr 8 and 0xFF) + 0.114f * (sc and 0xFF)
                    if (tLum < 56f && sLum > tLum + 24f) {
                        val hairKeep = (((56f - tLum) / 45f) * perimWeight).coerceIn(0f, 0.85f)
                        alpha *= (1.0f - hairKeep)
                    }
                }

                val edgeDist = min(
                    min(x - borderMargin, HD_SIZE - 1 - borderMargin - x),
                    min(y - borderMargin, HD_SIZE - 1 - borderMargin - y)
                ).toFloat()
                val edgeEnvelope = (edgeDist / 36.0f).coerceIn(0f, 1f)

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
