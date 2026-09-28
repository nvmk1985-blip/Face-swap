package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class HeadPoseAndBounds(
    val face: DetectedFace,
    val cranialBox: RectF,
    val eyeDistance: Float,
    val rollDegrees: Float,
    val yawRatio: Float
)

data class HeadSegmentationMasks(
    val cropSize: Int,
    val fullHeadAlpha: FloatArray, // Union of face, skull, hair, ears, and upper neck in [0, 1]
    val skinAndNeckWeight: FloatArray, // 1.0 on face/neck skin, 0.0 on hair (used for selective skin-tone adaptation)
    val usedSegmentationOnnx: Boolean,
    val segmentationSourceLabel: String
)

/**
 * Implements head/cranial geometry estimation, head/hair/neck semantic segmentation & alpha matting
 * (supporting `segformer_B5_ce.onnx` and `modnet.onnx` with a built-in Chromatic-Geodesic +
 * Joint Bilateral Guided Matting fallback), and disocclusion background-gap inpainting.
 */
object HeadSegmentationAndInpainting {

    /**
     * Computes the canonical 5-point landmark template inside a `cropSize x cropSize` head canvas.
     * Unlike the tight 112x112 or 128x128 face-only crop, the Head Replacement canvas reserves:
     *  - Top 42% for forehead, cranial dome, and top hair volume
     *  - Left/Right 26% for ears and side hairstyles
     *  - Bottom 24% below the mouth for chin, jawline, and upper neck transition
     */
    fun getCanonicalHeadTemplate(cropSize: Int, yawRatio: Float = 0f): Array<PointF> {
        val s = cropSize / 256.0f
        // Shift horizontal center slightly opposite to yaw so side hair on the back of the head fits
        val yawShiftX = (-yawRatio * 8.0f * s).coerceIn(-14f * s, 14f * s)
        val base = arrayOf(
            PointF(98.0f * s + yawShiftX, 114.0f * s),  // Left Eye
            PointF(158.0f * s + yawShiftX, 114.0f * s), // Right Eye
            PointF(128.0f * s + yawShiftX, 146.0f * s), // Nose Tip
            PointF(103.0f * s + yawShiftX, 178.0f * s), // Left Mouth Corner
            PointF(153.0f * s + yawShiftX, 178.0f * s)  // Right Mouth Corner
        )
        return base
    }

    fun estimateHeadAlignmentMatrix(
        srcLandmarks5: List<PointF>,
        cropSize: Int,
        targetYawRatio: Float = 0f
    ): FloatArray {
        require(srcLandmarks5.size >= 5) { "5 facial keypoints required for head alignment." }
        val dstPoints = getCanonicalHeadTemplate(cropSize, targetYawRatio)
        val n = 5

        var srcMeanX = 0f
        var srcMeanY = 0f
        var dstMeanX = 0f
        var dstMeanY = 0f
        for (i in 0 until n) {
            srcMeanX += srcLandmarks5[i].x
            srcMeanY += srcLandmarks5[i].y
            dstMeanX += dstPoints[i].x
            dstMeanY += dstPoints[i].y
        }
        srcMeanX /= n
        srcMeanY /= n
        dstMeanX /= n
        dstMeanY /= n

        var numA = 0f
        var numB = 0f
        var denom = 0f
        for (i in 0 until n) {
            val u = srcLandmarks5[i].x - srcMeanX
            val v = srcLandmarks5[i].y - srcMeanY
            val xp = dstPoints[i].x - dstMeanX
            val yp = dstPoints[i].y - dstMeanY
            numA += u * xp + v * yp
            numB += u * yp - v * xp
            denom += u * u + v * v
        }
        if (denom < 1e-6f) return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)

        val a = numA / denom
        val b = numB / denom
        val tx = dstMeanX - (a * srcMeanX - b * srcMeanY)
        val ty = dstMeanY - (b * srcMeanX + a * srcMeanY)
        return floatArrayOf(a, -b, tx, b, a, ty)
    }

    fun analyzeHeadPoseAndBounds(
        face: DetectedFace,
        imageWidth: Int,
        imageHeight: Int
    ): HeadPoseAndBounds {
        val kps = face.landmarks5
        val leftEye = kps[0]
        val rightEye = kps[1]
        val nose = kps[2]
        val leftMouth = kps[3]
        val rightMouth = kps[4]

        val eyeMidX = (leftEye.x + rightEye.x) * 0.5f
        val eyeMidY = (leftEye.y + rightEye.y) * 0.5f
        val mouthMidY = (leftMouth.y + rightMouth.y) * 0.5f

        val dx = rightEye.x - leftEye.x
        val dy = rightEye.y - leftEye.y
        val eyeDist = hypot(dx.toDouble(), dy.toDouble()).toFloat().coerceAtLeast(8f)
        val rollDeg = Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()

        val distLeftNose = hypot((nose.x - leftEye.x).toDouble(), (nose.y - leftEye.y).toDouble()).toFloat()
        val distRightNose = hypot((rightEye.x - nose.x).toDouble(), (rightEye.y - nose.y).toDouble()).toFloat()
        val yawRatio = ((distRightNose - distLeftNose) / eyeDist).coerceIn(-1f, 1f)

        val cranialLeft = (eyeMidX - eyeDist * 2.15f).coerceAtLeast(0f)
        val cranialRight = (eyeMidX + eyeDist * 2.15f).coerceAtMost(imageWidth.toFloat())
        val cranialTop = (eyeMidY - eyeDist * 2.35f).coerceAtLeast(0f)
        val cranialBottom = (mouthMidY + eyeDist * 1.40f).coerceAtMost(imageHeight.toFloat())

        return HeadPoseAndBounds(
            face = face,
            cranialBox = RectF(cranialLeft, cranialTop, cranialRight, cranialBottom),
            eyeDistance = eyeDist,
            rollDegrees = rollDeg,
            yawRatio = yawRatio
        )
    }

    /**
     * Segments the aligned `cropSize x cropSize` head bitmap into:
     *  - `fullHeadAlpha`: soft alpha matte `[0, 1]` covering face, skull, hair, ears, and upper neck
     *  - `skinAndNeckWeight`: weight `[0, 1]` identifying facial/neck skin vs hair so skin-tone transfer
     *    does not discolor dark/blonde/dyed hair.
     */
    fun segmentHeadHairNeck(
        ortEnv: OrtEnvironment?,
        alignedHeadCrop: Bitmap,
        segModelFile: File?,
        mattingModelFile: File?,
        preferHardwareAccel: Boolean = false
    ): HeadSegmentationMasks {
        val cropSize = alignedHeadCrop.width
        var masks: HeadSegmentationMasks? = null

        if (ortEnv != null && segModelFile != null && segModelFile.exists() && segModelFile.length() > 1024L) {
            masks = runCatching {
                runSegformerOnnx(ortEnv, segModelFile, alignedHeadCrop, preferHardwareAccel)
            }.getOrNull()
        }

        val baseMasks = masks ?: computeChromaticGeodesicHeadMatte(alignedHeadCrop)

        if (ortEnv != null && mattingModelFile != null && mattingModelFile.exists() && mattingModelFile.length() > 1024L) {
            val refinedMasks = runCatching {
                val refined = runModnetOnnx(
                    ortEnv,
                    mattingModelFile,
                    alignedHeadCrop,
                    baseMasks.fullHeadAlpha,
                    preferHardwareAccel
                )
                baseMasks.copy(
                    fullHeadAlpha = refined,
                    segmentationSourceLabel = "${baseMasks.segmentationSourceLabel} + modnet.onnx"
                )
            }.getOrNull()
            if (refinedMasks != null) return refinedMasks
        }

        return baseMasks
    }

    private fun runSegformerOnnx(
        ortEnv: OrtEnvironment,
        segModelFile: File,
        alignedHeadCrop: Bitmap,
        preferHardwareAccel: Boolean
    ): HeadSegmentationMasks {
        val cropSize = alignedHeadCrop.width
        val modelInputSize = 512
        val scaled512 = if (cropSize == modelInputSize) {
            alignedHeadCrop
        } else {
            Bitmap.createScaledBitmap(alignedHeadCrop, modelInputSize, modelInputSize, true)
        }

        val hw = modelInputSize * modelInputSize
        val pixels = IntArray(hw)
        scaled512.getPixels(pixels, 0, modelInputSize, 0, 0, modelInputSize, modelInputSize)
        if (scaled512 !== alignedHeadCrop) scaled512.recycle()

        val floatBuffer = FloatBuffer.allocate(3 * hw)
        // ImageNet normalization: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        for (i in 0 until hw) {
            val c = pixels[i]
            val r = ((c ushr 16) and 0xFF) / 255.0f
            val g = ((c ushr 8) and 0xFF) / 255.0f
            val b = (c and 0xFF) / 255.0f
            floatBuffer.put(i, (r - 0.485f) / 0.229f)
            floatBuffer.put(hw + i, (g - 0.456f) / 0.224f)
            floatBuffer.put(2 * hw + i, (b - 0.406f) / 0.225f)
        }
        floatBuffer.rewind()

        val headMaskCrop = FloatArray(cropSize * cropSize)
        val skinMaskCrop = FloatArray(cropSize * cropSize)

        OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
            ortEnv.createSession(segModelFile.absolutePath, opts).use { session ->
                val inputName = session.inputNames.first()
                val shape = longArrayOf(1L, 3L, modelInputSize.toLong(), modelInputSize.toLong())
                OnnxTensor.createTensor(ortEnv, floatBuffer, shape).use { inputTensor ->
                    session.run(mapOf(inputName to inputTensor)).use { result ->
                        val outTensor = result[0] as OnnxTensor
                        val outShape = outTensor.info.shape // [1, numClasses, outH, outW]
                        val numClasses = outShape[1].toInt()
                        val outH = outShape[2].toInt()
                        val outW = outShape[3].toInt()
                        val outFloats = FloatArray(numClasses * outH * outW)
                        outTensor.floatBuffer.get(outFloats)

                        val planeSize = outH * outW
                        for (y in 0 until cropSize) {
                            val sy = (y * outH / cropSize).coerceIn(0, outH - 1)
                            for (x in 0 until cropSize) {
                                val sx = (x * outW / cropSize).coerceIn(0, outW - 1)
                                val spatialIdx = sy * outW + sx
                                var bestClass = 0
                                var bestLogit = -Float.MAX_VALUE
                                for (cls in 0 until numClasses) {
                                    val logit = outFloats[cls * planeSize + spatialIdx]
                                    if (logit > bestLogit) {
                                        bestLogit = logit
                                        bestClass = cls
                                    }
                                }
                                // CelebAMask-HQ / ATR classes:
                                // 0=bg, 1..13=face/ears/eyes/nose/lips, 14..15=neck, 16=cloth, 17=hair, 18=hat
                                val isSkinOrNeck = bestClass in 1..15
                                val isHairOrHat = bestClass == 17 || bestClass == 18
                                val idx = y * cropSize + x
                                headMaskCrop[idx] = if (isSkinOrNeck || isHairOrHat) 1.0f else 0.0f
                                skinMaskCrop[idx] = if (isSkinOrNeck) 1.0f else 0.0f
                            }
                        }
                    }
                }
            }
        }

        val cropPixels = IntArray(cropSize * cropSize)
        alignedHeadCrop.getPixels(cropPixels, 0, cropSize, 0, 0, cropSize, cropSize)
        val smoothedAlpha = refineAlphaJointBilateral(headMaskCrop, cropPixels, cropSize)

        return HeadSegmentationMasks(
            cropSize = cropSize,
            fullHeadAlpha = smoothedAlpha,
            skinAndNeckWeight = skinMaskCrop,
            usedSegmentationOnnx = true,
            segmentationSourceLabel = "segformer_B5_ce.onnx (ONNX Runtime)"
        )
    }

    private fun runModnetOnnx(
        ortEnv: OrtEnvironment,
        mattingFile: File,
        alignedHeadCrop: Bitmap,
        coarseAlpha: FloatArray,
        preferHardwareAccel: Boolean
    ): FloatArray {
        val cropSize = alignedHeadCrop.width
        val modelSize = 512
        val scaled = if (cropSize == modelSize) {
            alignedHeadCrop
        } else {
            Bitmap.createScaledBitmap(alignedHeadCrop, modelSize, modelSize, true)
        }
        val hw = modelSize * modelSize
        val pixels = IntArray(hw)
        scaled.getPixels(pixels, 0, modelSize, 0, 0, modelSize, modelSize)
        if (scaled !== alignedHeadCrop) scaled.recycle()

        val fb = FloatBuffer.allocate(3 * hw)
        for (i in 0 until hw) {
            val c = pixels[i]
            fb.put(i, (((c ushr 16) and 0xFF) - 127.5f) / 127.5f)
            fb.put(hw + i, (((c ushr 8) and 0xFF) - 127.5f) / 127.5f)
            fb.put(2 * hw + i, ((c and 0xFF) - 127.5f) / 127.5f)
        }
        fb.rewind()

        val refined = FloatArray(cropSize * cropSize)
        OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
            ortEnv.createSession(mattingFile.absolutePath, opts).use { session ->
                val inputName = session.inputNames.first()
                val shape = longArrayOf(1L, 3L, modelSize.toLong(), modelSize.toLong())
                OnnxTensor.createTensor(ortEnv, fb, shape).use { inTensor ->
                    session.run(mapOf(inputName to inTensor)).use { res ->
                        val outTensor = res[0] as OnnxTensor
                        val matte512 = FloatArray(hw)
                        outTensor.floatBuffer.get(matte512)
                        for (y in 0 until cropSize) {
                            val sy = (y * modelSize / cropSize).coerceIn(0, modelSize - 1)
                            for (x in 0 until cropSize) {
                                val sx = (x * modelSize / cropSize).coerceIn(0, modelSize - 1)
                                val modVal = matte512[sy * modelSize + sx].coerceIn(0f, 1f)
                                val idx = y * cropSize + x
                                refined[idx] = (0.55f * coarseAlpha[idx] + 0.45f * modVal).coerceIn(0f, 1f)
                            }
                        }
                    }
                }
            }
        }
        return refined
    }

    /**
     * Built-in Android Head/Hair/Skull/Neck segmentation & matting engine.
     * Separates background from hair crown, skull, ears, face, and upper neck using exterior
     * background chromatic clustering, interior skin/hair sampling, and joint bilateral edge refinement.
     */
    fun computeChromaticGeodesicHeadMatte(alignedHeadCrop: Bitmap): HeadSegmentationMasks {
        val size = alignedHeadCrop.width
        val total = size * size
        val pixels = IntArray(total)
        alignedHeadCrop.getPixels(pixels, 0, size, 0, 0, size, size)

        // 1. Sample exterior background colors from top-left, top-right, and upper side borders
        val bgSamplesR = mutableListOf<Float>()
        val bgSamplesG = mutableListOf<Float>()
        val bgSamplesB = mutableListOf<Float>()
        val step = (size / 16).coerceAtLeast(2)

        for (y in 2 until (size * 0.45f).toInt() step step) {
            for (x in listOf(3, 8, size - 9, size - 4)) {
                val c = pixels[y * size + x]
                bgSamplesR.add(((c ushr 16) and 0xFF).toFloat())
                bgSamplesG.add(((c ushr 8) and 0xFF).toFloat())
                bgSamplesB.add((c and 0xFF).toFloat())
            }
        }
        for (x in 4 until size - 4 step step) {
            val c = pixels[3 * size + x]
            bgSamplesR.add(((c ushr 16) and 0xFF).toFloat())
            bgSamplesG.add(((c ushr 8) and 0xFF).toFloat())
            bgSamplesB.add((c and 0xFF).toFloat())
        }

        // 2. Sample interior facial skin color around canonical nose & cheeks (x in 0.38..0.62, y in 0.45..0.68)
        var skinR = 0f
        var skinG = 0f
        var skinB = 0f
        var skinCount = 0
        for (y in (size * 0.45f).toInt()..(size * 0.68f).toInt() step 2) {
            for (x in (size * 0.38f).toInt()..(size * 0.62f).toInt() step 2) {
                val c = pixels[y * size + x]
                skinR += (c ushr 16) and 0xFF
                skinG += (c ushr 8) and 0xFF
                skinB += c and 0xFF
                skinCount++
            }
        }
        if (skinCount > 0) {
            skinR /= skinCount
            skinG /= skinCount
            skinB /= skinCount
        }

        val rawAlpha = FloatArray(total)
        val skinWeight = FloatArray(total)
        val borderMargin = (size * 0.035f).toInt().coerceAtLeast(5)

        for (y in 0 until size) {
            val ny = y.toFloat() / size.toFloat() // [0, 1]
            for (x in 0 until size) {
                val nx = x.toFloat() / size.toFloat() // [0, 1]
                val idx = y * size + x

                if (x < borderMargin || x >= size - borderMargin ||
                    y < borderMargin || y >= size - borderMargin
                ) {
                    rawAlpha[idx] = 0f
                    skinWeight[idx] = 0f
                    continue
                }

                val c = pixels[idx]
                val r = ((c ushr 16) and 0xFF).toFloat()
                val g = ((c ushr 8) and 0xFF).toFloat()
                val b = (c and 0xFF).toFloat()

                // Minimum chromatic distance to exterior background samples
                var minBgDistSq = Float.MAX_VALUE
                for (k in bgSamplesR.indices) {
                    val dr = r - bgSamplesR[k]
                    val dg = g - bgSamplesG[k]
                    val db = b - bgSamplesB[k]
                    val dSq = dr * dr * 0.30f + dg * dg * 0.59f + db * db * 0.11f +
                        (abs(dr) + abs(dg) + abs(db)) * 4.0f
                    if (dSq < minBgDistSq) minBgDistSq = dSq
                }
                val bgLikelihood = (sqrt(minBgDistSq) / 38.0f).coerceIn(0f, 1f)

                // Distance to facial skin chrominance
                val skinDist = sqrt(
                    (r - skinR) * (r - skinR) * 0.35f +
                        (g - skinG) * (g - skinG) * 0.45f +
                        (b - skinB) * (b - skinB) * 0.20f
                )
                val isSkinLike = (1.0f - (skinDist / 62.0f)).coerceIn(0f, 1f)

                // Anatomical regions in canonical 256x256 head space:
                // A) Core Face Oval: center (0.50, 0.56), rx = 0.21, ry = 0.23
                val fDx = (nx - 0.50f) / 0.215f
                val fDy = (ny - 0.56f) / 0.235f
                val faceR = sqrt(fDx * fDx + fDy * fDy)
                val faceCoreAlpha = when {
                    faceR <= 0.85f -> 1.0f
                    faceR >= 1.22f -> 0.0f
                    else -> {
                        val t = (faceR - 0.85f) / (1.22f - 0.85f)
                        (0.5f * (1f + cos(Math.PI * t))).toFloat()
                    }
                }

                // B) Cranial & Hair Volume Dome: center (0.50, 0.46), rx = 0.39, ry = 0.39
                val hDx = (nx - 0.50f) / 0.395f
                val hDy = (ny - 0.46f) / 0.395f
                val hairR = sqrt(hDx * hDx + hDy * hDy)
                val hairEnvelope = when {
                    hairR <= 0.58f -> 1.0f
                    hairR >= 1.0f -> 0.0f
                    else -> {
                        val t = (hairR - 0.58f) / (1.0f - 0.58f)
                        (0.5f * (1f + cos(Math.PI * t))).toFloat()
                    }
                }
                // Hair foreground combines spatial cranial dome with background color difference
                val hairAlpha = if (ny < 0.80f) {
                    hairEnvelope * (0.28f + 0.72f * bgLikelihood)
                } else {
                    0f
                }

                // C) Upper Neck Column (ny in 0.70..0.92, nx in 0.33..0.67)
                val neckAlpha = if (ny in 0.68f..0.93f) {
                    val nxDist = abs(nx - 0.50f) / 0.165f
                    val nyTaper = ((0.93f - ny) / (0.93f - 0.68f)).coerceIn(0f, 1f)
                    if (nxDist < 1.0f) {
                        val horiz = (0.5f * (1f + cos(Math.PI * nxDist))).toFloat()
                        horiz * nyTaper * (0.55f + 0.45f * isSkinLike)
                    } else {
                        0f
                    }
                } else {
                    0f
                }

                val combinedAlpha = max(faceCoreAlpha, max(hairAlpha, neckAlpha)).coerceIn(0f, 1f)
                rawAlpha[idx] = combinedAlpha
                skinWeight[idx] = max(faceCoreAlpha * 0.9f, neckAlpha).coerceIn(0f, 1f) * isSkinLike
            }
        }

        val refinedAlpha = refineAlphaJointBilateral(rawAlpha, pixels, size)

        return HeadSegmentationMasks(
            cropSize = size,
            fullHeadAlpha = refinedAlpha,
            skinAndNeckWeight = skinWeight,
            usedSegmentationOnnx = false,
            segmentationSourceLabel = "Biometric Cranial-Hair-Neck Geodesic Matte + Guided Filter"
        )
    }

    /**
     * Joint bilateral edge-preserving filter that aligns the soft alpha matte to actual hair and
     * jawline luminance/color boundaries while keeping transitions smooth and organic.
     */
    fun refineAlphaJointBilateral(
        rawAlpha: FloatArray,
        pixels: IntArray,
        size: Int,
        radius: Int = 4
    ): FloatArray {
        val out = FloatArray(size * size)
        val borderMargin = (size * 0.035f).toInt().coerceAtLeast(5)
        val invColorSigmaSq = 1.0f / (2.0f * 28.0f * 28.0f)
        val invSpatialSigmaSq = 1.0f / (2.0f * (radius * 0.75f) * (radius * 0.75f))

        for (y in 0 until size) {
            for (x in 0 until size) {
                if (x < borderMargin || x >= size - borderMargin ||
                    y < borderMargin || y >= size - borderMargin
                ) {
                    out[y * size + x] = 0f
                    continue
                }
                val centerIdx = y * size + x
                val centerA = rawAlpha[centerIdx]
                if (centerA >= 0.995f || centerA <= 0.002f) {
                    // Still apply a gentle 3x3 average for anti-aliasing
                    var s = 0f
                    var cnt = 0
                    for (dy in -1..1) {
                        val yy = (y + dy).coerceIn(0, size - 1)
                        for (dx in -1..1) {
                            val xx = (x + dx).coerceIn(0, size - 1)
                            s += rawAlpha[yy * size + xx]
                            cnt++
                        }
                    }
                    out[centerIdx] = s / cnt
                    continue
                }

                val cc = pixels[centerIdx]
                val cr = (cc ushr 16) and 0xFF
                val cg = (cc ushr 8) and 0xFF
                val cb = cc and 0xFF

                var weightedSum = 0f
                var weightTotal = 0f

                val yMin = (y - radius).coerceAtLeast(0)
                val yMax = (y + radius).coerceAtMost(size - 1)
                val xMin = (x - radius).coerceAtLeast(0)
                val xMax = (x + radius).coerceAtMost(size - 1)

                for (ny in yMin..yMax) {
                    val dy = ny - y
                    for (nx in xMin..xMax) {
                        val dx = nx - x
                        val nIdx = ny * size + nx
                        val nc = pixels[nIdx]
                        val dr = ((nc ushr 16) and 0xFF) - cr
                        val dg = ((nc ushr 8) and 0xFF) - cg
                        val db = (nc and 0xFF) - cb

                        val colorDistSq = (dr * dr + dg * dg + db * db).toFloat()
                        val spatialDistSq = (dx * dx + dy * dy).toFloat()
                        val w = exp(-(spatialDistSq * invSpatialSigmaSq + colorDistSq * invColorSigmaSq))

                        weightedSum += rawAlpha[nIdx] * w
                        weightTotal += w
                    }
                }

                // Soft border envelope near the crop boundary so zero rectangular edges ever appear
                val edgeDist = min(
                    min(x - borderMargin, size - 1 - borderMargin - x),
                    min(y - borderMargin, size - 1 - borderMargin - y)
                ).toFloat()
                val edgeEnvelope = (edgeDist / 16.0f).coerceIn(0f, 1f)

                out[centerIdx] = ((weightedSum / weightTotal.coerceAtLeast(1e-5f)) * edgeEnvelope)
                    .coerceIn(0f, 1f)
            }
        }
        return out
    }

    /**
     * Inpaints disoccluded background gaps (`targetHeadAlpha > sourceHeadAlpha`) in the aligned
     * target head crop so that when a wide-haired or long-eared target head is replaced by a smaller
     * source head, the target's old hair/ears are cleanly replaced with surrounding background.
     */
    fun inpaintTargetBackgroundGap(
        ortEnv: OrtEnvironment?,
        targetAlignedCrop: Bitmap,
        targetHeadAlpha: FloatArray,
        sourceHeadAlpha: FloatArray,
        lamaModelFile: File?,
        preferHardwareAccel: Boolean = false
    ): Bitmap {
        val size = targetAlignedCrop.width
        val total = size * size
        val gapMask = FloatArray(total)
        var gapPixelCount = 0

        for (i in 0 until total) {
            // Regions occupied by the target's old head/hair that are NOT fully covered by the new source head
            val gap = (targetHeadAlpha[i] - sourceHeadAlpha[i] * 0.85f).coerceIn(0f, 1f)
            gapMask[i] = gap
            if (gap > 0.15f) gapPixelCount++
        }

        if (gapPixelCount < 16) {
            return targetAlignedCrop.copy(Bitmap.Config.ARGB_8888, true)
        }

        // Try ONNX LaMa inpainting if lama_fp32.onnx is installed
        if (ortEnv != null && lamaModelFile != null && lamaModelFile.exists() && lamaModelFile.length() > 1024L) {
            val lamaOut = runCatching {
                runLamaInpaintOnnx(ortEnv, lamaModelFile, targetAlignedCrop, gapMask, preferHardwareAccel)
            }.getOrNull()
            if (lamaOut != null) return lamaOut
        }

        // Built-in Multi-Scale Directional Boundary Marching Background Inpainter
        val pixels = IntArray(total)
        targetAlignedCrop.getPixels(pixels, 0, size, 0, 0, size, size)
        val inpainted = pixels.copyOf()

        // For each scanline y (especially in the upper 78% above the shoulders/collar),
        // sample clean exterior background pixels from left and right outside the target head mask
        val maxInpaintRow = (size * 0.82f).toInt()
        for (y in 0 until maxInpaintRow) {
            val rowOffset = y * size

            // Find left exterior background anchor (where targetHeadAlpha < 0.12f)
            var leftAnchorX = 2
            for (x in 2 until size / 2) {
                if (targetHeadAlpha[rowOffset + x] < 0.12f) {
                    leftAnchorX = x
                } else {
                    break
                }
            }
            val leftColor = pixels[rowOffset + leftAnchorX]

            // Find right exterior background anchor
            var rightAnchorX = size - 3
            for (x in (size - 3) downTo size / 2) {
                if (targetHeadAlpha[rowOffset + x] < 0.12f) {
                    rightAnchorX = x
                } else {
                    break
                }
            }
            val rightColor = pixels[rowOffset + rightAnchorX]

            // Also sample top anchor at column x
            val span = (rightAnchorX - leftAnchorX).coerceAtLeast(1).toFloat()
            for (x in leftAnchorX..rightAnchorX) {
                val idx = rowOffset + x
                val gapAlpha = gapMask[idx]
                if (gapAlpha > 0.05f) {
                    val tHoriz = ((x - leftAnchorX) / span).coerceIn(0f, 1f)
                    val topColor = pixels[min(y, 4) * size + x]

                    val hR = ((leftColor ushr 16 and 0xFF) * (1f - tHoriz) +
                        (rightColor ushr 16 and 0xFF) * tHoriz)
                    val hG = ((leftColor ushr 8 and 0xFF) * (1f - tHoriz) +
                        (rightColor ushr 8 and 0xFF) * tHoriz)
                    val hB = ((leftColor and 0xFF) * (1f - tHoriz) +
                        (rightColor and 0xFF) * tHoriz)

                    // Blend 75% horizontal boundary propagation + 25% top sky/wall anchor
                    val bgR = (0.75f * hR + 0.25f * (topColor ushr 16 and 0xFF)).toInt().coerceIn(0, 255)
                    val bgG = (0.75f * hG + 0.25f * (topColor ushr 8 and 0xFF)).toInt().coerceIn(0, 255)
                    val bgB = (0.75f * hB + 0.25f * (topColor and 0xFF)).toInt().coerceIn(0, 255)

                    val orig = pixels[idx]
                    val oR = (orig ushr 16) and 0xFF
                    val oG = (orig ushr 8) and 0xFF
                    val oB = orig and 0xFF

                    val blend = (gapAlpha * 1.15f).coerceIn(0f, 1f)
                    val fR = (oR * (1f - blend) + bgR * blend).toInt().coerceIn(0, 255)
                    val fG = (oG * (1f - blend) + bgG * blend).toInt().coerceIn(0, 255)
                    val fB = (oB * (1f - blend) + bgB * blend).toInt().coerceIn(0, 255)
                    inpainted[idx] = (0xFF shl 24) or (fR shl 16) or (fG shl 8) or fB
                }
            }
        }

        val outBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(inpainted, 0, size, 0, 0, size, size)
        return outBmp
    }

    private fun runLamaInpaintOnnx(
        ortEnv: OrtEnvironment,
        lamaFile: File,
        cropBitmap: Bitmap,
        gapMask: FloatArray,
        preferHardwareAccel: Boolean
    ): Bitmap {
        val origSize = cropBitmap.width
        val modelSize = 512
        val scaled = if (origSize == modelSize) {
            cropBitmap
        } else {
            Bitmap.createScaledBitmap(cropBitmap, modelSize, modelSize, true)
        }
        val hw = modelSize * modelSize
        val pixels = IntArray(hw)
        scaled.getPixels(pixels, 0, modelSize, 0, 0, modelSize, modelSize)
        if (scaled !== cropBitmap) scaled.recycle()

        val imgBuf = FloatBuffer.allocate(3 * hw)
        val maskBuf = FloatBuffer.allocate(hw)
        for (y in 0 until modelSize) {
            val sy = (y * origSize / modelSize).coerceIn(0, origSize - 1)
            for (x in 0 until modelSize) {
                val sx = (x * origSize / modelSize).coerceIn(0, origSize - 1)
                val i = y * modelSize + x
                val m = if (gapMask[sy * origSize + sx] > 0.2f) 1.0f else 0.0f
                val c = pixels[i]
                val r = ((c ushr 16) and 0xFF) / 255.0f
                val g = ((c ushr 8) and 0xFF) / 255.0f
                val b = (c and 0xFF) / 255.0f
                imgBuf.put(i, r * (1f - m))
                imgBuf.put(hw + i, g * (1f - m))
                imgBuf.put(2 * hw + i, b * (1f - m))
                maskBuf.put(i, m)
            }
        }
        imgBuf.rewind()
        maskBuf.rewind()

        OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
            ortEnv.createSession(lamaFile.absolutePath, opts).use { session ->
                val names = session.inputNames.toList()
                val imgName = names.firstOrNull { it.contains("image", ignoreCase = true) } ?: names[0]
                val maskName = names.firstOrNull { it.contains("mask", ignoreCase = true) }
                    ?: names.getOrElse(1) { names[0] }

                OnnxTensor.createTensor(ortEnv, imgBuf, longArrayOf(1L, 3L, 512L, 512L)).use { imgT ->
                    OnnxTensor.createTensor(ortEnv, maskBuf, longArrayOf(1L, 1L, 512L, 512L)).use { maskT ->
                        session.run(mapOf(imgName to imgT, maskName to maskT)).use { res ->
                            val outT = res[0] as OnnxTensor
                            val outFloats = FloatArray(3 * hw)
                            outT.floatBuffer.get(outFloats)
                            val maxVal = outFloats.maxOrNull() ?: 1.0f
                            val scale = if (maxVal <= 1.5f) 255.0f else 1.0f
                            val outPx = IntArray(hw)
                            for (i in 0 until hw) {
                                val r = (outFloats[i] * scale).toInt().coerceIn(0, 255)
                                val g = (outFloats[hw + i] * scale).toInt().coerceIn(0, 255)
                                val b = (outFloats[2 * hw + i] * scale).toInt().coerceIn(0, 255)
                                outPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                            }
                            val bmp512 = Bitmap.createBitmap(modelSize, modelSize, Bitmap.Config.ARGB_8888)
                            bmp512.setPixels(outPx, 0, modelSize, 0, 0, modelSize, modelSize)
                            return if (origSize == modelSize) {
                                bmp512
                            } else {
                                val resized = Bitmap.createScaledBitmap(bmp512, origSize, origSize, true)
                                bmp512.recycle()
                                resized
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Renders a diagnostic visual preview of the detected Head / Hair / Neck segmentation matte
     * and cranial bounding geometry over the source head photo.
     */
    fun createHeadSegmentationPreviewBitmap(
        alignedHeadCrop: Bitmap,
        masks: HeadSegmentationMasks
    ): Bitmap {
        val size = alignedHeadCrop.width
        val total = size * size
        val pixels = IntArray(total)
        alignedHeadCrop.getPixels(pixels, 0, size, 0, 0, size, size)
        val out = IntArray(total)

        for (i in 0 until total) {
            val c = pixels[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            val headA = masks.fullHeadAlpha[i]
            val skinW = masks.skinAndNeckWeight[i]

            if (headA > 0.15f) {
                // Highlight hair/cranial dome with electric cyan tint and face/neck with emerald tint
                val tintR = if (skinW > 0.35f) 0 else 0
                val tintG = if (skinW > 0.35f) 230 else 229
                val tintB = if (skinW > 0.35f) 118 else 255
                val nr = (r * 0.72f + tintR * 0.28f * headA).toInt().coerceIn(0, 255)
                val ng = (g * 0.72f + tintG * 0.28f * headA).toInt().coerceIn(0, 255)
                val nb = (b * 0.72f + tintB * 0.28f * headA).toInt().coerceIn(0, 255)
                out[i] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            } else {
                // Dim exterior background to clearly show the segmented head/hair/neck boundary
                val dimR = (r * 0.28f).toInt()
                val dimG = (g * 0.28f).toInt()
                val dimB = (b * 0.35f).toInt()
                out[i] = (0xFF shl 24) or (dimR shl 16) or (dimG shl 8) or dimB
            }
        }
        val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        bmp.setPixels(out, 0, size, 0, 0, size, size)
        return bmp
    }
}
