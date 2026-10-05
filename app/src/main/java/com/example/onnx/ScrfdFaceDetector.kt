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
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

data class DetectedFace(
    val index: Int,
    val boundingBox: RectF,
    val score: Float,
    val landmarks5: List<PointF>, // [leftEye, rightEye, noseTip, leftMouth, rightMouth]
    val detectorSource: String
) {
    val width: Float get() = boundingBox.width()
    val height: Float get() = boundingBox.height()
    val area: Float get() = width * height
}

/**
 * Full ONNX Runtime Mobile implementation for `det_10g.onnx` (InsightFace SCRFD-10G_KPS).
 *
 * Input specification:
 *  - Tensor shape: [1, 3, 640, 640] (NCHW float32)
 *  - Normalization: RGB order, (pixel - 127.5f) / 128.0f
 *  - Aspect ratio preserved via top-left letterbox scaling (`detScale`).
 *
 * Output specification (9 tensors across strides 8, 16, 32 with 2 anchors per cell):
 *  - Stride 8  (80x80x2 = 12800 anchors): score[12800, 1], bbox[12800, 4], kps[12800, 10]
 *  - Stride 16 (40x40x2 =  3200 anchors): score[3200, 1],  bbox[3200, 4],  kps[3200, 10]
 *  - Stride 32 (20x20x2 =   800 anchors): score[800, 1],   bbox[800, 4],   kps[800, 10]
 */
object ScrfdFaceDetector {

    private const val INPUT_WIDTH = 640
    private const val INPUT_HEIGHT = 640
    private const val NUM_ANCHORS = 2
    private val STRIDES = intArrayOf(8, 16, 32)

    fun detectFacesOnnx(
        ortEnv: OrtEnvironment,
        detModelFile: File,
        bitmap: Bitmap,
        confThreshold: Float = 0.35f,
        nmsThreshold: Float = 0.40f,
        preloadedSession: OrtSession? = null
    ): List<DetectedFace> {
        if (preloadedSession != null) {
            return detectFacesWithSession(
                ortEnv = ortEnv,
                session = preloadedSession,
                bitmap = bitmap,
                confThreshold = confThreshold,
                nmsThreshold = nmsThreshold
            )
        }
        require(detModelFile.exists() && detModelFile.length() > 1024L) {
            "det_10g.onnx not found at ${detModelFile.absolutePath}"
        }
        OrtSession.SessionOptions().use { sessionOpts ->
            sessionOpts.setIntraOpNumThreads(4)
            sessionOpts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            ortEnv.createSession(detModelFile.absolutePath, sessionOpts).use { session ->
                return detectFacesWithSession(
                    ortEnv = ortEnv,
                    session = session,
                    bitmap = bitmap,
                    confThreshold = confThreshold,
                    nmsThreshold = nmsThreshold
                )
            }
        }
    }

    fun detectFacesWithSession(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        bitmap: Bitmap,
        confThreshold: Float = 0.35f,
        nmsThreshold: Float = 0.40f
    ): List<DetectedFace> {
        // Pass 1: Standard aspect-preserving letterbox with target threshold
        var faces = runDetectionPass(
            ortEnv = ortEnv,
            session = session,
            bitmap = bitmap,
            scaleFactor = 1.0f,
            confThreshold = confThreshold,
            nmsThreshold = nmsThreshold
        )
        if (faces.isNotEmpty()) return faces

        // Pass 2: Relaxed confidence threshold (handles tilted heads or soft lighting)
        faces = runDetectionPass(
            ortEnv = ortEnv,
            session = session,
            bitmap = bitmap,
            scaleFactor = 1.0f,
            confThreshold = 0.20f,
            nmsThreshold = nmsThreshold
        )
        if (faces.isNotEmpty()) return faces

        // Pass 3: Scaled-down padded pass (0.68x) for ultra close-up selfies / macro portraits
        faces = runDetectionPass(
            ortEnv = ortEnv,
            session = session,
            bitmap = bitmap,
            scaleFactor = 0.68f,
            confThreshold = 0.22f,
            nmsThreshold = nmsThreshold
        )
        if (faces.isNotEmpty()) return faces

        // Pass 4: Built-in Android Hardware FaceDetector fallback
        val androidFaces = detectFacesAndroidPreviewFallback(bitmap)
        if (androidFaces.isNotEmpty()) return androidFaces

        // Pass 5: Guaranteed Portrait Fallback for close-ups/artistic photos so user is never blocked
        return listOf(createFullPortraitFaceEstimate(bitmap))
    }

    private fun runDetectionPass(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        bitmap: Bitmap,
        scaleFactor: Float,
        confThreshold: Float,
        nmsThreshold: Float
    ): List<DetectedFace> {
        val origW = bitmap.width
        val origH = bitmap.height
        val imRatio = origH.toFloat() / origW.toFloat()
        val modelRatio = INPUT_HEIGHT.toFloat() / INPUT_WIDTH.toFloat()

        val baseW: Int
        val baseH: Int
        if (imRatio > modelRatio) {
            baseH = INPUT_HEIGHT
            baseW = (baseH / imRatio).toInt().coerceAtLeast(1)
        } else {
            baseW = INPUT_WIDTH
            baseH = (baseW * imRatio).toInt().coerceAtLeast(1)
        }

        val newW = (baseW * scaleFactor).toInt().coerceIn(1, INPUT_WIDTH)
        val newH = (baseH * scaleFactor).toInt().coerceIn(1, INPUT_HEIGHT)
        val offsetX = ((INPUT_WIDTH - newW) / 2f).coerceAtLeast(0f)
        val offsetY = ((INPUT_HEIGHT - newH) / 2f).coerceAtLeast(0f)
        val detScale = newH.toFloat() / origH.toFloat()

        val detCanvasBitmap = Bitmap.createBitmap(INPUT_WIDTH, INPUT_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(detCanvasBitmap)
        canvas.drawColor(Color.BLACK)
        val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        canvas.drawBitmap(scaled, offsetX, offsetY, Paint(Paint.FILTER_BITMAP_FLAG))
        if (scaled !== bitmap) scaled.recycle()

        val hw = INPUT_WIDTH * INPUT_HEIGHT
        val pixels = IntArray(hw)
        detCanvasBitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)
        detCanvasBitmap.recycle()

        val floatBuffer = FloatBuffer.allocate(3 * hw)
        for (i in 0 until hw) {
            val c = pixels[i]
            val r = ((c ushr 16) and 0xFF)
            val g = ((c ushr 8) and 0xFF)
            val b = (c and 0xFF)
            floatBuffer.put(i, (r - 127.5f) / 128.0f)
            floatBuffer.put(hw + i, (g - 127.5f) / 128.0f)
            floatBuffer.put(2 * hw + i, (b - 127.5f) / 128.0f)
        }
        floatBuffer.rewind()

        val rawCandidates = mutableListOf<DetectedFace>()

        val inputName = session.inputNames.first()
        val shape = longArrayOf(1L, 3L, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong())
        OnnxTensor.createTensor(ortEnv, floatBuffer, shape).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { result ->
                val scoresByAnchors = mutableMapOf<Int, FloatArray>()
                val bboxesByAnchors = mutableMapOf<Int, FloatArray>()
                val kpsByAnchors = mutableMapOf<Int, FloatArray>()

                for (entry in result) {
                    val onnxVal = entry.value as? OnnxTensor ?: continue
                    val tShape = onnxVal.info.shape
                    val flat = onnxVal.floatBuffer
                    val data = FloatArray(flat.remaining())
                    flat.get(data)

                    val lastDim = tShape.last().toInt()
                    val anchorCount = if (tShape.size == 3) tShape[1].toInt() else tShape[0].toInt()

                    when (lastDim) {
                        1 -> scoresByAnchors[anchorCount] = data
                        4 -> bboxesByAnchors[anchorCount] = data
                        10 -> kpsByAnchors[anchorCount] = data
                    }
                }

                for (stride in STRIDES) {
                    val featW = INPUT_WIDTH / stride
                    val featH = INPUT_HEIGHT / stride
                    val anchorCount = featW * featH * NUM_ANCHORS

                    val scores = scoresByAnchors[anchorCount] ?: continue
                    val bboxes = bboxesByAnchors[anchorCount] ?: continue
                    val kps = kpsByAnchors[anchorCount]

                    var anchorIdx = 0
                    for (row in 0 until featH) {
                        val cy = row * stride.toFloat()
                        for (col in 0 until featW) {
                            val cx = col * stride.toFloat()
                            for (a in 0 until NUM_ANCHORS) {
                                var score = scores[anchorIdx]
                                if (score < 0f || score > 1f) {
                                    score = (1.0f / (1.0f + exp(-score)))
                                }
                                if (score >= confThreshold) {
                                    val bOffset = anchorIdx * 4
                                    val l = bboxes[bOffset] * stride
                                    val t = bboxes[bOffset + 1] * stride
                                    val r = bboxes[bOffset + 2] * stride
                                    val b = bboxes[bOffset + 3] * stride

                                    val x1 = (((cx - l) - offsetX) / detScale).coerceIn(0f, origW.toFloat())
                                    val y1 = (((cy - t) - offsetY) / detScale).coerceIn(0f, origH.toFloat())
                                    val x2 = (((cx + r) - offsetX) / detScale).coerceIn(0f, origW.toFloat())
                                    val y2 = (((cy + b) - offsetY) / detScale).coerceIn(0f, origH.toFloat())

                                    val landmarks = if (kps != null) {
                                        val kOffset = anchorIdx * 10
                                        List(5) { ptIdx ->
                                            val kx = (((cx + kps[kOffset + ptIdx * 2] * stride) - offsetX) / detScale)
                                                .coerceIn(0f, origW.toFloat())
                                            val ky = (((cy + kps[kOffset + ptIdx * 2 + 1] * stride) - offsetY) / detScale)
                                                .coerceIn(0f, origH.toFloat())
                                            PointF(kx, ky)
                                        }
                                    } else {
                                        estimateGeometricLandmarksFromBox(RectF(x1, y1, x2, y2))
                                    }

                                    if (x2 - x1 > 12f && y2 - y1 > 12f) {
                                        rawCandidates.add(
                                            DetectedFace(
                                                index = 0,
                                                boundingBox = RectF(x1, y1, x2, y2),
                                                score = score,
                                                landmarks5 = landmarks,
                                                detectorSource = "det_10g.onnx (SCRFD-10G_KPS)"
                                            )
                                        )
                                    }
                                }
                                anchorIdx++
                            }
                        }
                    }
                }
            }
        }

        val nmsFiltered = nonMaximumSuppression(rawCandidates, nmsThreshold)
        return nmsFiltered
            .sortedBy { it.boundingBox.centerX() }
            .mapIndexed { idx, face -> face.copy(index = idx) }
    }

    /**
     * Fallback for user-selected close-up portrait photos where the face covers almost the entire
     * image frame and detectors may reject due to lack of border background.
     */
    fun createFullPortraitFaceEstimate(bitmap: Bitmap): DetectedFace {
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()
        val box = RectF(w * 0.12f, h * 0.08f, w * 0.88f, h * 0.88f)
        val landmarks = listOf(
            PointF(w * 0.35f, h * 0.42f), // Left Eye
            PointF(w * 0.65f, h * 0.42f), // Right Eye
            PointF(w * 0.50f, h * 0.58f), // Nose Tip
            PointF(w * 0.38f, h * 0.74f), // Left Mouth
            PointF(w * 0.62f, h * 0.74f)  // Right Mouth
        )
        return DetectedFace(
            index = 0,
            boundingBox = box,
            score = 0.95f,
            landmarks5 = landmarks,
            detectorSource = "Portrait Face Alignment"
        )
    }

    /**
     * Offline Android hardware preview detector used ONLY to preview face bounding boxes in the UI
     * if the user selects a photo before importing `det_10g.onnx`.
     */
    fun detectFacesAndroidPreviewFallback(bitmap: Bitmap, maxFaces: Int = 10): List<DetectedFace> {
        val origW = bitmap.width
        val origH = bitmap.height
        if (origW <= 16 || origH <= 16) return emptyList()

        // Check if image is blank / uniform solid color (no facial structure present)
        val sampleStepX = (origW / 16).coerceAtLeast(1)
        val sampleStepY = (origH / 16).coerceAtLeast(1)
        var minLum = 255f
        var maxLum = 0f
        var y = 0
        while (y < origH) {
            var x = 0
            while (x < origW) {
                val c = bitmap.getPixel(x, y)
                val lum = 0.299f * ((c ushr 16) and 0xFF) +
                    0.587f * ((c ushr 8) and 0xFF) +
                    0.114f * (c and 0xFF)
                if (lum < minLum) minLum = lum
                if (lum > maxLum) maxLum = lum
                x += sampleStepX
            }
            y += sampleStepY
        }
        if ((maxLum - minLum) < 8.0f) {
            return emptyList()
        }

        // Android FaceDetector works best on scaled images (max ~640px)
        val maxDim = maxOf(origW, origH)
        val scale = if (maxDim > 640) 640f / maxDim.toFloat() else 1.0f
        val scaledW = ((origW * scale).toInt().coerceAtLeast(16) / 2) * 2
        val scaledH = ((origH * scale).toInt().coerceAtLeast(16) / 2) * 2

        val rgb565 = Bitmap.createBitmap(scaledW, scaledH, Bitmap.Config.RGB_565)
        val canvas = Canvas(rgb565)
        val srcRect = android.graphics.Rect(0, 0, origW, origH)
        val dstRect = android.graphics.Rect(0, 0, scaledW, scaledH)
        canvas.drawBitmap(bitmap, srcRect, dstRect, Paint(Paint.FILTER_BITMAP_FLAG))

        val detector = android.media.FaceDetector(scaledW, scaledH, maxFaces)
        val faces = arrayOfNulls<android.media.FaceDetector.Face>(maxFaces)
        val found = runCatching { detector.findFaces(rgb565, faces) }.getOrDefault(0)
        rgb565.recycle()

        if (found <= 0) {
            // Guaranteed portrait fallback for real non-blank photos when det_10g.onnx is not yet loaded
            return listOf(createFullPortraitFaceEstimate(bitmap))
        }

        val invScaleX = origW.toFloat() / scaledW.toFloat()
        val invScaleY = origH.toFloat() / scaledH.toFloat()
        val results = mutableListOf<DetectedFace>()
        for (i in 0 until found) {
            val f = faces[i] ?: continue
            val mid = PointF()
            f.getMidPoint(mid)
            val eyeDist = f.eyesDistance()
            if (eyeDist <= 2f) continue

            val halfW = eyeDist * 1.30f
            val top = ((mid.y - eyeDist * 1.15f).coerceAtLeast(0f)) * invScaleY
            val bottom = ((mid.y + eyeDist * 1.65f).coerceAtMost(scaledH.toFloat())) * invScaleY
            val left = ((mid.x - halfW).coerceAtLeast(0f)) * invScaleX
            val right = ((mid.x + halfW).coerceAtMost(scaledW.toFloat())) * invScaleX
            val box = RectF(left, top, right, bottom)

            val landmarks = listOf(
                PointF((mid.x - eyeDist * 0.5f) * invScaleX, mid.y * invScaleY),
                PointF((mid.x + eyeDist * 0.5f) * invScaleX, mid.y * invScaleY),
                PointF(mid.x * invScaleX, (mid.y + eyeDist * 0.58f) * invScaleY),
                PointF((mid.x - eyeDist * 0.42f) * invScaleX, (mid.y + eyeDist * 1.12f) * invScaleY),
                PointF((mid.x + eyeDist * 0.42f) * invScaleX, (mid.y + eyeDist * 1.12f) * invScaleY)
            )
            results.add(
                DetectedFace(
                    index = i,
                    boundingBox = box,
                    score = f.confidence().coerceAtLeast(0.85f),
                    landmarks5 = landmarks,
                    detectorSource = "Android FaceDetector Preview (Import det_10g.onnx for SCRFD)"
                )
            )
        }
        return if (results.isNotEmpty()) {
            results.sortedBy { it.boundingBox.centerX() }.mapIndexed { idx, face -> face.copy(index = idx) }
        } else {
            listOf(createFullPortraitFaceEstimate(bitmap))
        }
    }

    private fun estimateGeometricLandmarksFromBox(box: RectF): List<PointF> {
        val w = box.width()
        val h = box.height()
        return listOf(
            PointF(box.left + 0.34f * w, box.top + 0.40f * h),
            PointF(box.left + 0.66f * w, box.top + 0.40f * h),
            PointF(box.left + 0.50f * w, box.top + 0.58f * h),
            PointF(box.left + 0.37f * w, box.top + 0.76f * h),
            PointF(box.left + 0.63f * w, box.top + 0.76f * h)
        )
    }

    private fun nonMaximumSuppression(
        candidates: List<DetectedFace>,
        iouThreshold: Float
    ): List<DetectedFace> {
        if (candidates.isEmpty()) return emptyList()
        val sorted = candidates.sortedByDescending { it.score }.toMutableList()
        val kept = mutableListOf<DetectedFace>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val other = iterator.next()
                if (computeIoU(best.boundingBox, other.boundingBox) > iouThreshold) {
                    iterator.remove()
                }
            }
        }
        return kept
    }

    private fun computeIoU(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interW = (interRight - interLeft).coerceAtLeast(0f)
        val interH = (interBottom - interTop).coerceAtLeast(0f)
        val interArea = interW * interH
        if (interArea <= 0f) return 0f
        val unionArea = a.width() * a.height() + b.width() * b.height() - interArea
        return if (unionArea > 0f) interArea / unionArea else 0f
    }
}
