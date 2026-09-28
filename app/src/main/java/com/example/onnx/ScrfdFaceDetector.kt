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
        confThreshold: Float = 0.50f,
        nmsThreshold: Float = 0.40f
    ): List<DetectedFace> {
        require(detModelFile.exists() && detModelFile.length() > 1024L) {
            "det_10g.onnx not found at ${detModelFile.absolutePath}"
        }

        // 1. Letterbox resize preserving aspect ratio into 640x640 canvas
        val origW = bitmap.width
        val origH = bitmap.height
        val imRatio = origH.toFloat() / origW.toFloat()
        val modelRatio = INPUT_HEIGHT.toFloat() / INPUT_WIDTH.toFloat()

        val newW: Int
        val newH: Int
        if (imRatio > modelRatio) {
            newH = INPUT_HEIGHT
            newW = (newH / imRatio).toInt().coerceAtLeast(1)
        } else {
            newW = INPUT_WIDTH
            newH = (newW * imRatio).toInt().coerceAtLeast(1)
        }
        val detScale = newH.toFloat() / origH.toFloat()

        val detCanvasBitmap = Bitmap.createBitmap(INPUT_WIDTH, INPUT_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(detCanvasBitmap)
        canvas.drawColor(Color.BLACK)
        val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        canvas.drawBitmap(scaled, 0f, 0f, Paint(Paint.FILTER_BITMAP_FLAG))
        if (scaled !== bitmap) scaled.recycle()

        // 2. Convert 640x640 ARGB_8888 to NCHW float32 RGB normalized: (px - 127.5f) / 128.0f
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

        OrtSession.SessionOptions().use { sessionOpts ->
            sessionOpts.setIntraOpNumThreads(4)
            sessionOpts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            ortEnv.createSession(detModelFile.absolutePath, sessionOpts).use { session ->
                val inputName = session.inputNames.first()
                val shape = longArrayOf(1L, 3L, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong())
                OnnxTensor.createTensor(ortEnv, floatBuffer, shape).use { inputTensor ->
                    session.run(mapOf(inputName to inputTensor)).use { result ->
                        // Group outputs by number of anchors (12800, 3200, 800) and feature width (1, 4, 10)
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
                                        // Handle models that output raw logits vs pre-sigmoid probabilities
                                        if (score < 0f || score > 1f) {
                                            score = (1.0f / (1.0f + exp(-score)))
                                        }
                                        if (score >= confThreshold) {
                                            val bOffset = anchorIdx * 4
                                            val l = bboxes[bOffset] * stride
                                            val t = bboxes[bOffset + 1] * stride
                                            val r = bboxes[bOffset + 2] * stride
                                            val b = bboxes[bOffset + 3] * stride

                                            val x1 = ((cx - l) / detScale).coerceIn(0f, origW.toFloat())
                                            val y1 = ((cy - t) / detScale).coerceIn(0f, origH.toFloat())
                                            val x2 = ((cx + r) / detScale).coerceIn(0f, origW.toFloat())
                                            val y2 = ((cy + b) / detScale).coerceIn(0f, origH.toFloat())

                                            val landmarks = if (kps != null) {
                                                val kOffset = anchorIdx * 10
                                                List(5) { ptIdx ->
                                                    val kx = ((cx + kps[kOffset + ptIdx * 2] * stride) / detScale)
                                                        .coerceIn(0f, origW.toFloat())
                                                    val ky = ((cy + kps[kOffset + ptIdx * 2 + 1] * stride) / detScale)
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
            }
        }

        val nmsFiltered = nonMaximumSuppression(rawCandidates, nmsThreshold)
        // Sort left-to-right by horizontal center for intuitive face indexing in UI
        return nmsFiltered
            .sortedBy { it.boundingBox.centerX() }
            .mapIndexed { idx, face -> face.copy(index = idx) }
    }

    /**
     * Offline Android hardware preview detector used ONLY to preview face bounding boxes in the UI
     * if the user selects a photo before importing `det_10g.onnx`.
     */
    fun detectFacesAndroidPreviewFallback(bitmap: Bitmap, maxFaces: Int = 10): List<DetectedFace> {
        val evenW = if (bitmap.width % 2 == 0) bitmap.width else bitmap.width - 1
        val evenH = if (bitmap.height % 2 == 0) bitmap.height else bitmap.height - 1
        if (evenW <= 16 || evenH <= 16) return emptyList()

        val rgb565 = Bitmap.createBitmap(evenW, evenH, Bitmap.Config.RGB_565)
        val canvas = Canvas(rgb565)
        canvas.drawBitmap(bitmap, 0f, 0f, null)

        val detector = android.media.FaceDetector(evenW, evenH, maxFaces)
        val faces = arrayOfNulls<android.media.FaceDetector.Face>(maxFaces)
        val found = detector.findFaces(rgb565, faces)
        rgb565.recycle()

        val results = mutableListOf<DetectedFace>()
        for (i in 0 until found) {
            val f = faces[i] ?: continue
            val mid = PointF()
            f.getMidPoint(mid)
            val eyeDist = f.eyesDistance()
            if (eyeDist <= 4f) continue

            val halfW = eyeDist * 1.30f
            val top = (mid.y - eyeDist * 1.15f).coerceAtLeast(0f)
            val bottom = (mid.y + eyeDist * 1.65f).coerceAtMost(evenH.toFloat())
            val left = (mid.x - halfW).coerceAtLeast(0f)
            val right = (mid.x + halfW).coerceAtMost(evenW.toFloat())
            val box = RectF(left, top, right, bottom)

            val landmarks = listOf(
                PointF(mid.x - eyeDist * 0.5f, mid.y),                  // Left Eye
                PointF(mid.x + eyeDist * 0.5f, mid.y),                  // Right Eye
                PointF(mid.x, mid.y + eyeDist * 0.58f),                 // Nose Tip
                PointF(mid.x - eyeDist * 0.42f, mid.y + eyeDist * 1.12f), // Left Mouth
                PointF(mid.x + eyeDist * 0.42f, mid.y + eyeDist * 1.12f)  // Right Mouth
            )
            results.add(
                DetectedFace(
                    index = i,
                    boundingBox = box,
                    score = f.confidence(),
                    landmarks5 = landmarks,
                    detectorSource = "Android FaceDetector Preview (Import det_10g.onnx for SCRFD)"
                )
            )
        }
        return results.sortedBy { it.boundingBox.centerX() }.mapIndexed { idx, face -> face.copy(index = idx) }
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
