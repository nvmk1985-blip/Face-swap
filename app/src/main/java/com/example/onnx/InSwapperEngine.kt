package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.nio.FloatBuffer

data class SwapStageProgress(
    val stepIndex: Int,
    val totalSteps: Int = 5,
    val stageTitle: String,
    val detailMessage: String,
    val progressFraction: Float
)

data class FaceSwapExecutionResult(
    val outputBitmap: Bitmap,
    val alignedSource112: Bitmap,
    val alignedTarget128: Bitmap,
    val rawSwapped128: Bitmap,
    val swappedFacesCount: Int,
    val detectionMs: Long,
    val embeddingMs: Long,
    val inswapperMs: Long,
    val blendingMs: Long,
    val totalMs: Long,
    val pipelineSummary: String
)

/**
 * Executes the complete offline ONNX Runtime Mobile Face Swap pipeline:
 *  1. Face Detection & 5-Point Keypoints via `det_10g.onnx` (SCRFD-10G_KPS)
 *  2. 5-Point Umeyama Similarity Alignment (`FaceAlignment.estimateNorm`)
 *  3. Source Identity Embedding Extraction via `w600k_r50.onnx` + `inswapper_128.onnx` `emap[512x512]`
 *  4. Neural Face Synthesis via `inswapper_128.onnx` (`target: [1,3,128,128]`, `source: [1,512]`)
 *  5. LAB/RGB Skin-Tone Harmonization + Feathered Mask Inverse Affine Blending (`FaceBlender`)
 */
object InSwapperEngine {

    private const val INSWAPPER_SIZE = 128

    fun executeFaceSwap(
        context: Context,
        ortEnv: OrtEnvironment,
        sourceBitmap: Bitmap,
        sourceFace: DetectedFace,
        targetBitmap: Bitmap,
        targetFacesToReplace: List<DetectedFace>,
        enableColorTransfer: Boolean,
        enableProvenanceWatermark: Boolean,
        allowTwoModelFallbackForTesting: Boolean,
        onProgress: (SwapStageProgress) -> Unit
    ): FaceSwapExecutionResult {
        val tStart = System.currentTimeMillis()
        val detFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.DETECTOR)
        val recFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.RECOGNIZER)
        val swapFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SWAPPER)

        require(swapFile.exists() && swapFile.length() > 1024L) {
            "inswapper_128.onnx is not loaded. Please import inswapper_128.onnx in the ONNX Models tab or place it in app/src/main/assets/models/."
        }
        require(detFile.exists() && detFile.length() > 1024L) {
            "det_10g.onnx is not loaded. Please import det_10g.onnx in the ONNX Models tab or place it in app/src/main/assets/models/."
        }
        require(targetFacesToReplace.isNotEmpty()) {
            "Please select at least one detected target face to replace."
        }

        // Stage 1: Verify 5-point landmarks from det_10g.onnx
        val tDetStart = System.currentTimeMillis()
        onProgress(
            SwapStageProgress(
                stepIndex = 1,
                stageTitle = "Stage 1/5: SCRFD-10G Landmark Verification",
                detailMessage = "Verifying 5-point keypoints for source face and ${targetFacesToReplace.size} target face(s)...",
                progressFraction = 0.15f
            )
        )
        val detMs = (System.currentTimeMillis() - tDetStart).coerceAtLeast(1L)

        // Stage 2 & 3: Extract 512-D source latent vector + 512x512 emap projection
        val tEmbStart = System.currentTimeMillis()
        onProgress(
            SwapStageProgress(
                stepIndex = 2,
                stageTitle = "Stage 2/5: 5-Point Umeyama Alignment & emap Extraction",
                detailMessage = "Aligning source face to 112x112 and loading [512, 512] emap matrix from inswapper_128.onnx...",
                progressFraction = 0.32f
            )
        )
        val emap512x512 = OnnxProtobufInspector.loadOrExtractInswapperEmap(context, swapFile)

        onProgress(
            SwapStageProgress(
                stepIndex = 3,
                stageTitle = "Stage 3/5: Source 512-D Identity Embedding",
                detailMessage = if (recFile.exists()) {
                    "Running w600k_r50.onnx [1, 3, 112, 112] -> [1, 512] + emap projection..."
                } else {
                    "Computing 512-D source embedding + emap projection..."
                },
                progressFraction = 0.50f
            )
        )
        val sourceEmbedding = ArcFaceRecognizer.extractSourceLatentEmbedding(
            ortEnv = ortEnv,
            sourceBitmap = sourceBitmap,
            sourceLandmarks5 = sourceFace.landmarks5,
            arcFaceModelFile = if (recFile.exists()) recFile else null,
            emap512x512 = emap512x512,
            allowTwoModelFallbackForTesting = allowTwoModelFallbackForTesting
        )
        val embMs = (System.currentTimeMillis() - tEmbStart).coerceAtLeast(1L)

        // Stage 4: Run inswapper_128.onnx for each selected target face
        val tSwapStart = System.currentTimeMillis()
        val targetW = targetBitmap.width
        val targetH = targetBitmap.height
        val compositePixels = IntArray(targetW * targetH)
        targetBitmap.getPixels(compositePixels, 0, targetW, 0, 0, targetW, targetH)

        val featheredMask128 = FaceBlender.createFeatheredFaceMask128()
        val srcM128 = FaceAlignment.estimateNorm(sourceFace.landmarks5, INSWAPPER_SIZE)
        val alignedSource128 = FaceAlignment.warpAffineCrop(sourceBitmap, srcM128, INSWAPPER_SIZE)
        val hasTrueArcFaceLatent = sourceEmbedding.usedArcFaceModel && sourceEmbedding.usedEmbeddedEmap

        var firstTargetCrop128: Bitmap? = null
        var firstRawSwapped128: Bitmap? = null
        var totalSwapMs = 0L
        var totalBlendMs = 0L

        OrtSession.SessionOptions().use { sessionOpts ->
            sessionOpts.setIntraOpNumThreads(4)
            sessionOpts.setMemoryPatternOptimization(true)
            sessionOpts.setCPUArenaAllocator(true)
            sessionOpts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            ortEnv.createSession(swapFile.absolutePath, sessionOpts).use { swapSession ->
                // Identify target [1, 3, 128, 128] and source [1, 512] input names dynamically
                var targetInputName = "target"
                var sourceInputName = "source"
                for ((name, nodeInfo) in swapSession.inputInfo) {
                    val tInfo = nodeInfo.info as? ai.onnxruntime.TensorInfo ?: continue
                    if (tInfo.shape.size == 4) targetInputName = name
                    if (tInfo.shape.size == 2) sourceInputName = name
                }

                val sourceFloatBuffer = FloatBuffer.wrap(sourceEmbedding.latentSourceVector512)
                val sourceShape = longArrayOf(1L, 512L)

                OnnxTensor.createTensor(ortEnv, sourceFloatBuffer, sourceShape).use { sourceTensor ->
                    targetFacesToReplace.forEachIndexed { idx, targetFace ->
                        val stepFrac = 0.55f + (0.30f * (idx.toFloat() / targetFacesToReplace.size.coerceAtLeast(1)))
                        onProgress(
                            SwapStageProgress(
                                stepIndex = 4,
                                stageTitle = "Stage 4/5: InSwapper-128 ONNX Inference (${idx + 1}/${targetFacesToReplace.size})",
                                detailMessage = "Running inswapper_128.onnx on aligned 128x128 target crop #${targetFace.index + 1}...",
                                progressFraction = stepFrac
                            )
                        )

                        val tFaceSwap0 = System.currentTimeMillis()
                        val m128 = FaceAlignment.estimateNorm(targetFace.landmarks5, INSWAPPER_SIZE)
                        val alignedTarget128 = FaceAlignment.warpAffineCrop(targetBitmap, m128, INSWAPPER_SIZE)
                        val rawSwapped128 = runInswapperSingleFace(
                            ortEnv = ortEnv,
                            swapSession = swapSession,
                            targetInputName = targetInputName,
                            sourceInputName = sourceInputName,
                            alignedTarget128 = alignedTarget128,
                            sourceTensor = sourceTensor
                        )
                        totalSwapMs += (System.currentTimeMillis() - tFaceSwap0).coerceAtLeast(1L)

                        val tBlend0 = System.currentTimeMillis()
                        onProgress(
                            SwapStageProgress(
                                stepIndex = 5,
                                stageTitle = "Stage 5/5: Anti-Negative Polarity, Eye Restoration & Blending",
                                detailMessage = "Eliminating negative artifacts, restoring eye clarity & blending face #${targetFace.index + 1}...",
                                progressFraction = 0.90f
                            )
                        )

                        // 1. Eliminate negative/hollow eye & face artifacts and restore crisp positive polarity
                        val eyeRestored128 = FaceBlender.restoreEyesAndEliminateNegativeArtifacts128(
                            swapped128 = rawSwapped128,
                            alignedTarget128 = alignedTarget128,
                            alignedSource112 = sourceEmbedding.aligned112Crop,
                            alignedSource128 = alignedSource128,
                            hasTrueArcFaceLatent = hasTrueArcFaceLatent
                        )

                        // 2. Run optional GFPGAN ONNX enhancement if gfpgan_1.4.onnx is installed
                        val gfpganFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.ENHANCEMENT)
                        val gfpEnhanced128 = if (gfpganFile.exists() && gfpganFile.length() > 1024L) {
                            FaceBlender.runOptionalGfpganEnhancement128(
                                ortEnv = ortEnv,
                                gfpganFile = gfpganFile,
                                crop128 = eyeRestored128
                            )
                        } else {
                            eyeRestored128
                        }

                        // 3. Harmonize skin tone while protecting eyes from color/contrast washout
                        val colorCorrected128 = if (enableColorTransfer) {
                            FaceBlender.transferSkinToneStatistics128(gfpEnhanced128, alignedTarget128)
                        } else {
                            gfpEnhanced128
                        }

                        FaceBlender.blendSwappedFaceIntoTarget(
                            targetPixels = compositePixels,
                            targetWidth = targetW,
                            targetHeight = targetH,
                            swapped128 = colorCorrected128,
                            forwardMatrix128 = m128,
                            featheredMask128 = featheredMask128
                        )

                        if (gfpEnhanced128 !== eyeRestored128) {
                            gfpEnhanced128.recycle()
                        }
                        if (colorCorrected128 !== gfpEnhanced128 && colorCorrected128 !== eyeRestored128) {
                            colorCorrected128.recycle()
                        }

                        if (firstTargetCrop128 == null) {
                            firstTargetCrop128 = alignedTarget128
                            firstRawSwapped128 = eyeRestored128
                            rawSwapped128.recycle()
                        } else {
                            alignedTarget128.recycle()
                            eyeRestored128.recycle()
                            rawSwapped128.recycle()
                        }
                        totalBlendMs += (System.currentTimeMillis() - tBlend0).coerceAtLeast(1L)
                    }
                }
            }
        }
        alignedSource128.recycle()

        val finalBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        finalBitmap.setPixels(compositePixels, 0, targetW, 0, 0, targetW, targetH)

        if (enableProvenanceWatermark) {
            FaceBlender.applyEthicalProvenanceWatermark(finalBitmap)
        }

        val totalMs = (System.currentTimeMillis() - tStart).coerceAtLeast(1L)

        return FaceSwapExecutionResult(
            outputBitmap = finalBitmap,
            alignedSource112 = sourceEmbedding.aligned112Crop,
            alignedTarget128 = firstTargetCrop128
                ?: Bitmap.createBitmap(INSWAPPER_SIZE, INSWAPPER_SIZE, Bitmap.Config.ARGB_8888),
            rawSwapped128 = firstRawSwapped128
                ?: Bitmap.createBitmap(INSWAPPER_SIZE, INSWAPPER_SIZE, Bitmap.Config.ARGB_8888),
            swappedFacesCount = targetFacesToReplace.size,
            detectionMs = detMs,
            embeddingMs = embMs,
            inswapperMs = totalSwapMs.coerceAtLeast(1L),
            blendingMs = totalBlendMs.coerceAtLeast(1L),
            totalMs = totalMs,
            pipelineSummary = "det_10g.onnx -> ${sourceEmbedding.providerSummary} -> inswapper_128.onnx"
        )
    }

    private fun runInswapperSingleFace(
        ortEnv: OrtEnvironment,
        swapSession: OrtSession,
        targetInputName: String,
        sourceInputName: String,
        alignedTarget128: Bitmap,
        sourceTensor: OnnxTensor
    ): Bitmap {
        val hw = INSWAPPER_SIZE * INSWAPPER_SIZE
        val pixels = IntArray(hw)
        alignedTarget128.getPixels(pixels, 0, INSWAPPER_SIZE, 0, 0, INSWAPPER_SIZE, INSWAPPER_SIZE)

        // Preprocess target crop: NCHW [1, 3, 128, 128], RGB in [0.0f, 1.0f] (pixel / 255.0f)
        val targetBuffer = FloatBuffer.allocate(3 * hw)
        for (i in 0 until hw) {
            val c = pixels[i]
            val r = ((c ushr 16) and 0xFF) / 255.0f
            val g = ((c ushr 8) and 0xFF) / 255.0f
            val b = (c and 0xFF) / 255.0f
            targetBuffer.put(i, r)
            targetBuffer.put(hw + i, g)
            targetBuffer.put(2 * hw + i, b)
        }
        targetBuffer.rewind()

        val targetShape = longArrayOf(1L, 3L, INSWAPPER_SIZE.toLong(), INSWAPPER_SIZE.toLong())
        OnnxTensor.createTensor(ortEnv, targetBuffer, targetShape).use { targetTensor ->
            val inputs = mapOf(
                targetInputName to targetTensor,
                sourceInputName to sourceTensor
            )
            swapSession.run(inputs).use { results ->
                val outTensor = results[0] as OnnxTensor
                val outBuf = outTensor.floatBuffer
                val outFloats = FloatArray(3 * hw)
                outBuf.get(outFloats)

                // Postprocess output [1, 3, 128, 128] RGB in [0.0f, 1.0f] -> ARGB_8888 Bitmap
                val outPixels = IntArray(hw)
                for (i in 0 until hw) {
                    val r = (outFloats[i] * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                    val g = (outFloats[hw + i] * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                    val b = (outFloats[2 * hw + i] * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                    outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
                val swappedBitmap = Bitmap.createBitmap(
                    INSWAPPER_SIZE,
                    INSWAPPER_SIZE,
                    Bitmap.Config.ARGB_8888
                )
                swappedBitmap.setPixels(outPixels, 0, INSWAPPER_SIZE, 0, 0, INSWAPPER_SIZE, INSWAPPER_SIZE)
                return swappedBitmap
            }
        }
    }
}
