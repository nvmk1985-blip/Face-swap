package com.example.ui

import ai.onnxruntime.OrtEnvironment
import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.data.SwapAuditDatabase
import com.example.data.SwapAuditLog
import com.example.data.SwapAuditRepository
import com.example.onnx.DetectedFace
import com.example.onnx.FaceAlignment
import com.example.onnx.FaceSwapExecutionResult
import com.example.onnx.GhostHeadReplacementEngine
import com.example.onnx.HeadSegmentationAndInpainting
import com.example.onnx.InSwapperEngine
import com.example.onnx.ModelSlot
import com.example.onnx.OnnxModelInspection
import com.example.onnx.OnnxProtobufInspector
import com.example.onnx.ScrfdFaceDetector
import com.example.onnx.SwapStageProgress
import com.example.util.ImageGalleryHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class AppTab {
    STUDIO,
    MODELS,
    HISTORY,
    ETHICS
}

enum class StudioMode(val title: String, val badge: String) {
    FACE_SWAP("Face Swap", "MODE 1 — InsightFace / InSwapper-128"),
    HEAD_REPLACEMENT("Head Replacement", "MODE 2 — GHOST 2.0 Android Full Head/Hair/Neck")
}

data class HeadAlignmentPreviewState(
    val segmentedSourceHeadBitmap: Bitmap,
    val alignedTargetHeadBitmap: Bitmap,
    val sourceRollDeg: Float,
    val targetRollDeg: Float,
    val segmentationLabel: String
)

data class FaceSwapUiState(
    val currentTab: AppTab = AppTab.STUDIO,
    val studioMode: StudioMode = StudioMode.FACE_SWAP,
    val modelInspections: List<OnnxModelInspection> = emptyList(),
    val isInspectingModels: Boolean = false,
    val sourceBitmap: Bitmap? = null,
    val sourceFaces: List<DetectedFace> = emptyList(),
    val selectedSourceFaceIndex: Int = 0,
    val isDetectingSource: Boolean = false,
    val targetBitmap: Bitmap? = null,
    val targetFaces: List<DetectedFace> = emptyList(),
    val selectedTargetFaceIndex: Int = 0,
    val replaceAllTargetFaces: Boolean = false,
    val isDetectingTarget: Boolean = false,
    val headPreviewState: HeadAlignmentPreviewState? = null,
    val isGeneratingHeadPreview: Boolean = false,
    val consentConfirmed: Boolean = false,
    val enableColorTransfer: Boolean = true,
    val enableProvenanceWatermark: Boolean = true,
    val allowTwoModelFallbackForTesting: Boolean = false,
    val preferHardwareAcceleration: Boolean = true,
    val lowMemoryMode: Boolean = false,
    val isSwapping: Boolean = false,
    val swapProgress: SwapStageProgress? = null,
    val swapResult: FaceSwapExecutionResult? = null,
    val lastAuditLogId: Int? = null,
    val showOriginalInComparison: Boolean = false,
    val statusBannerMessage: String? = null,
    val errorBannerMessage: String? = null
) {
    val isDetectorReady: Boolean
        get() = modelInspections.any { it.slot == ModelSlot.DETECTOR && it.isValidOnnx }

    val isRecognizerReady: Boolean
        get() = modelInspections.any { it.slot == ModelSlot.RECOGNIZER && it.isValidOnnx }

    val isSwapperReady: Boolean
        get() = modelInspections.any { it.slot == ModelSlot.SWAPPER && it.isValidOnnx }

    val isSegmentationOnnxReady: Boolean
        get() = modelInspections.any { it.slot == ModelSlot.SEGMENTATION && it.isValidOnnx }

    val canExecuteSwap: Boolean
        get() = !isSwapping &&
            consentConfirmed &&
            sourceBitmap != null &&
            sourceFaces.isNotEmpty() &&
            targetBitmap != null &&
            targetFaces.isNotEmpty() &&
            when (studioMode) {
                StudioMode.FACE_SWAP ->
                    isDetectorReady && isSwapperReady && (isRecognizerReady || allowTwoModelFallbackForTesting)
                StudioMode.HEAD_REPLACEMENT ->
                    true // Works with installed ONNX models + built-in Cranial-Hair-Neck CV pipeline
            }
}

class FaceSwapViewModel(application: Application) : AndroidViewModel(application) {

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val auditRepository: SwapAuditRepository = SwapAuditRepository(
        SwapAuditDatabase.getInstance(application).swapAuditDao()
    )

    private val _uiState = MutableStateFlow(
        FaceSwapUiState(lowMemoryMode = detectIsLowRamDevice(application))
    )
    val uiState: StateFlow<FaceSwapUiState> = _uiState.asStateFlow()

    val auditLogs: StateFlow<List<SwapAuditLog>> = auditRepository.allLogs.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = emptyList()
    )

    init {
        refreshModelInspections()
    }

    private fun detectIsLowRamDevice(context: Context): Boolean {
        return runCatching {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            am?.isLowRamDevice == true
        }.getOrDefault(false)
    }

    fun selectTab(tab: AppTab) {
        _uiState.update { it.copy(currentTab = tab) }
    }

    fun selectStudioMode(mode: StudioMode) {
        _uiState.update {
            it.copy(
                studioMode = mode,
                swapResult = null,
                headPreviewState = null,
                errorBannerMessage = null,
                statusBannerMessage = null
            )
        }
    }

    fun clearMessages() {
        _uiState.update { it.copy(statusBannerMessage = null, errorBannerMessage = null) }
    }

    fun setConsentConfirmed(confirmed: Boolean) {
        _uiState.update { it.copy(consentConfirmed = confirmed, errorBannerMessage = null) }
    }

    fun setEnableColorTransfer(enabled: Boolean) {
        _uiState.update { it.copy(enableColorTransfer = enabled) }
    }

    fun setEnableProvenanceWatermark(enabled: Boolean) {
        _uiState.update { it.copy(enableProvenanceWatermark = enabled) }
    }

    fun setAllowTwoModelFallback(allowed: Boolean) {
        _uiState.update { it.copy(allowTwoModelFallbackForTesting = allowed) }
    }

    fun setPreferHardwareAcceleration(enabled: Boolean) {
        _uiState.update { it.copy(preferHardwareAcceleration = enabled) }
    }

    fun setLowMemoryMode(enabled: Boolean) {
        _uiState.update { it.copy(lowMemoryMode = enabled) }
    }

    fun setReplaceAllTargetFaces(replaceAll: Boolean) {
        _uiState.update { it.copy(replaceAllTargetFaces = replaceAll) }
    }

    fun toggleComparisonMode(showOriginal: Boolean) {
        _uiState.update { it.copy(showOriginalInComparison = showOriginal) }
    }

    fun selectSourceFace(index: Int) {
        _uiState.update { state ->
            if (index in state.sourceFaces.indices) {
                state.copy(selectedSourceFaceIndex = index, headPreviewState = null)
            } else {
                state
            }
        }
    }

    fun selectTargetFace(index: Int) {
        _uiState.update { state ->
            if (index in state.targetFaces.indices) {
                state.copy(
                    selectedTargetFaceIndex = index,
                    replaceAllTargetFaces = false,
                    headPreviewState = null
                )
            } else {
                state
            }
        }
    }

    fun refreshModelInspections() {
        viewModelScope.launch {
            _uiState.update { it.copy(isInspectingModels = true) }
            val inspections = withContext(Dispatchers.IO) {
                OnnxProtobufInspector.inspectAllModels(getApplication(), ortEnv)
            }
            _uiState.update {
                it.copy(
                    modelInspections = inspections,
                    isInspectingModels = false
                )
            }
            val detReady = inspections.any { it.slot == ModelSlot.DETECTOR && it.isValidOnnx }
            if (detReady) {
                _uiState.value.sourceBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = true) }
                _uiState.value.targetBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = false) }
            }
        }
    }

    fun importOnnxModel(uri: Uri, slot: ModelSlot) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isInspectingModels = true,
                    statusBannerMessage = "Importing ${slot.canonicalFileName} into ${slot.categoryTitle}...",
                    errorBannerMessage = null
                )
            }
            val result = withContext(Dispatchers.IO) {
                OnnxProtobufInspector.importModelFromUri(getApplication(), uri, slot)
            }
            result.fold(
                onSuccess = { file ->
                    val inspections = withContext(Dispatchers.IO) {
                        OnnxProtobufInspector.inspectAllModels(getApplication(), ortEnv)
                    }
                    val importedInspection = inspections.firstOrNull { it.slot == slot }
                    val valid = importedInspection?.isValidOnnx == true
                    _uiState.update {
                        it.copy(
                            modelInspections = inspections,
                            isInspectingModels = false,
                            statusBannerMessage = if (valid) {
                                "Installed ${slot.canonicalFileName} (${formatBytes(file.length())}) in ${slot.categoryTitle}"
                            } else {
                                null
                            },
                            errorBannerMessage = if (!valid) {
                                importedInspection?.verificationNote
                                    ?: "Selected file is not a valid ONNX model."
                            } else {
                                null
                            }
                        )
                    }
                    if (slot == ModelSlot.DETECTOR && valid) {
                        _uiState.value.sourceBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = true) }
                        _uiState.value.targetBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = false) }
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isInspectingModels = false,
                            statusBannerMessage = null,
                            errorBannerMessage = "Model import failed: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun onSourceImageSelected(uri: Uri) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isDetectingSource = true,
                    errorBannerMessage = null,
                    statusBannerMessage = null,
                    headPreviewState = null
                )
            }
            val decoded = withContext(Dispatchers.IO) {
                ImageGalleryHelper.decodeUriToBitmap(getApplication(), uri)
            }
            decoded.fold(
                onSuccess = { bitmap ->
                    _uiState.update {
                        it.copy(
                            sourceBitmap = bitmap,
                            selectedSourceFaceIndex = 0,
                            swapResult = null
                        )
                    }
                    detectFacesForBitmap(bitmap, isSource = true)
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isDetectingSource = false,
                            errorBannerMessage = "Could not load source photo: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun onTargetImageSelected(uri: Uri) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isDetectingTarget = true,
                    errorBannerMessage = null,
                    statusBannerMessage = null,
                    headPreviewState = null
                )
            }
            val decoded = withContext(Dispatchers.IO) {
                ImageGalleryHelper.decodeUriToBitmap(getApplication(), uri)
            }
            decoded.fold(
                onSuccess = { bitmap ->
                    _uiState.update {
                        it.copy(
                            targetBitmap = bitmap,
                            selectedTargetFaceIndex = 0,
                            swapResult = null
                        )
                    }
                    detectFacesForBitmap(bitmap, isSource = false)
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isDetectingTarget = false,
                            errorBannerMessage = "Could not load target photo: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    /**
     * Explicit `[ Detect Head ]` action for Mode 2 (Head Replacement): re-runs head/face/landmark
     * and cranial volume detection on both loaded Source and Target photos.
     */
    fun detectHeadsOnLoadedPhotos() {
        val src = _uiState.value.sourceBitmap
        val tgt = _uiState.value.targetBitmap
        if (src == null && tgt == null) {
            _uiState.update {
                it.copy(errorBannerMessage = "Please select a Source Head photo and/or Target Photo first.")
            }
            return
        }
        src?.let { detectFacesForBitmap(it, isSource = true) }
        tgt?.let { detectFacesForBitmap(it, isSource = false) }
        _uiState.update {
            it.copy(
                statusBannerMessage = "Detected head pose, 5-point landmarks, and cranial/hair/neck bounds."
            )
        }
    }

    /**
     * Explicit `[ Preview ]` action for Mode 2 (Head Replacement): runs Stage 2 (Cranial Alignment)
     * and Stage 4 (Head/Hair/Neck Segmentation) to show the user a live preview of the extracted
     * source head + hair + neck mask alongside the aligned target head region before full replacement.
     */
    fun generateHeadReplacementPreview() {
        val state = _uiState.value
        val srcBmp = state.sourceBitmap
        val tgtBmp = state.targetBitmap
        if (srcBmp == null || state.sourceFaces.isEmpty()) {
            _uiState.update {
                it.copy(errorBannerMessage = "Select a Source Head photo with a detectable face/head first.")
            }
            return
        }
        if (tgtBmp == null || state.targetFaces.isEmpty()) {
            _uiState.update {
                it.copy(errorBannerMessage = "Select a Target photo with a detectable face/head first.")
            }
            return
        }

        val srcFace = state.sourceFaces.getOrElse(state.selectedSourceFaceIndex) { state.sourceFaces.first() }
        val tgtFace = state.targetFaces.getOrElse(state.selectedTargetFaceIndex) { state.targetFaces.first() }

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isGeneratingHeadPreview = true,
                    errorBannerMessage = null,
                    statusBannerMessage = "Generating Head/Hair/Neck Segmentation & Alignment Preview..."
                )
            }
            runCatching {
                withContext(Dispatchers.Default) {
                    val previewSize = 256
                    val srcPose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(
                        srcFace,
                        srcBmp.width,
                        srcBmp.height
                    )
                    val tgtPose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(
                        tgtFace,
                        tgtBmp.width,
                        tgtBmp.height
                    )
                    val srcMat = HeadSegmentationAndInpainting.estimateHeadAlignmentMatrix(
                        srcFace.landmarks5,
                        previewSize,
                        srcPose.yawRatio
                    )
                    val tgtMat = HeadSegmentationAndInpainting.estimateHeadAlignmentMatrix(
                        tgtFace.landmarks5,
                        previewSize,
                        srcPose.yawRatio
                    )
                    val alignedSrc = FaceAlignment.warpAffineCrop(srcBmp, srcMat, previewSize)
                    val alignedTgt = FaceAlignment.warpAffineCrop(tgtBmp, tgtMat, previewSize)

                    val segFile = OnnxProtobufInspector.resolveModelFile(getApplication(), ModelSlot.SEGMENTATION)
                    val matFile = OnnxProtobufInspector.resolveModelFile(getApplication(), ModelSlot.MATTING)
                    val masks = HeadSegmentationAndInpainting.segmentHeadHairNeck(
                        ortEnv = ortEnv,
                        alignedHeadCrop = alignedSrc,
                        segModelFile = if (segFile.exists()) segFile else null,
                        mattingModelFile = if (matFile.exists()) matFile else null,
                        preferHardwareAccel = state.preferHardwareAcceleration
                    )
                    val segPreviewBmp = HeadSegmentationAndInpainting.createHeadSegmentationPreviewBitmap(
                        alignedSrc,
                        masks
                    )
                    alignedSrc.recycle()

                    HeadAlignmentPreviewState(
                        segmentedSourceHeadBitmap = segPreviewBmp,
                        alignedTargetHeadBitmap = alignedTgt,
                        sourceRollDeg = srcPose.rollDegrees,
                        targetRollDeg = tgtPose.rollDegrees,
                        segmentationLabel = masks.segmentationSourceLabel
                    )
                }
            }.fold(
                onSuccess = { preview ->
                    _uiState.update {
                        it.copy(
                            isGeneratingHeadPreview = false,
                            headPreviewState = preview,
                            statusBannerMessage = "Head/Hair/Neck preview ready (${preview.segmentationLabel})."
                        )
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isGeneratingHeadPreview = false,
                            errorBannerMessage = "Preview generation failed: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    private fun detectFacesForBitmap(bitmap: Bitmap, isSource: Boolean) {
        viewModelScope.launch {
            if (isSource) {
                _uiState.update { it.copy(isDetectingSource = true) }
            } else {
                _uiState.update { it.copy(isDetectingTarget = true) }
            }

            val detResult = withContext(Dispatchers.Default) {
                runCatching {
                    val detFile = OnnxProtobufInspector.resolveModelFile(
                        getApplication(),
                        ModelSlot.DETECTOR
                    )
                    if (detFile.exists() && detFile.length() > 1024L) {
                        ScrfdFaceDetector.detectFacesOnnx(ortEnv, detFile, bitmap)
                    } else {
                        ScrfdFaceDetector.detectFacesAndroidPreviewFallback(bitmap)
                    }
                }
            }

            detResult.fold(
                onSuccess = { faces ->
                    val roleLabel = if (isSource) "source" else "target"
                    val warning = if (faces.isEmpty()) {
                        "No clear face/head detected in the $roleLabel photo. Please select a well-lit portrait."
                    } else {
                        null
                    }
                    _uiState.update { state ->
                        if (isSource) {
                            state.copy(
                                sourceFaces = faces,
                                selectedSourceFaceIndex = 0,
                                isDetectingSource = false,
                                errorBannerMessage = warning ?: state.errorBannerMessage
                            )
                        } else {
                            state.copy(
                                targetFaces = faces,
                                selectedTargetFaceIndex = 0,
                                isDetectingTarget = false,
                                errorBannerMessage = warning ?: state.errorBannerMessage
                            )
                        }
                    }
                },
                onFailure = { err ->
                    _uiState.update { state ->
                        if (isSource) {
                            state.copy(
                                sourceFaces = emptyList(),
                                isDetectingSource = false,
                                errorBannerMessage = "Detection failed on source photo: ${err.message}"
                            )
                        } else {
                            state.copy(
                                targetFaces = emptyList(),
                                isDetectingTarget = false,
                                errorBannerMessage = "Detection failed on target photo: ${err.message}"
                            )
                        }
                    }
                }
            )
        }
    }

    fun runActiveModePipeline() {
        val state = _uiState.value
        val srcBitmap = state.sourceBitmap
        val tgtBitmap = state.targetBitmap

        if (!state.consentConfirmed) {
            _uiState.update {
                it.copy(errorBannerMessage = "Please confirm ethical consent before running inference.")
            }
            return
        }
        if (srcBitmap == null || state.sourceFaces.isEmpty()) {
            _uiState.update {
                it.copy(errorBannerMessage = "Select a source photo with at least one detected face/head.")
            }
            return
        }
        if (tgtBitmap == null || state.targetFaces.isEmpty()) {
            _uiState.update {
                it.copy(errorBannerMessage = "Select a target photo with at least one detected face/head.")
            }
            return
        }

        val selectedSourceFace = state.sourceFaces.getOrElse(state.selectedSourceFaceIndex) {
            state.sourceFaces.first()
        }
        val targetsToSwap = if (state.replaceAllTargetFaces) {
            state.targetFaces
        } else {
            listOf(
                state.targetFaces.getOrElse(state.selectedTargetFaceIndex) {
                    state.targetFaces.first()
                }
            )
        }

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isSwapping = true,
                    errorBannerMessage = null,
                    statusBannerMessage = null,
                    showOriginalInComparison = false,
                    swapProgress = SwapStageProgress(
                        stepIndex = 1,
                        totalSteps = if (state.studioMode == StudioMode.HEAD_REPLACEMENT) 6 else 5,
                        stageTitle = "Stage 1: Detection",
                        detailMessage = "Preparing native memory arena...",
                        progressFraction = 0.05f
                    )
                )
            }

            try {
                val execResult = withContext(Dispatchers.Default) {
                    when (state.studioMode) {
                        StudioMode.FACE_SWAP -> InSwapperEngine.executeFaceSwap(
                            context = getApplication(),
                            ortEnv = ortEnv,
                            sourceBitmap = srcBitmap,
                            sourceFace = selectedSourceFace,
                            targetBitmap = tgtBitmap,
                            targetFacesToReplace = targetsToSwap,
                            enableColorTransfer = state.enableColorTransfer,
                            enableProvenanceWatermark = state.enableProvenanceWatermark,
                            allowTwoModelFallbackForTesting = state.allowTwoModelFallbackForTesting,
                            onProgress = { progress ->
                                _uiState.update { s -> s.copy(swapProgress = progress) }
                            }
                        )

                        StudioMode.HEAD_REPLACEMENT -> GhostHeadReplacementEngine.executeFullHeadReplacement(
                            context = getApplication(),
                            ortEnv = ortEnv,
                            sourceBitmap = srcBitmap,
                            sourceFace = selectedSourceFace,
                            targetBitmap = tgtBitmap,
                            targetFacesToReplace = targetsToSwap,
                            enableColorTransfer = state.enableColorTransfer,
                            enableProvenanceWatermark = state.enableProvenanceWatermark,
                            allowTwoModelFallbackForTesting = state.allowTwoModelFallbackForTesting,
                            lowMemoryMode = state.lowMemoryMode,
                            preferHardwareAccel = state.preferHardwareAcceleration,
                            onProgress = { progress ->
                                _uiState.update { s -> s.copy(swapProgress = progress) }
                            }
                        )
                    }
                }

                val logId = auditRepository.insert(
                    SwapAuditLog(
                        sourceResolution = "${srcBitmap.width}x${srcBitmap.height}",
                        targetResolution = "${tgtBitmap.width}x${tgtBitmap.height}",
                        swappedFacesCount = execResult.swappedFacesCount,
                        pipelineSummary = "[${state.studioMode.title}] ${execResult.pipelineSummary}",
                        detectionMs = execResult.detectionMs,
                        embeddingMs = execResult.embeddingMs,
                        inswapperMs = execResult.inswapperMs,
                        blendingMs = execResult.blendingMs,
                        totalMs = execResult.totalMs,
                        colorTransferEnabled = state.enableColorTransfer,
                        watermarkEnabled = state.enableProvenanceWatermark
                    )
                )

                _uiState.update {
                    it.copy(
                        isSwapping = false,
                        swapProgress = null,
                        swapResult = execResult,
                        lastAuditLogId = logId,
                        statusBannerMessage = "${state.studioMode.title} completed in ${execResult.totalMs} ms (${execResult.swappedFacesCount} target(s) replaced)."
                    )
                }
            } catch (oom: OutOfMemoryError) {
                System.gc()
                _uiState.update {
                    it.copy(
                        isSwapping = false,
                        swapProgress = null,
                        lowMemoryMode = true,
                        errorBannerMessage = "Device ran low on memory. Enabled Low-Memory Tiled Mode automatically — please try again."
                    )
                }
            } catch (t: Throwable) {
                _uiState.update {
                    it.copy(
                        isSwapping = false,
                        swapProgress = null,
                        errorBannerMessage = t.message ?: "Execution failed: ${t.javaClass.simpleName}"
                    )
                }
            }
        }
    }

    fun saveResultToGallery() {
        val result = _uiState.value.swapResult ?: return
        val logId = _uiState.value.lastAuditLogId
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) {
                ImageGalleryHelper.saveBitmapToGallery(getApplication(), result.outputBitmap)
            }
            saved.fold(
                onSuccess = { path ->
                    if (logId != null) {
                        auditRepository.markSavedToGallery(logId, path)
                    }
                    _uiState.update {
                        it.copy(
                            statusBannerMessage = "Saved to Android Gallery: $path",
                            errorBannerMessage = null
                        )
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(errorBannerMessage = "Could not save to Gallery: ${err.message}")
                    }
                }
            )
        }
    }

    fun deleteAuditLog(id: Int) {
        viewModelScope.launch {
            auditRepository.deleteById(id)
        }
    }

    fun clearAllAuditLogs() {
        viewModelScope.launch {
            auditRepository.clearAll()
        }
    }

    private fun formatBytes(bytes: Long): String {
        val mb = bytes / (1024.0 * 1024.0)
        return if (mb >= 1.0) "%.1f MB".format(mb) else "${bytes / 1024} KB"
    }
}
