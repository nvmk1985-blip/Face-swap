package com.example.ui.tabs

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.FaceRetouchingNatural
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Preview
import androidx.compose.material.icons.filled.Radar
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.VerifiedUser
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.R
import com.example.onnx.DetectedFace
import com.example.ui.FaceSwapUiState
import com.example.ui.StudioMode
import com.example.ui.components.FaceDetectionCanvas
import com.example.ui.theme.AmberWarning
import com.example.ui.theme.ElectricCyan
import com.example.ui.theme.NeonEmerald
import com.example.ui.theme.ObsidianBg
import com.example.ui.theme.RoyalViolet

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun StudioTab(
    uiState: FaceSwapUiState,
    onSelectStudioMode: (StudioMode) -> Unit,
    onPickSourcePhoto: () -> Unit,
    onPickTargetPhoto: () -> Unit,
    onDetectHead: () -> Unit,
    onPreviewHead: () -> Unit,
    onSelectSourceFace: (Int) -> Unit,
    onSelectTargetFace: (Int) -> Unit,
    onToggleReplaceAllFaces: (Boolean) -> Unit,
    onConsentChanged: (Boolean) -> Unit,
    onColorTransferChanged: (Boolean) -> Unit,
    onWatermarkChanged: (Boolean) -> Unit,
    onTwoModelFallbackChanged: (Boolean) -> Unit,
    onHardwareAccelChanged: (Boolean) -> Unit,
    onLowMemoryChanged: (Boolean) -> Unit,
    onRunSwap: () -> Unit,
    onSaveToGallery: () -> Unit,
    onToggleCompareOriginal: (Boolean) -> Unit,
    onOpenModelsTab: () -> Unit,
    onBrowseSourceFile: () -> Unit = onPickSourcePhoto,
    onBrowseTargetFile: () -> Unit = onPickTargetPhoto,
    modifier: Modifier = Modifier
) {
    val isHeadMode = uiState.studioMode == StudioMode.HEAD_REPLACEMENT

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 640.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Mode Switcher: MODE 1 (Face Swap) vs MODE 2 (Head Replacement)
            StudioModeSwitcherCard(
                currentMode = uiState.studioMode,
                onSelectMode = onSelectStudioMode
            )

            // Hero Studio Banner + Pipeline Status
            HeroPipelineStatusCard(
                uiState = uiState,
                onOpenModelsTab = onOpenModelsTab
            )

            // 1. Source Photo Selection Card (SOURCE FACE vs SOURCE HEAD)
            PhotoSelectionCard(
                title = if (isHeadMode) "SOURCE HEAD" else "1. Source Identity Face",
                subtitle = if (isHeadMode) {
                    "Head, hair, skull structure & upper neck to transfer"
                } else {
                    "Select photo containing the donor face identity (112x112 -> 512-D embedding)"
                },
                buttonLabel = stringResource(R.string.btn_select_source),
                buttonTestTag = "select_source_photo_button",
                browseTestTag = "browse_source_file_button",
                bitmap = uiState.sourceBitmap,
                faces = uiState.sourceFaces,
                selectedFaceIndex = uiState.selectedSourceFaceIndex,
                replaceAllFaces = false,
                showCranialHeadBounds = isHeadMode,
                isDetecting = uiState.isDetectingSource,
                onPickPhoto = onPickSourcePhoto,
                onBrowseFile = onBrowseSourceFile,
                onSelectFace = onSelectSourceFace,
                showMultiFaceToggle = false,
                onToggleMultiFace = {}
            )

            // 2. Target Photo Selection Card
            PhotoSelectionCard(
                title = if (isHeadMode) "TARGET PHOTO" else "2. Target Scene Photo",
                subtitle = if (isHeadMode) {
                    "Body, pose, clothing & background to preserve (tap head to replace)"
                } else {
                    "Select target photo and tap any detected face to replace"
                },
                buttonLabel = stringResource(R.string.btn_select_target),
                buttonTestTag = "select_target_photo_button",
                browseTestTag = "browse_target_file_button",
                bitmap = uiState.targetBitmap,
                faces = uiState.targetFaces,
                selectedFaceIndex = uiState.selectedTargetFaceIndex,
                replaceAllFaces = uiState.replaceAllTargetFaces,
                showCranialHeadBounds = isHeadMode,
                isDetecting = uiState.isDetectingTarget,
                onPickPhoto = onPickTargetPhoto,
                onBrowseFile = onBrowseTargetFile,
                onSelectFace = onSelectTargetFace,
                showMultiFaceToggle = uiState.targetFaces.size > 1,
                onToggleMultiFace = onToggleReplaceAllFaces
            )

            // Mode 2 Specific Controls: [ Detect Head ] and [ Preview ]
            if (isHeadMode) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedButton(
                        onClick = onDetectHead,
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .testTag("detect_head_button"),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Radar,
                            contentDescription = stringResource(R.string.btn_detect_head),
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(stringResource(R.string.btn_detect_head))
                    }

                    OutlinedButton(
                        onClick = onPreviewHead,
                        enabled = !uiState.isGeneratingHeadPreview,
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .testTag("preview_head_button"),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Preview,
                            contentDescription = stringResource(R.string.btn_preview_head),
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            if (uiState.isGeneratingHeadPreview) {
                                "Segmenting..."
                            } else {
                                stringResource(R.string.btn_preview_head)
                            }
                        )
                    }
                }

                // Live Head / Hair / Neck Segmentation & Cranial Alignment Preview
                uiState.headPreviewState?.let { preview ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surface
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(14.dp),
                            verticalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            Text(
                                text = "Head / Hair / Neck Segmentation & Pose Preview",
                                style = MaterialTheme.typography.titleMedium,
                                color = ElectricCyan
                            )
                            Text(
                                text = "Parser: ${preview.segmentationLabel}",
                                style = MaterialTheme.typography.labelMedium,
                                color = NeonEmerald
                            )
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceEvenly
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Image(
                                        bitmap = preview.segmentedSourceHeadBitmap.asImageBitmap(),
                                        contentDescription = "Segmented Source Head & Hair Mask",
                                        modifier = Modifier
                                            .size(128.dp)
                                            .clip(RoundedCornerShape(12.dp))
                                            .border(
                                                1.5.dp,
                                                ElectricCyan,
                                                RoundedCornerShape(12.dp)
                                            )
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = "Source Matte (${"%.1f".format(preview.sourceRollDeg)}°)",
                                        style = MaterialTheme.typography.labelMedium
                                    )
                                }
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Image(
                                        bitmap = preview.alignedTargetHeadBitmap.asImageBitmap(),
                                        contentDescription = "Aligned Target Head Region",
                                        modifier = Modifier
                                            .size(128.dp)
                                            .clip(RoundedCornerShape(12.dp))
                                            .border(
                                                1.5.dp,
                                                RoyalViolet,
                                                RoundedCornerShape(12.dp)
                                            )
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = "Target Pose (${"%.1f".format(preview.targetRollDeg)}°)",
                                        style = MaterialTheme.typography.labelMedium
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // 3. Performance, Blending & Ethical Consent Card
            SafeguardsAndBlendingCard(
                uiState = uiState,
                onConsentChanged = onConsentChanged,
                onColorTransferChanged = onColorTransferChanged,
                onWatermarkChanged = onWatermarkChanged,
                onTwoModelFallbackChanged = onTwoModelFallbackChanged,
                onHardwareAccelChanged = onHardwareAccelChanged,
                onLowMemoryChanged = onLowMemoryChanged
            )

            // 4. Primary Action Button: [ REPLACE HEAD ] or [ Run Offline Face Swap ]
            Button(
                onClick = onRunSwap,
                enabled = uiState.canExecuteSwap,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .testTag("run_face_swap_button"),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isHeadMode) NeonEmerald else ElectricCyan,
                    contentColor = ObsidianBg
                )
            ) {
                if (uiState.isSwapping) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(22.dp),
                        color = ObsidianBg,
                        strokeWidth = 2.5.dp
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        text = if (isHeadMode) {
                            "Replacing Head, Hair & Neck..."
                        } else {
                            "Running Offline ONNX Inference..."
                        },
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.AutoFixHigh,
                        contentDescription = null
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        text = if (isHeadMode) {
                            stringResource(R.string.btn_replace_head)
                        } else {
                            stringResource(R.string.btn_run_swap)
                        },
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // Processing Progress Stages
            AnimatedVisibility(visible = uiState.swapProgress != null) {
                uiState.swapProgress?.let { prog ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        ),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = prog.stageTitle,
                                    style = MaterialTheme.typography.titleMedium,
                                    color = ElectricCyan
                                )
                                Text(
                                    text = "${(prog.progressFraction * 100).toInt()}%",
                                    style = MaterialTheme.typography.labelMedium,
                                    color = NeonEmerald
                                )
                            }
                            LinearProgressIndicator(
                                progress = { prog.progressFraction },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(8.dp)
                                    .clip(CircleShape),
                                color = ElectricCyan,
                                trackColor = ObsidianBg
                            )
                            Text(
                                text = prog.detailMessage,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            if (isHeadMode) {
                                Text(
                                    text = "Stages: Detection -> Alignment -> Head generation -> Segmentation -> Blending -> Enhancement",
                                    style = MaterialTheme.typography.labelMedium,
                                    color = NeonEmerald
                                )
                            }
                        }
                    }
                }
            }

            // 5. Result Preview + [ Before / After ] + [ Save Image ]
            uiState.swapResult?.let {
                SwapResultCard(
                    uiState = uiState,
                    onToggleCompareOriginal = onToggleCompareOriginal,
                    onSaveToGallery = onSaveToGallery
                )
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun StudioModeSwitcherCard(
    currentMode: StudioMode,
    onSelectMode: (StudioMode) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                StudioMode.entries.forEach { mode ->
                    val selected = currentMode == mode
                    val containerColor = if (selected) {
                        if (mode == StudioMode.HEAD_REPLACEMENT) NeonEmerald else ElectricCyan
                    } else {
                        MaterialTheme.colorScheme.surfaceVariant
                    }
                    val contentColor = if (selected) {
                        ObsidianBg
                    } else {
                        MaterialTheme.colorScheme.onSurface
                    }

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(containerColor)
                            .clickable { onSelectMode(mode) }
                            .testTag(
                                if (mode == StudioMode.FACE_SWAP) {
                                    "mode_selector_face_swap"
                                } else {
                                    "mode_selector_head_replacement"
                                }
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = mode.title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = contentColor
                        )
                    }
                }
            }
            Text(
                text = currentMode.badge,
                style = MaterialTheme.typography.labelMedium,
                color = if (currentMode == StudioMode.HEAD_REPLACEMENT) NeonEmerald else ElectricCyan
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun HeroPipelineStatusCard(
    uiState: FaceSwapUiState,
    onOpenModelsTab: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(116.dp)
            ) {
                Image(
                    painter = painterResource(id = R.drawable.img_hero_pipeline),
                    contentDescription = "Biometric face and head alignment banner",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(
                                    ObsidianBg.copy(alpha = 0.25f),
                                    ObsidianBg.copy(alpha = 0.92f)
                                )
                            )
                        )
                )
                Column(
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .padding(14.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.VerifiedUser,
                            contentDescription = "Offline Air-Gapped Badge",
                            tint = NeonEmerald,
                            modifier = Modifier.size(18.dp)
                        )
                        Text(
                            text = "AIRPLANE-MODE READY • 100% LOCAL ON-DEVICE INFERENCE",
                            style = MaterialTheme.typography.labelMedium,
                            color = NeonEmerald
                        )
                    }
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = if (uiState.studioMode == StudioMode.HEAD_REPLACEMENT) {
                            "Mode 2: Full Head, Hair, Skull & Neck Replacement"
                        } else {
                            "Mode 1: SCRFD-10G + ArcFace + InSwapper-128"
                        },
                        style = MaterialTheme.typography.titleLarge,
                        color = Color.White
                    )
                }
            }

            Column(
                modifier = Modifier.padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    ModelStatusPill(
                        label = "det_10g.onnx",
                        isReady = uiState.isDetectorReady,
                        onClick = onOpenModelsTab
                    )
                    ModelStatusPill(
                        label = "inswapper_128.onnx",
                        isReady = uiState.isSwapperReady,
                        onClick = onOpenModelsTab
                    )
                    ModelStatusPill(
                        label = "w600k_r50.onnx",
                        isReady = uiState.isRecognizerReady,
                        onClick = onOpenModelsTab
                    )
                    if (uiState.studioMode == StudioMode.HEAD_REPLACEMENT) {
                        ModelStatusPill(
                            label = if (uiState.isSegmentationOnnxReady) {
                                "segformer_B5_ce.onnx"
                            } else {
                                "Head/Hair Matte: Built-In CV"
                            },
                            isReady = true,
                            onClick = onOpenModelsTab
                        )
                    }
                }

                if (!uiState.isDetectorReady || !uiState.isSwapperReady) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(10.dp))
                            .background(MaterialTheme.colorScheme.surfaceVariant)
                            .clickable(onClick = onOpenModelsTab)
                            .padding(10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Memory,
                            contentDescription = "Model Status",
                            tint = AmberWarning,
                            modifier = Modifier.size(20.dp)
                        )
                        Text(
                            text = "Tap to open Model Status & GHOST 2.0 Feasibility Matrix or import .onnx files.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ModelStatusPill(
    label: String,
    isReady: Boolean,
    onClick: () -> Unit
) {
    val tint = if (isReady) NeonEmerald else AmberWarning
    Surface(
        modifier = Modifier.clickable(onClick = onClick),
        shape = RoundedCornerShape(50),
        color = tint.copy(alpha = 0.14f)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Icon(
                imageVector = if (isReady) Icons.Default.CheckCircle else Icons.Default.WarningAmber,
                contentDescription = label,
                tint = tint,
                modifier = Modifier.size(15.dp)
            )
            Text(
                text = if (isReady) "✓ $label" else "✗ $label",
                style = MaterialTheme.typography.labelMedium,
                color = tint
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PhotoSelectionCard(
    title: String,
    subtitle: String,
    buttonLabel: String,
    buttonTestTag: String,
    browseTestTag: String,
    bitmap: android.graphics.Bitmap?,
    faces: List<DetectedFace>,
    selectedFaceIndex: Int,
    replaceAllFaces: Boolean,
    showCranialHeadBounds: Boolean,
    isDetecting: Boolean,
    onPickPhoto: () -> Unit,
    onBrowseFile: () -> Unit,
    onSelectFace: (Int) -> Unit,
    showMultiFaceToggle: Boolean,
    onToggleMultiFace: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Column(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    OutlinedButton(
                        onClick = onPickPhoto,
                        modifier = Modifier
                            .weight(1f)
                            .testTag(buttonTestTag),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.AddPhotoAlternate,
                            contentDescription = buttonLabel,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = buttonLabel)
                    }
                    OutlinedButton(
                        onClick = onBrowseFile,
                        modifier = Modifier
                            .weight(1f)
                            .testTag(browseTestTag),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.AddPhotoAlternate,
                            contentDescription = "Browse Files",
                            modifier = Modifier.size(18.dp),
                            tint = NeonEmerald
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = "Browse Files", color = NeonEmerald)
                    }
                }
            }

            if (isDetecting) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(ObsidianBg),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = ElectricCyan)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Detecting Face, 5-Point Keypoints & Cranial Volume...",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            } else if (bitmap != null) {
                FaceDetectionCanvas(
                    bitmap = bitmap,
                    faces = faces,
                    selectedFaceIndex = selectedFaceIndex,
                    highlightAllFaces = replaceAllFaces,
                    showCranialHeadBounds = showCranialHeadBounds,
                    onFaceTapped = onSelectFace,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(250.dp)
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Detected: ${faces.size} (${bitmap.width}x${bitmap.height}px)",
                        style = MaterialTheme.typography.labelMedium,
                        color = if (faces.isNotEmpty()) NeonEmerald else AmberWarning
                    )
                    faces.firstOrNull()?.let { first ->
                        Text(
                            text = first.detectorSource,
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                if (faces.size > 1) {
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        faces.forEach { face ->
                            FilterChip(
                                selected = !replaceAllFaces && face.index == selectedFaceIndex,
                                onClick = { onSelectFace(face.index) },
                                label = {
                                    Text(
                                        "Head #${face.index + 1} (${(face.score * 100).toInt()}%)"
                                    )
                                }
                            )
                        }
                    }
                }

                if (showMultiFaceToggle) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(10.dp))
                            .background(MaterialTheme.colorScheme.surfaceVariant)
                            .padding(horizontal = 12.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = stringResource(R.string.multi_face_mode_label),
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = FontWeight.SemiBold
                            )
                            Text(
                                text = "Process all ${faces.size} detected heads/faces sequentially",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Switch(
                            checked = replaceAllFaces,
                            onCheckedChange = onToggleMultiFace
                        )
                    }
                }
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(140.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(ObsidianBg)
                        .border(
                            1.dp,
                            MaterialTheme.colorScheme.surfaceVariant,
                            RoundedCornerShape(12.dp)
                        )
                        .clickable(onClick = onPickPhoto),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.FaceRetouchingNatural,
                            contentDescription = buttonLabel,
                            tint = ElectricCyan,
                            modifier = Modifier.size(34.dp)
                        )
                        Text(
                            text = "Tap to select photo from Android Gallery",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SafeguardsAndBlendingCard(
    uiState: FaceSwapUiState,
    onConsentChanged: (Boolean) -> Unit,
    onColorTransferChanged: (Boolean) -> Unit,
    onWatermarkChanged: (Boolean) -> Unit,
    onTwoModelFallbackChanged: (Boolean) -> Unit,
    onHardwareAccelChanged: (Boolean) -> Unit,
    onLowMemoryChanged: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                text = "Hardware Acceleration, Memory & Ethical Safeguards",
                style = MaterialTheme.typography.titleMedium
            )

            if (!uiState.isRecognizerReady && uiState.studioMode == StudioMode.FACE_SWAP) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(AmberWarning.copy(alpha = 0.14f))
                        .border(1.dp, AmberWarning.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = "Missing w600k_r50.onnx (512-D ArcFace Recognizer)",
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Bold,
                        color = AmberWarning
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "Enable 2-Model Testing Mode (when w600k_r50.onnx is absent)",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.weight(1f)
                        )
                        Switch(
                            checked = uiState.allowTwoModelFallbackForTesting,
                            onCheckedChange = onTwoModelFallbackChanged,
                            modifier = Modifier.testTag("two_model_fallback_switch")
                        )
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Android NNAPI / GPU Acceleration (with CPU Fallback)",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "Uses hardware delegate where supported and falls back to multi-thread CPU",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = uiState.preferHardwareAcceleration,
                    onCheckedChange = onHardwareAccelChanged
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Low-Memory Tiled Optimization Mode",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "Sequential model unloading + 256x256 head tiles for low-RAM devices",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = uiState.lowMemoryMode,
                    onCheckedChange = onLowMemoryChanged
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Checkbox(
                    checked = uiState.enableColorTransfer,
                    onCheckedChange = onColorTransferChanged
                )
                Text(
                    text = stringResource(R.string.color_transfer_checkbox_label),
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Checkbox(
                    checked = uiState.enableProvenanceWatermark,
                    onCheckedChange = onWatermarkChanged
                )
                Text(
                    text = stringResource(R.string.watermark_checkbox_label),
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .padding(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Checkbox(
                    checked = uiState.consentConfirmed,
                    onCheckedChange = onConsentChanged,
                    modifier = Modifier.testTag("consent_checkbox")
                )
                Text(
                    text = stringResource(R.string.consent_checkbox_label),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}

@Composable
private fun SwapResultCard(
    uiState: FaceSwapUiState,
    onToggleCompareOriginal: (Boolean) -> Unit,
    onSaveToGallery: () -> Unit
) {
    val result = uiState.swapResult ?: return
    val displayBitmap = if (uiState.showOriginalInComparison && uiState.targetBitmap != null) {
        uiState.targetBitmap
    } else {
        result.outputBitmap
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (uiState.showOriginalInComparison) {
                            "Before (Original Target)"
                        } else {
                            "After (${uiState.studioMode.title} Result)"
                        },
                        style = MaterialTheme.typography.titleLarge,
                        color = ElectricCyan
                    )
                    Text(
                        text = "Total Latency: ${result.totalMs} ms • ${result.swappedFacesCount} Replaced",
                        style = MaterialTheme.typography.labelMedium,
                        color = NeonEmerald
                    )
                }

                OutlinedButton(
                    onClick = { onToggleCompareOriginal(!uiState.showOriginalInComparison) },
                    modifier = Modifier.testTag("compare_before_after_button")
                ) {
                    Icon(
                        imageVector = Icons.Default.Compare,
                        contentDescription = stringResource(R.string.btn_before_after),
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(stringResource(R.string.btn_before_after))
                }
            }

            Image(
                bitmap = displayBitmap.asImageBitmap(),
                contentDescription = "Output Preview",
                modifier = Modifier
                    .fillMaxWidth()
                    .height(320.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(ObsidianBg),
                contentScale = ContentScale.Fit
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                AlignedCropThumb(bitmap = result.alignedSource112, label = "Source Crop")
                AlignedCropThumb(bitmap = result.alignedTarget128, label = "Target Crop")
                AlignedCropThumb(bitmap = result.rawSwapped128, label = "Synthesized")
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(ObsidianBg)
                    .padding(10.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Align/Emb: ${result.embeddingMs}ms",
                    style = MaterialTheme.typography.labelMedium,
                    color = ElectricCyan
                )
                Text(
                    text = "Gen: ${result.inswapperMs}ms",
                    style = MaterialTheme.typography.labelMedium,
                    color = RoyalViolet
                )
                Text(
                    text = "Seg/Blend: ${result.blendingMs}ms",
                    style = MaterialTheme.typography.labelMedium,
                    color = NeonEmerald
                )
            }

            Button(
                onClick = onSaveToGallery,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp)
                    .testTag("save_to_gallery_button"),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = NeonEmerald,
                    contentColor = ObsidianBg
                )
            ) {
                Icon(
                    imageVector = Icons.Default.SaveAlt,
                    contentDescription = stringResource(R.string.btn_save_gallery)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = stringResource(R.string.btn_save_gallery),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
private fun AlignedCropThumb(
    bitmap: android.graphics.Bitmap,
    label: String
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = label,
            modifier = Modifier
                .size(76.dp)
                .clip(RoundedCornerShape(10.dp))
                .border(1.dp, ElectricCyan.copy(alpha = 0.5f), RoundedCornerShape(10.dp))
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium
        )
    }
}
