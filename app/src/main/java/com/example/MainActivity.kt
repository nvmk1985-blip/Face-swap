package com.example

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.onnx.ModelSlot
import com.example.ui.AppTab
import com.example.ui.FaceSwapViewModel
import com.example.ui.tabs.HistoryAuditTab
import com.example.ui.tabs.ModelsInspectorTab
import com.example.ui.tabs.SafetyLicensesTab
import com.example.ui.tabs.StudioTab
import com.example.ui.theme.CoralError
import com.example.ui.theme.ElectricCyan
import com.example.ui.theme.MyApplicationTheme
import com.example.ui.theme.NeonEmerald

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                FaceSwapStudioApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FaceSwapStudioApp(
    viewModel: FaceSwapViewModel = viewModel()
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val auditLogs by viewModel.auditLogs.collectAsStateWithLifecycle()

    var pendingImportSlot by remember { mutableStateOf(ModelSlot.DETECTOR) }

    val sourcePhotoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.onSourceImageSelected(uri)
        }
    }

    val targetPhotoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.onTargetImageSelected(uri)
        }
    }

    val sourceFilePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.onSourceImageSelected(uri)
        }
    }

    val targetFilePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.onTargetImageSelected(uri)
        }
    }

    val onnxDocumentPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.importOnnxModel(uri, pendingImportSlot)
        }
    }

    val onnxFolderTreePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { treeUri ->
        if (treeUri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    treeUri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.importAllModelsFromFolder(treeUri)
        }
    }

    if (uiState.currentTab != AppTab.STUDIO) {
        BackHandler {
            viewModel.selectTab(AppTab.STUDIO)
        }
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = stringResource(R.string.app_name),
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.ExtraBold
                        )
                        Text(
                            text = stringResource(R.string.subtitle_offline_engine),
                            style = MaterialTheme.typography.labelMedium,
                            color = ElectricCyan
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground
                )
            )
        },
        bottomBar = {
            NavigationBar(
                containerColor = MaterialTheme.colorScheme.surface
            ) {
                NavigationBarItem(
                    selected = uiState.currentTab == AppTab.STUDIO,
                    onClick = { viewModel.selectTab(AppTab.STUDIO) },
                    icon = {
                        Icon(
                            imageVector = Icons.Default.AutoFixHigh,
                            contentDescription = stringResource(R.string.tab_studio)
                        )
                    },
                    label = { Text(stringResource(R.string.tab_studio)) },
                    modifier = Modifier.testTag("nav_tab_studio")
                )
                NavigationBarItem(
                    selected = uiState.currentTab == AppTab.MODELS,
                    onClick = { viewModel.selectTab(AppTab.MODELS) },
                    icon = {
                        Icon(
                            imageVector = Icons.Default.Memory,
                            contentDescription = stringResource(R.string.tab_models)
                        )
                    },
                    label = { Text(stringResource(R.string.tab_models)) },
                    modifier = Modifier.testTag("nav_tab_models")
                )
                NavigationBarItem(
                    selected = uiState.currentTab == AppTab.HISTORY,
                    onClick = { viewModel.selectTab(AppTab.HISTORY) },
                    icon = {
                        Icon(
                            imageVector = Icons.Default.History,
                            contentDescription = stringResource(R.string.tab_history)
                        )
                    },
                    label = { Text(stringResource(R.string.tab_history)) },
                    modifier = Modifier.testTag("nav_tab_history")
                )
                NavigationBarItem(
                    selected = uiState.currentTab == AppTab.ETHICS,
                    onClick = { viewModel.selectTab(AppTab.ETHICS) },
                    icon = {
                        Icon(
                            imageVector = Icons.Default.Shield,
                            contentDescription = stringResource(R.string.tab_ethics)
                        )
                    },
                    label = { Text(stringResource(R.string.tab_ethics)) },
                    modifier = Modifier.testTag("nav_tab_ethics")
                )
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            AnimatedVisibility(
                visible = uiState.errorBannerMessage != null || uiState.statusBannerMessage != null
            ) {
                val isError = uiState.errorBannerMessage != null
                val bannerColor = if (isError) CoralError else NeonEmerald
                val message = uiState.errorBannerMessage ?: uiState.statusBannerMessage.orEmpty()

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 6.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(bannerColor.copy(alpha = 0.16f))
                        .clickable { viewModel.clearMessages() }
                        .padding(horizontal = 14.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = message,
                        style = MaterialTheme.typography.bodyMedium,
                        color = bannerColor,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f)
                    )
                    IconButton(
                        onClick = { viewModel.clearMessages() },
                        modifier = Modifier.size(28.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Dismiss notification",
                            tint = bannerColor,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }
            }

            when (uiState.currentTab) {
                AppTab.STUDIO -> StudioTab(
                    uiState = uiState,
                    onSelectStudioMode = viewModel::selectStudioMode,
                    onPickSourcePhoto = {
                        sourcePhotoPicker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    },
                    onPickTargetPhoto = {
                        targetPhotoPicker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    },
                    onBrowseSourceFile = {
                        sourceFilePicker.launch(arrayOf("image/*"))
                    },
                    onBrowseTargetFile = {
                        targetFilePicker.launch(arrayOf("image/*"))
                    },
                    onDetectHead = viewModel::detectHeadsOnLoadedPhotos,
                    onPreviewHead = viewModel::generateHeadReplacementPreview,
                    onSelectSourceFace = viewModel::selectSourceFace,
                    onSelectTargetFace = viewModel::selectTargetFace,
                    onToggleReplaceAllFaces = viewModel::setReplaceAllTargetFaces,
                    onSkinToneModeChanged = viewModel::setSkinToneMode,
                    onFaceReactionModeChanged = viewModel::setFaceReactionMode,
                    onConsentChanged = viewModel::setConsentConfirmed,
                    onColorTransferChanged = viewModel::setEnableColorTransfer,
                    onWatermarkChanged = viewModel::setEnableProvenanceWatermark,
                    onTwoModelFallbackChanged = viewModel::setAllowTwoModelFallback,
                    onHardwareAccelChanged = viewModel::setPreferHardwareAcceleration,
                    onLowMemoryChanged = viewModel::setLowMemoryMode,
                    onRunSwap = viewModel::runActiveModePipeline,
                    onSaveToGallery = viewModel::saveResultToGallery,
                    onToggleCompareOriginal = viewModel::toggleComparisonMode,
                    onOpenModelsTab = { viewModel.selectTab(AppTab.MODELS) }
                )

                AppTab.MODELS -> ModelsInspectorTab(
                    inspections = uiState.modelInspections,
                    isInspecting = uiState.isInspectingModels,
                    memoryServiceState = uiState.memoryServiceState,
                    onImportModelForSlot = { slot ->
                        pendingImportSlot = slot
                        onnxDocumentPicker.launch(arrayOf("*/*"))
                    },
                    onRefreshModels = viewModel::refreshModelInspections,
                    onImportAllFromFolder = {
                        onnxFolderTreePicker.launch(null)
                    },
                    onPreloadMemoryModels = viewModel::preloadModelsIntoMemory,
                    onReleaseMemoryModels = viewModel::releaseModelsFromMemory
                )

                AppTab.HISTORY -> HistoryAuditTab(
                    auditLogs = auditLogs,
                    onDeleteLog = viewModel::deleteAuditLog,
                    onClearAll = viewModel::clearAllAuditLogs
                )

                AppTab.ETHICS -> SafetyLicensesTab()
            }
        }
    }
}
