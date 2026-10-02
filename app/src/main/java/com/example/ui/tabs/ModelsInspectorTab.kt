package com.example.ui.tabs

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.onnx.GhostComponentFeasibility
import com.example.onnx.ModelRequirementLevel
import com.example.onnx.ModelSlot
import com.example.onnx.OnnxModelInspection
import com.example.onnx.OnnxProtobufInspector
import com.example.ui.theme.AmberWarning
import com.example.ui.theme.CoralError
import com.example.ui.theme.ElectricCyan
import com.example.ui.theme.NeonEmerald
import com.example.ui.theme.ObsidianBg
import com.example.ui.theme.RoyalViolet

@Composable
fun ModelsInspectorTab(
    inspections: List<OnnxModelInspection>,
    isInspecting: Boolean,
    onImportModelForSlot: (ModelSlot) -> Unit,
    onRefreshModels: () -> Unit,
    onImportAllFromFolder: () -> Unit = {},
    modifier: Modifier = Modifier
) {
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
            // 1. GHOST 2.0 Feasibility Statement & Directory Structure Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.weight(1f)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Info,
                                contentDescription = "Model Status",
                                tint = ElectricCyan
                            )
                            Text(
                                text = "Local Model Status & Directory Layout",
                                style = MaterialTheme.typography.titleMedium,
                                color = ElectricCyan
                            )
                        }
                        OutlinedButton(
                            onClick = onRefreshModels,
                            enabled = !isInspecting,
                            modifier = Modifier.testTag("refresh_models_button")
                        ) {
                            Icon(
                                imageVector = Icons.Default.Refresh,
                                contentDescription = "Rescan Models",
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Rescan")
                        }
                    }

                    Button(
                        onClick = onImportAllFromFolder,
                        enabled = !isInspecting,
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("import_all_folder_button"),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Folder,
                            contentDescription = "Auto-Import All from Folder",
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "1-Tap Auto-Import All .onnx from Folder",
                            fontWeight = FontWeight.Bold
                        )
                    }

                    Text(
                        text = "Tip: Put all your .onnx files in one folder (e.g. Downloads) and tap the button above once. The app will load all models at once and remember the folder!",
                        style = MaterialTheme.typography.labelMedium,
                        color = NeonEmerald
                    )

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
                            text = "FULL GHOST 2.0 ANDROID PORT IS NOT CURRENTLY FEASIBLE AS IMPLEMENTED",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            color = AmberWarning
                        )
                        Text(
                            text = "Why: Official GHOST 2.0 checkpoints (aligner_1020_gaze_final.ckpt ~1.15 GB and blender_lama.ckpt ~680 MB) are PyTorch Lightning checkpoints coupled to PyTorch3D / Nvdiffrast CUDA mesh rasterizers and EMOCA/FLAME pickle graphs that cannot run in Android ONNX Runtime. Below is the Android-compatible offline architecture using ONNX Runtime Mobile + native CV modules.",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }

                    // Local Model Directory Tree
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(10.dp))
                            .background(ObsidianBg)
                            .padding(12.dp),
                        verticalAlignment = Alignment.Top,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Folder,
                            contentDescription = "Local Model Directory",
                            tint = NeonEmerald
                        )
                        Text(
                            text = "Local On-Device Storage Hierarchy (filesDir/models/):\n" +
                                "├── face_swap/      (det_10g.onnx, w600k_r50.onnx, inswapper_128.onnx)\n" +
                                "├── head_swap/      (Android GHOST Head Pose & Cranial Warp Config)\n" +
                                "├── segmentation/   (segformer_B5_ce.onnx — or built-in Geodesic Matte)\n" +
                                "├── matting/        (modnet.onnx — or built-in Joint Bilateral Filter)\n" +
                                "├── inpainting/     (lama_fp32.onnx — or built-in Boundary Marching)\n" +
                                "└── enhancement/    (gfpgan_1.4.onnx — or built-in Tiled Unsharp Pass)",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }

            // 2. Installed vs Missing Model Slots
            Text(
                text = "On-Device ONNX Model Slots (No Automatic Background Downloads)",
                style = MaterialTheme.typography.titleMedium,
                color = ElectricCyan
            )

            ModelSlot.entries.forEach { slot ->
                val inspection = inspections.firstOrNull { it.slot == slot }
                ModelInspectionCard(
                    slot = slot,
                    inspection = inspection,
                    onImportModel = { onImportModelForSlot(slot) }
                )
            }

            // 3. Official GHOST 2.0 Component Feasibility Report
            Text(
                text = "GHOST 2.0 Component-by-Component Android Feasibility Matrix",
                style = MaterialTheme.typography.titleMedium,
                color = NeonEmerald
            )

            OnnxProtobufInspector.GHOST_2_FEASIBILITY_MATRIX.forEach { item ->
                GhostFeasibilityCard(item = item)
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun ModelInspectionCard(
    slot: ModelSlot,
    inspection: OnnxModelInspection?,
    onImportModel: () -> Unit
) {
    val isReady = inspection?.isValidOnnx == true
    val statusColor = when {
        isReady -> NeonEmerald
        slot.requirementLevel == ModelRequirementLevel.OPTIONAL_HEAD_SWAP -> ElectricCyan
        else -> CoralError
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = if (isReady) Icons.Default.CheckCircle else Icons.Default.WarningAmber,
                        contentDescription = slot.displayName,
                        tint = statusColor,
                        modifier = Modifier.size(20.dp)
                    )
                    Text(
                        text = if (isReady) "✓ Installed: ${slot.canonicalFileName}" else "✗ Missing: ${slot.canonicalFileName}",
                        style = MaterialTheme.typography.titleMedium,
                        color = statusColor
                    )
                }

                Surface(
                    shape = RoundedCornerShape(50),
                    color = if (slot.requirementLevel == ModelRequirementLevel.OPTIONAL_HEAD_SWAP) {
                        RoyalViolet.copy(alpha = 0.22f)
                    } else {
                        AmberWarning.copy(alpha = 0.20f)
                    }
                ) {
                    Text(
                        text = slot.requirementLevel.label,
                        style = MaterialTheme.typography.labelMedium,
                        color = if (slot.requirementLevel == ModelRequirementLevel.OPTIONAL_HEAD_SWAP) {
                            ElectricCyan
                        } else {
                            AmberWarning
                        },
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                }
            }

            Text(
                text = "Directory: ${slot.categoryTitle}${slot.canonicalFileName} • Approx Size: ${slot.approximateSizeLabel}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurface
            )

            Text(
                text = slot.roleSummary,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(ObsidianBg)
                    .border(1.dp, statusColor.copy(alpha = 0.35f), RoundedCornerShape(12.dp))
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "• Input:  ${slot.expectedInputSignature}\n• Output: ${slot.expectedOutputSignature}",
                    style = MaterialTheme.typography.labelMedium
                )
                if (inspection != null && inspection.isPresent) {
                    val mb = "%.2f MB".format(inspection.fileSizeBytes / (1024.0 * 1024.0))
                    Text(
                        text = "• Installed Size: $mb | SHA-256: ${inspection.sha256Prefix ?: "n/a"} | IR v${inspection.irVersion ?: "?"}",
                        style = MaterialTheme.typography.labelMedium,
                        color = NeonEmerald
                    )
                    if (inspection.runtimeInputs.isNotEmpty()) {
                        val inDesc = inspection.runtimeInputs.joinToString("; ") {
                            "${it.name}: ${it.elementType}${it.formattedShape()}"
                        }
                        Text(
                            text = "• Verified Runtime Inputs: $inDesc",
                            style = MaterialTheme.typography.labelMedium,
                            color = ElectricCyan
                        )
                    }
                    if (slot == ModelSlot.SWAPPER) {
                        Text(
                            text = "• Embedded [512, 512] emap Matrix: ${if (inspection.hasEmbeddedEmap512x512) "EXTRACTED" else "NOT FOUND"}",
                            style = MaterialTheme.typography.labelMedium,
                            color = NeonEmerald
                        )
                    }
                } else {
                    Text(
                        text = "• Action: ${inspection?.verificationNote ?: "Place ${slot.canonicalFileName} in ${slot.categoryTitle}"}",
                        style = MaterialTheme.typography.labelMedium,
                        color = AmberWarning
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "License: ${slot.licenseSummary}",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Button(
                    onClick = onImportModel,
                    modifier = Modifier.testTag("import_model_${slot.id}_button"),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.FileOpen,
                        contentDescription = "Import ${slot.canonicalFileName}",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(if (isReady) "Replace .onnx" else "Import .onnx")
                }
            }
        }
    }
}

@Composable
private fun GhostFeasibilityCard(item: GhostComponentFeasibility) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = item.componentName,
                    style = MaterialTheme.typography.titleMedium,
                    color = ElectricCyan,
                    modifier = Modifier.weight(1f)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (item.runsDirectlyInAndroidOrt) "ONNX DIRECT: YES" else "PYTORCH CKPT: NO",
                    style = MaterialTheme.typography.labelMedium,
                    color = if (item.runsDirectlyInAndroidOrt) NeonEmerald else AmberWarning
                )
            }
            Text(
                text = "Purpose: ${item.purpose}",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = "Framework: ${item.framework} | Size: ${item.approxSize} | RAM: ${item.expectedRam}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "I/O: ${item.inputOutputSpec}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "Android Strategy: ${item.conversionAndReplacementStatus}",
                style = MaterialTheme.typography.bodyMedium,
                color = NeonEmerald
            )
            Text(
                text = "Hardware: ${item.hardwareCompatibility} | License: ${item.license}",
                style = MaterialTheme.typography.labelMedium,
                color = AmberWarning
            )
        }
    }
}
