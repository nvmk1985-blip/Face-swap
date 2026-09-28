package com.example.ui.tabs

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.Gavel
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.data.SwapAuditLog
import com.example.ui.theme.AmberWarning
import com.example.ui.theme.ElectricCyan
import com.example.ui.theme.NeonEmerald
import com.example.ui.theme.ObsidianBg
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HistoryAuditTab(
    auditLogs: List<SwapAuditLog>,
    onDeleteLog: (Int) -> Unit,
    onClearAll: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .widthIn(max = 640.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Local Swap Audit Log (Room SQLite)",
                        style = MaterialTheme.typography.titleLarge
                    )
                    Text(
                        text = "100% on-device performance & provenance telemetry",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (auditLogs.isNotEmpty()) {
                    OutlinedButton(
                        onClick = onClearAll,
                        modifier = Modifier.testTag("clear_audit_logs_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.DeleteSweep,
                            contentDescription = "Clear All Logs",
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Clear")
                    }
                }
            }

            if (auditLogs.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.History,
                            contentDescription = "No Swap Sessions Yet",
                            tint = ElectricCyan,
                            modifier = Modifier.size(48.dp)
                        )
                        Text(
                            text = "No offline swap executions recorded yet.",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Text(
                            text = "Run a face swap in the Studio tab to log per-stage ONNX Runtime latency.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(auditLogs, key = { it.id }) { log ->
                        AuditLogItemCard(
                            log = log,
                            onDelete = { onDeleteLog(log.id) }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun AuditLogItemCard(
    log: SwapAuditLog,
    onDelete: () -> Unit
) {
    val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Session #${log.id} • ${formatter.format(Date(log.timestampMillis))}",
                    style = MaterialTheme.typography.titleMedium,
                    color = ElectricCyan
                )
                IconButton(onClick = onDelete) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Delete Audit Entry",
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Text(
                text = "Pipeline: ${log.pipelineSummary}",
                style = MaterialTheme.typography.labelMedium
            )
            Text(
                text = "Source: ${log.sourceResolution} -> Target: ${log.targetResolution} (${log.swappedFacesCount} face(s))",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(ObsidianBg)
                    .padding(8.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Embed: ${log.embeddingMs}ms",
                    style = MaterialTheme.typography.labelMedium,
                    color = ElectricCyan
                )
                Text(
                    text = "Swap: ${log.inswapperMs}ms",
                    style = MaterialTheme.typography.labelMedium,
                    color = NeonEmerald
                )
                Text(
                    text = "Blend: ${log.blendingMs}ms",
                    style = MaterialTheme.typography.labelMedium,
                    color = AmberWarning
                )
                Text(
                    text = "Total: ${log.totalMs}ms",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold
                )
            }

            log.savedGalleryPath?.let { path ->
                Text(
                    text = "Saved: $path",
                    style = MaterialTheme.typography.labelMedium,
                    color = NeonEmerald
                )
            }
        }
    }
}

@Composable
fun SafetyLicensesTab(modifier: Modifier = Modifier) {
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
            // 1. Model License Verification
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Gavel,
                            contentDescription = "AI Model License Verification",
                            tint = AmberWarning
                        )
                        Text(
                            text = "AI Model License & Redistribution Verification",
                            style = MaterialTheme.typography.titleMedium,
                            color = AmberWarning
                        )
                    }
                    Text(
                        text = "1. inswapper_128.onnx (InsightFace Model Zoo):\n" +
                            "   • License: STRICT NON-COMMERCIAL RESEARCH ONLY.\n" +
                            "   • Commercial use, commercial redistribution, or paid app distribution is prohibited without a separate commercial license from InsightFace.\n\n" +
                            "2. det_10g.onnx (SCRFD-10G_KPS) & w600k_r50.onnx (ArcFace WebFace600K):\n" +
                            "   • License: InsightFace python-package code is MIT, but pre-trained buffalo_l weights (det_10g.onnx and w600k_r50.onnx trained on WebFace600K) are restricted to Non-Commercial Academic/Research use.\n\n" +
                            "3. Microsoft ONNX Runtime Mobile (com.microsoft.onnxruntime:onnxruntime-android):\n" +
                            "   • License: MIT License (Permissive commercial & non-commercial use).",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // 2. Safeguards Against Non-Consensual Intimate Imagery (NCII) & Deceptive Misuse
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Shield,
                            contentDescription = "Safeguards Against Misuse",
                            tint = NeonEmerald
                        )
                        Text(
                            text = "Anti-Misuse & NCII Safeguards",
                            style = MaterialTheme.typography.titleMedium,
                            color = NeonEmerald
                        )
                    }
                    Text(
                        text = "• Mandatory Consent Gate: The Swap execution button remains hardware-locked until the user explicitly confirms consent from all depicted individuals.\n" +
                            "• Synthetic AI Provenance Badge: Generated outputs include a 'SYNTHETIC AI • OFFLINE ONNX' provenance watermark badge by default to prevent deceptive impersonation or disinformation.\n" +
                            "• Strict Prohibition of NCII: Creating non-consensual intimate imagery, identity fraud, or defamatory media is strictly prohibited.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // 3. Complete Offline Architecture & Model Placement Guide
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Lock,
                            contentDescription = "Offline Setup Guide",
                            tint = ElectricCyan
                        )
                        Text(
                            text = "100% Offline Setup & Testing Guide",
                            style = MaterialTheme.typography.titleMedium,
                            color = ElectricCyan
                        )
                    }
                    Text(
                        text = "Method A — Bundle in Android Studio Project before building APK:\n" +
                            "Copy your ONNX files into:\n" +
                            "  • app/src/main/assets/models/det_10g.onnx\n" +
                            "  • app/src/main/assets/models/w600k_r50.onnx\n" +
                            "  • app/src/main/assets/models/inswapper_128.onnx\n\n" +
                            "Method B — Import on Android Device at Runtime (Air-Gapped):\n" +
                            "1. Enable Airplane Mode on your Android phone.\n" +
                            "2. Open the 'ONNX Models' tab in FaceSwap Studio and tap 'Import .onnx' for det_10g.onnx, w600k_r50.onnx, and inswapper_128.onnx from your device Downloads folder.\n" +
                            "3. Verify all tensor shapes and the extracted [512, 512] emap matrix.\n" +
                            "4. Open 'Swap Studio', pick Source & Target photos from Gallery, select the face to replace, check the consent box, and tap 'Run Offline Face Swap'.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}
