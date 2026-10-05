package com.example.ui.components

import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.RectF
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.example.onnx.DetectedFace
import com.example.onnx.HeadSegmentationAndInpainting
import com.example.ui.theme.AmberWarning
import com.example.ui.theme.ElectricCyan
import com.example.ui.theme.NeonEmerald
import com.example.ui.theme.ObsidianBg
import com.example.ui.theme.RoyalViolet
import kotlin.math.hypot
import kotlin.math.min

/**
 * Renders a Bitmap with interactive face bounding boxes, numbered person badges (#1, #2, #3...),
 * 5-point SCRFD keypoints, and (when `showCranialHeadBounds = true` in Mode 2 Head Replacement)
 * the expanded cranial/hair/skull/neck bounding volume.
 */
@Composable
fun FaceDetectionCanvas(
    bitmap: Bitmap,
    faces: List<DetectedFace>,
    selectedFaceIndex: Int,
    highlightAllFaces: Boolean = false,
    showCranialHeadBounds: Boolean = false,
    onFaceTapped: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(ObsidianBg)
    ) {
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(faces, bitmap) {
                    detectTapGestures { tapOffset ->
                        val canvasW = size.width.toFloat()
                        val canvasH = size.height.toFloat()
                        if (canvasW <= 0f || canvasH <= 0f) return@detectTapGestures

                        val scale = min(canvasW / bitmap.width, canvasH / bitmap.height)
                        val drawnW = bitmap.width * scale
                        val drawnH = bitmap.height * scale
                        val offsetX = (canvasW - drawnW) * 0.5f
                        val offsetY = (canvasH - drawnH) * 0.5f

                        val imgX = (tapOffset.x - offsetX) / scale
                        val imgY = (tapOffset.y - offsetY) / scale

                        val directFace = faces.firstOrNull { face ->
                            val pose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(
                                face,
                                bitmap.width,
                                bitmap.height
                            )
                            val padX = face.boundingBox.width() * 0.18f
                            val padY = face.boundingBox.height() * 0.18f
                            val paddedBox = RectF(
                                face.boundingBox.left - padX,
                                face.boundingBox.top - padY,
                                face.boundingBox.right + padX,
                                face.boundingBox.bottom + padY
                            )
                            paddedBox.contains(imgX, imgY) ||
                                (showCranialHeadBounds && pose.cranialBox.contains(imgX, imgY))
                        }
                        val chosenFace = directFace ?: faces.minByOrNull { face ->
                            hypot(
                                (face.boundingBox.centerX() - imgX).toDouble(),
                                (face.boundingBox.centerY() - imgY).toDouble()
                            )
                        }?.takeIf { face ->
                            val maxDist = maxOf(face.boundingBox.width(), face.boundingBox.height()) * 1.15f
                            hypot(
                                (face.boundingBox.centerX() - imgX).toDouble(),
                                (face.boundingBox.centerY() - imgY).toDouble()
                            ) <= maxDist
                        }

                        if (chosenFace != null) {
                            onFaceTapped(chosenFace.index)
                        }
                    }
                }
        ) {
            val canvasW = size.width
            val canvasH = size.height
            val scale = min(canvasW / bitmap.width, canvasH / bitmap.height)
            val drawnW = bitmap.width * scale
            val drawnH = bitmap.height * scale
            val offsetX = (canvasW - drawnW) * 0.5f
            val offsetY = (canvasH - drawnH) * 0.5f

            drawImage(
                image = bitmap.asImageBitmap(),
                dstOffset = IntOffset(offsetX.toInt(), offsetY.toInt()),
                dstSize = IntSize(drawnW.toInt(), drawnH.toInt())
            )

            faces.forEach { face ->
                val isSelected = highlightAllFaces || (face.index == selectedFaceIndex)
                val boxColor = if (isSelected) ElectricCyan else RoyalViolet.copy(alpha = 0.85f)
                val strokePx = if (isSelected) 3.dp.toPx() else 1.8.dp.toPx()

                if (showCranialHeadBounds) {
                    val headPose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(
                        face,
                        bitmap.width,
                        bitmap.height
                    )
                    val cLeft = offsetX + headPose.cranialBox.left * scale
                    val cTop = offsetY + headPose.cranialBox.top * scale
                    val cW = headPose.cranialBox.width() * scale
                    val cH = headPose.cranialBox.height() * scale

                    drawRoundRect(
                        color = if (isSelected) NeonEmerald else AmberWarning.copy(alpha = 0.7f),
                        topLeft = Offset(cLeft, cTop),
                        size = Size(cW, cH),
                        cornerRadius = CornerRadius(14.dp.toPx(), 14.dp.toPx()),
                        style = Stroke(
                            width = 2.dp.toPx(),
                            pathEffect = PathEffect.dashPathEffect(floatArrayOf(12f, 8f), 0f)
                        )
                    )
                }

                val left = offsetX + face.boundingBox.left * scale
                val top = offsetY + face.boundingBox.top * scale
                val width = face.boundingBox.width() * scale
                val height = face.boundingBox.height() * scale

                if (isSelected) {
                    drawRoundRect(
                        color = ElectricCyan.copy(alpha = 0.16f),
                        topLeft = Offset(left, top),
                        size = Size(width, height),
                        cornerRadius = CornerRadius(8.dp.toPx(), 8.dp.toPx())
                    )
                }

                drawRoundRect(
                    color = boxColor,
                    topLeft = Offset(left, top),
                    size = Size(width, height),
                    cornerRadius = CornerRadius(8.dp.toPx(), 8.dp.toPx()),
                    style = Stroke(width = strokePx)
                )

                // Draw numbered person badge (#1, #2, #3...) so user can easily identify each person in multi-person photos
                val badgeLabel = if (isSelected) "✓ #${face.index + 1}" else "#${face.index + 1}"
                val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = if (isSelected) android.graphics.Color.rgb(11, 16, 33) else android.graphics.Color.WHITE
                    textSize = 11.dp.toPx()
                    isFakeBoldText = true
                }
                val textW = textPaint.measureText(badgeLabel)
                val badgePadH = 6.dp.toPx()
                val badgeH = 18.dp.toPx()
                val badgeW = textW + badgePadH * 2f
                val badgeTop = (top - badgeH - 2.dp.toPx()).coerceAtLeast(offsetY + 2.dp.toPx())
                val badgeLeft = left.coerceIn(offsetX + 2.dp.toPx(), (offsetX + drawnW - badgeW - 2.dp.toPx()).coerceAtLeast(offsetX + 2.dp.toPx()))

                drawRoundRect(
                    color = if (isSelected) NeonEmerald else RoyalViolet.copy(alpha = 0.92f),
                    topLeft = Offset(badgeLeft, badgeTop),
                    size = Size(badgeW, badgeH),
                    cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx())
                )
                drawContext.canvas.nativeCanvas.drawText(
                    badgeLabel,
                    badgeLeft + badgePadH,
                    badgeTop + badgeH * 0.74f,
                    textPaint
                )

                face.landmarks5.forEachIndexed { ptIdx, pt ->
                    val px = offsetX + pt.x * scale
                    val py = offsetY + pt.y * scale
                    val kpColor = when (ptIdx) {
                        0, 1 -> NeonEmerald
                        2 -> ElectricCyan
                        else -> Color(0xFFFFD54F)
                    }
                    drawCircle(
                        color = ObsidianBg,
                        radius = 4.5.dp.toPx(),
                        center = Offset(px, py)
                    )
                    drawCircle(
                        color = kpColor,
                        radius = 3.dp.toPx(),
                        center = Offset(px, py)
                    )
                }
            }
        }
    }
}
