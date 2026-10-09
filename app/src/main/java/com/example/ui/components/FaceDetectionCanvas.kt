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

                val pose3D = face.headPose3D
                val badgeLabel = if (isSelected) {
                    "✓ #${face.index + 1} • 106-Pt 3D (Y:${"%.0f".format(pose3D.yawDeg)}° P:${"%.0f".format(pose3D.pitchDeg)}° R:${"%.0f".format(pose3D.rollDeg)}°)"
                } else {
                    "#${face.index + 1} • 106-Pt"
                }
                val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = if (isSelected) android.graphics.Color.rgb(11, 16, 33) else android.graphics.Color.WHITE
                    textSize = 10.5.dp.toPx()
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

                // Render 106-Point Dense 3D Reconstructed Facial Contours & Landmarks
                val pts106 = face.landmarks106
                if (pts106.size >= 106) {
                    fun drawPolyline106(indices: IntRange, closed: Boolean, lineColor: Color, strokeWidthPx: Float) {
                        val path = androidx.compose.ui.graphics.Path()
                        var first = true
                        for (idx in indices) {
                            val pt = pts106[idx]
                            val cx = offsetX + pt.x * scale
                            val cy = offsetY + pt.y * scale
                            if (first) {
                                path.moveTo(cx, cy)
                                first = false
                            } else {
                                path.lineTo(cx, cy)
                            }
                        }
                        if (closed) path.close()
                        drawPath(
                            path = path,
                            color = lineColor,
                            style = Stroke(width = strokeWidthPx)
                        )
                    }

                    val strokeW = if (isSelected) 1.5.dp.toPx() else 1.0.dp.toPx()
                    // 0..32: 33-point 3D Jawline Contour
                    drawPolyline106(0..32, closed = false, lineColor = NeonEmerald.copy(alpha = if (isSelected) 0.85f else 0.45f), strokeWidthPx = strokeW)
                    // 33..42 & 43..52: Left & Right 10-point Eyebrow Contours
                    drawPolyline106(33..42, closed = true, lineColor = AmberWarning.copy(alpha = if (isSelected) 0.80f else 0.40f), strokeWidthPx = strokeW)
                    drawPolyline106(43..52, closed = true, lineColor = AmberWarning.copy(alpha = if (isSelected) 0.80f else 0.40f), strokeWidthPx = strokeW)
                    // 53..60 & 63..70: Left & Right 8-point Eyelid Contours
                    drawPolyline106(53..60, closed = true, lineColor = ElectricCyan.copy(alpha = if (isSelected) 0.85f else 0.45f), strokeWidthPx = strokeW)
                    drawPolyline106(63..70, closed = true, lineColor = ElectricCyan.copy(alpha = if (isSelected) 0.85f else 0.45f), strokeWidthPx = strokeW)
                    // 73..76 & 77..85: Nose Bridge (4 pts) & Alar Nostril Wings (9 pts)
                    drawPolyline106(73..76, closed = false, lineColor = ElectricCyan.copy(alpha = if (isSelected) 0.80f else 0.40f), strokeWidthPx = strokeW)
                    drawPolyline106(77..85, closed = false, lineColor = ElectricCyan.copy(alpha = if (isSelected) 0.80f else 0.40f), strokeWidthPx = strokeW)
                    // 86..97 & 98..105: Outer Lip Vermilion (12 pts) & Inner Oral Aperture (8 pts)
                    drawPolyline106(86..97, closed = true, lineColor = Color(0xFFFF80AB).copy(alpha = if (isSelected) 0.85f else 0.45f), strokeWidthPx = strokeW)
                    drawPolyline106(98..105, closed = true, lineColor = Color(0xFFFFD54F).copy(alpha = if (isSelected) 0.80f else 0.40f), strokeWidthPx = strokeW)

                    // Draw all 106 dense landmark dots
                    val dotRadius = if (isSelected) 1.5.dp.toPx() else 1.1.dp.toPx()
                    pts106.forEachIndexed { idx, pt ->
                        val px = offsetX + pt.x * scale
                        val py = offsetY + pt.y * scale
                        val dotColor = when (idx) {
                            in 0..32 -> NeonEmerald
                            in 33..52 -> AmberWarning
                            in 53..72 -> ElectricCyan
                            in 73..85 -> Color(0xFF80D8FF)
                            else -> Color(0xFFFF80AB)
                        }
                        drawCircle(
                            color = dotColor,
                            radius = dotRadius,
                            center = Offset(px, py)
                        )
                    }
                }

                // Highlight the 5 primary anchor keypoints (Iris Centers, Nose Tip, Mouth Corners)
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
                        radius = 3.0.dp.toPx(),
                        center = Offset(px, py)
                    )
                    drawCircle(
                        color = kpColor,
                        radius = 2.0.dp.toPx(),
                        center = Offset(px, py)
                    )
                }
            }
        }
    }
}
