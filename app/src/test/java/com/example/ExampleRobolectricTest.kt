package com.example

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import androidx.test.core.app.ApplicationProvider
import com.example.onnx.DetectedFace
import com.example.onnx.FaceAlignment
import com.example.onnx.GhostHeadReplacementEngine
import com.example.onnx.HeadSegmentationAndInpainting
import com.example.onnx.OnnxProtobufInspector
import com.example.onnx.ScrfdFaceDetector
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class ExampleRobolectricTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    private val ortEnv: OrtEnvironment?
        get() = runCatching { OrtEnvironment.getEnvironment() }.getOrNull()

    private fun createSyntheticPortrait(
        width: Int,
        height: Int,
        bgColor: Int,
        skinColor: Int,
        hairColor: Int,
        hairRadiusScale: Float = 1.0f,
        rollDegrees: Float = 0f,
        yawOffsetPx: Float = 0f,
        centerX: Float = width * 0.5f,
        centerY: Float = height * 0.5f,
        eyeDist: Float = width * 0.16f
    ): Pair<Bitmap, DetectedFace> {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height) { bgColor }

        val rad = Math.toRadians(rollDegrees.toDouble())
        val cosA = kotlin.math.cos(rad).toFloat()
        val sinA = kotlin.math.sin(rad).toFloat()

        val hairRx = eyeDist * 1.65f * hairRadiusScale
        val hairRy = eyeDist * 1.75f * hairRadiusScale
        val faceRx = eyeDist * 1.05f
        val faceRy = eyeDist * 1.22f

        for (y in 0 until height) {
            for (x in 0 until width) {
                val dx = x - centerX
                val dy = y - centerY
                // Inverse rotate into head-local coordinates
                val lx = dx * cosA + dy * sinA
                val ly = -dx * sinA + dy * cosA

                // Hair dome check
                val hNormX = lx / hairRx
                val hNormY = (ly + hairRy * 0.30f) / hairRy
                if (hNormX * hNormX + hNormY * hNormY <= 1.0f) {
                    pixels[y * width + x] = hairColor
                }

                // Neck column check
                if (kotlin.math.abs(lx) <= eyeDist * 0.55f &&
                    ly >= eyeDist * 0.85f &&
                    ly <= eyeDist * 1.85f
                ) {
                    pixels[y * width + x] = skinColor
                }

                // Face/skull oval check
                val fNormX = lx / faceRx
                val fNormY = (ly - eyeDist * 0.12f) / faceRy
                if (fNormX * fNormX + fNormY * fNormY <= 1.0f) {
                    // Add subtle spatial shading so variance > 0
                    val shade = ((lx + ly) * 0.15f).toInt()
                    val r = (((skinColor ushr 16) and 0xFF) + shade).coerceIn(0, 255)
                    val g = (((skinColor ushr 8) and 0xFF) + shade).coerceIn(0, 255)
                    val b = ((skinColor and 0xFF) + shade).coerceIn(0, 255)
                    pixels[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)

        fun rot(dx: Float, dy: Float): PointF {
            return PointF(
                centerX + dx * cosA - dy * sinA,
                centerY + dx * sinA + dy * cosA
            )
        }

        val landmarks = listOf(
            rot(-eyeDist * 0.5f, 0f),
            rot(eyeDist * 0.5f, 0f),
            rot(yawOffsetPx, eyeDist * 0.55f),
            rot(-eyeDist * 0.42f, eyeDist * 1.05f),
            rot(eyeDist * 0.42f, eyeDist * 1.05f)
        )
        val box = RectF(
            centerX - eyeDist * 1.2f,
            centerY - eyeDist * 1.2f,
            centerX + eyeDist * 1.2f,
            centerY + eyeDist * 1.4f
        )
        val face = DetectedFace(
            index = 0,
            boundingBox = box,
            score = 0.98f,
            landmarks5 = landmarks,
            detectorSource = "SyntheticTest"
        )
        return bmp to face
    }

    @Test
    fun `test01 front-facing source and front-facing target head replacement`() {
        val (srcBmp, srcFace) = createSyntheticPortrait(
            width = 320,
            height = 320,
            bgColor = Color.rgb(20, 30, 50),
            skinColor = Color.rgb(225, 180, 155),
            hairColor = Color.rgb(40, 25, 15)
        )
        val (tgtBmp, tgtFace) = createSyntheticPortrait(
            width = 320,
            height = 320,
            bgColor = Color.rgb(210, 220, 235),
            skinColor = Color.rgb(210, 165, 140),
            hairColor = Color.rgb(180, 150, 70)
        )

        val stagesVisited = mutableListOf<Int>()
        val result = GhostHeadReplacementEngine.executeFullHeadReplacement(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = true,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = { stagesVisited.add(it.stepIndex) }
        )

        assertEquals(320, result.outputBitmap.width)
        assertEquals(320, result.outputBitmap.height)
        assertEquals(1, result.swappedFacesCount)
        assertTrue(stagesVisited.containsAll(listOf(1, 2, 3, 4, 5, 6)))
    }

    @Test
    fun `test02 different lighting conditions adapt cleanly`() {
        val (srcBmp, srcFace) = createSyntheticPortrait(
            width = 280,
            height = 280,
            bgColor = Color.rgb(15, 15, 20),
            skinColor = Color.rgb(120, 90, 75), // Low-key dim lighting
            hairColor = Color.rgb(20, 20, 25)
        )
        val (tgtBmp, tgtFace) = createSyntheticPortrait(
            width = 280,
            height = 280,
            bgColor = Color.rgb(245, 245, 250),
            skinColor = Color.rgb(245, 210, 190), // High-key bright lighting
            hairColor = Color.rgb(90, 60, 40)
        )

        val result = GhostHeadReplacementEngine.executeFullHeadReplacement(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )

        val centerPixel = result.outputBitmap.getPixel(140, 150)
        val r = (centerPixel ushr 16) and 0xFF
        // Skin tone should be brightened toward target illumination (> 140)
        assertTrue("Expected adapted skin luminance > 140, got $r", r > 140)
    }

    @Test
    fun `test03 different skin tones harmonize while preserving hair`() {
        val (srcBmp, srcFace) = createSyntheticPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(30, 40, 55),
            skinColor = Color.rgb(240, 200, 180),
            hairColor = Color.rgb(25, 20, 20)
        )
        val (tgtBmp, tgtFace) = createSyntheticPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(60, 70, 80),
            skinColor = Color.rgb(135, 90, 65),
            hairColor = Color.rgb(30, 25, 25)
        )

        val result = GhostHeadReplacementEngine.executeFullHeadReplacement(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )
        assertNotNull(result.outputBitmap)
        // Exterior background corner (5, 5) must remain intact target background
        assertEquals(tgtBmp.getPixel(5, 5), result.outputBitmap.getPixel(5, 5))
    }

    @Test
    fun `test04 different head angles roll and yaw alignment`() {
        val (srcBmp, srcFace) = createSyntheticPortrait(
            width = 300,
            height = 300,
            bgColor = Color.rgb(25, 30, 40),
            skinColor = Color.rgb(220, 175, 150),
            hairColor = Color.rgb(50, 35, 25),
            rollDegrees = -18f,
            yawOffsetPx = -6f
        )
        val (tgtBmp, tgtFace) = createSyntheticPortrait(
            width = 300,
            height = 300,
            bgColor = Color.rgb(200, 210, 220),
            skinColor = Color.rgb(215, 170, 145),
            hairColor = Color.rgb(80, 50, 30),
            rollDegrees = 22f,
            yawOffsetPx = 8f
        )

        val srcPose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(srcFace, 300, 300)
        val tgtPose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(tgtFace, 300, 300)
        assertEquals(-18f, srcPose.rollDegrees, 1.5f)
        assertEquals(22f, tgtPose.rollDegrees, 1.5f)
        assertTrue(srcPose.yawRatio != tgtPose.yawRatio)

        val result = GhostHeadReplacementEngine.executeFullHeadReplacement(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )
        assertEquals(300, result.outputBitmap.width)
    }

    @Test
    fun `test05 different hairstyles disocclusion background gap inpainting`() {
        // Source has compact short hair (scale 0.85), Target has wide voluminous hair (scale 1.35)
        val (srcBmp, srcFace) = createSyntheticPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(30, 30, 30),
            skinColor = Color.rgb(220, 180, 150),
            hairColor = Color.rgb(40, 30, 20),
            hairRadiusScale = 0.85f
        )
        val (tgtBmp, tgtFace) = createSyntheticPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(230, 240, 250), // Bright sky background
            skinColor = Color.rgb(220, 180, 150),
            hairColor = Color.rgb(200, 20, 20), // Bright red wide target hair to be erased in gap
            hairRadiusScale = 1.35f
        )

        val srcMasks = HeadSegmentationAndInpainting.computeChromaticGeodesicHeadMatte(srcBmp)
        val tgtMasks = HeadSegmentationAndInpainting.computeChromaticGeodesicHeadMatte(tgtBmp)
        val inpainted = HeadSegmentationAndInpainting.inpaintTargetBackgroundGap(
            ortEnv = ortEnv,
            targetAlignedCrop = tgtBmp,
            targetHeadAlpha = tgtMasks.fullHeadAlpha,
            sourceHeadAlpha = srcMasks.fullHeadAlpha,
            lamaModelFile = null
        )
        assertEquals(256, inpainted.width)
        assertEquals(256, inpainted.height)
    }

    @Test
    fun `test06 different non-square image resolutions`() {
        val (srcBmp, srcFace) = createSyntheticPortrait(
            width = 240,
            height = 360,
            bgColor = Color.rgb(40, 50, 60),
            skinColor = Color.rgb(220, 180, 155),
            hairColor = Color.rgb(30, 20, 15)
        )
        val (tgtBmp, tgtFace) = createSyntheticPortrait(
            width = 480,
            height = 300,
            bgColor = Color.rgb(180, 195, 210),
            skinColor = Color.rgb(210, 170, 145),
            hairColor = Color.rgb(70, 45, 25)
        )

        val result = GhostHeadReplacementEngine.executeFullHeadReplacement(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = true,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )
        assertEquals(480, result.outputBitmap.width)
        assertEquals(300, result.outputBitmap.height)
    }

    @Test
    fun `test07 no detectable face handled gracefully without crash`() {
        val blankBmp = Bitmap.createBitmap(240, 240, Bitmap.Config.ARGB_8888)
        blankBmp.eraseColor(Color.DKGRAY)

        val detected = ScrfdFaceDetector.detectFacesAndroidPreviewFallback(blankBmp)
        assertTrue("Blank image must return empty face list gracefully", detected.isEmpty())
    }

    @Test
    fun `test08 multiple faces in target image sequential head replacement`() {
        val (srcBmp, srcFace) = createSyntheticPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(20, 25, 35),
            skinColor = Color.rgb(230, 185, 160),
            hairColor = Color.rgb(35, 25, 20)
        )
        val (tgtBmp, face1) = createSyntheticPortrait(
            width = 440,
            height = 260,
            bgColor = Color.rgb(200, 210, 225),
            skinColor = Color.rgb(210, 170, 145),
            hairColor = Color.rgb(80, 55, 35),
            centerX = 130f,
            centerY = 130f,
            eyeDist = 42f
        )
        val (_, face2Raw) = createSyntheticPortrait(
            width = 440,
            height = 260,
            bgColor = Color.rgb(200, 210, 225),
            skinColor = Color.rgb(195, 155, 130),
            hairColor = Color.rgb(45, 30, 20),
            centerX = 310f,
            centerY = 130f,
            eyeDist = 42f
        )
        val face2 = face2Raw.copy(index = 1)

        val result = GhostHeadReplacementEngine.executeFullHeadReplacement(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(face1, face2),
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )
        assertEquals(2, result.swappedFacesCount)
    }

    @Test
    fun `test09 low-memory android device tiled processing and cleanup`() {
        val (srcBmp, srcFace) = createSyntheticPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(30, 35, 45),
            skinColor = Color.rgb(225, 180, 150),
            hairColor = Color.rgb(40, 30, 20)
        )
        val enhanced = GhostHeadReplacementEngine.applyTiledDetailEnhancement(srcBmp)
        assertEquals(256, enhanced.width)
        assertEquals(256, enhanced.height)
        enhanced.recycle()

        val result = GhostHeadReplacementEngine.executeFullHeadReplacement(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = srcBmp,
            targetFacesToReplace = listOf(srcFace),
            enableColorTransfer = false,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )
        assertEquals(256, result.outputBitmap.width)
    }

    @Test
    fun `test10 airplane-mode offline operation and local model directory layout`() {
        // 1. Verify zero INTERNET permission is requested in AndroidManifest
        val pkgInfo = context.packageManager.getPackageInfo(
            context.packageName,
            PackageManager.GET_PERMISSIONS
        )
        val requestedPerms = pkgInfo.requestedPermissions?.toList() ?: emptyList()
        assertFalse(
            "App must not request android.permission.INTERNET for strict offline operation",
            requestedPerms.contains(android.Manifest.permission.INTERNET)
        )

        // 2. Verify all 6 structured model subdirectories are created locally
        val root = OnnxProtobufInspector.getModelsRootDir(context)
        val expectedSubdirs = listOf(
            "face_swap",
            "head_swap",
            "segmentation",
            "matting",
            "inpainting",
            "enhancement"
        )
        for (sub in expectedSubdirs) {
            val dir = File(root, sub)
            assertTrue("Expected models/$sub directory to exist", dir.exists() && dir.isDirectory)
        }

        // 3. Verify GHOST 2.0 Feasibility Matrix has all 7 reference components documented
        assertEquals(7, OnnxProtobufInspector.GHOST_2_FEASIBILITY_MATRIX.size)
    }
}
