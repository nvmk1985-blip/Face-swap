package com.example

import com.example.onnx.ArcFaceRecognizer
import com.example.onnx.FaceBlender
import kotlin.math.abs
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ExampleUnitTest {
    @Test
    fun l2Normalize_producesUnitLength512Vector() {
        val raw = FloatArray(512) { idx -> (idx + 1).toFloat() * 0.1f }
        val normed = ArcFaceRecognizer.l2Normalize(raw)
        var sumSq = 0.0
        for (v in normed) sumSq += (v * v).toDouble()
        assertEquals(1.0, sqrt(sumSq), 1e-4)
    }

    @Test
    fun featheredFaceMask128_hasZeroBordersAndUnitCenter() {
        val mask = FaceBlender.createFeatheredFaceMask128()
        assertEquals(128 * 128, mask.size)
        // Center of face (64, 68) must be 1.0f
        assertEquals(1.0f, mask[68 * 128 + 64], 1e-4f)
        // Outer corners/edges must be 0.0f so no box seam appears
        assertEquals(0.0f, mask[0], 1e-5f)
        assertEquals(0.0f, mask[127], 1e-5f)
        assertEquals(0.0f, mask[127 * 128 + 127], 1e-5f)
        assertTrue(abs(mask[40 * 128 + 40]) > 0f)
    }
}
