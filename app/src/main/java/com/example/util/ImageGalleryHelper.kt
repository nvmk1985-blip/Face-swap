package com.example.util

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import java.io.OutputStream
import kotlin.math.max

object ImageGalleryHelper {

    private const val MAX_IMAGE_DIMENSION = 1600

    /**
     * Memory-safely decodes a gallery image Uri into an ARGB_8888 Bitmap, applying EXIF orientation
     * and capping the maximum dimension at 1600px to avoid OutOfMemoryError on mobile devices.
     */
    fun decodeUriToBitmap(context: Context, uri: Uri): Result<Bitmap> {
        return runCatching {
            val boundsOpts = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            context.contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, boundsOpts)
            } ?: error("Unable to open image stream from Gallery URI.")

            if (boundsOpts.outWidth <= 0 || boundsOpts.outHeight <= 0) {
                error("Unsupported or corrupted image file format.")
            }

            var sampleSize = 1
            val longestSide = max(boundsOpts.outWidth, boundsOpts.outHeight)
            while (longestSide / sampleSize > MAX_IMAGE_DIMENSION) {
                sampleSize *= 2
            }

            val decodeOpts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
                inMutable = false
            }

            val rawBitmap = context.contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, decodeOpts)
            } ?: error("Failed to decode bitmap pixels.")

            val rotationDegrees = readExifRotation(context, uri)
            val oriented = if (rotationDegrees != 0f) {
                val matrix = Matrix().apply { postRotate(rotationDegrees) }
                val rotated = Bitmap.createBitmap(
                    rawBitmap,
                    0,
                    0,
                    rawBitmap.width,
                    rawBitmap.height,
                    matrix,
                    true
                )
                if (rotated !== rawBitmap) rawBitmap.recycle()
                rotated
            } else {
                rawBitmap
            }

            if (oriented.config != Bitmap.Config.ARGB_8888) {
                val argb = oriented.copy(Bitmap.Config.ARGB_8888, false)
                oriented.recycle()
                argb
            } else {
                oriented
            }
        }
    }

    private fun readExifRotation(context: Context, uri: Uri): Float {
        return runCatching {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                val exif = ExifInterface(stream)
                when (
                    exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                ) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                    ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                    ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                    else -> 0f
                }
            } ?: 0f
        }.getOrDefault(0f)
    }

    /**
     * Saves the swapped Bitmap to the Android device's Gallery (`Pictures/FaceSwapStudio`)
     * using MediaStore without requiring internet or legacy storage permissions on Android 10+.
     */
    fun saveBitmapToGallery(context: Context, bitmap: Bitmap): Result<String> {
        return runCatching {
            val fileName = "FaceSwap_${System.currentTimeMillis()}.png"
            val resolver = context.contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(
                        MediaStore.Images.Media.RELATIVE_PATH,
                        "${Environment.DIRECTORY_PICTURES}/FaceSwapStudio"
                    )
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val imageUri = resolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            ) ?: error("Could not create MediaStore entry in Android Gallery.")

            val outputStream: OutputStream = resolver.openOutputStream(imageUri)
                ?: error("Could not open MediaStore output stream.")

            outputStream.use { out ->
                if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) {
                    error("Failed to compress PNG bitmap.")
                }
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentValues.clear()
                contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(imageUri, contentValues, null, null)
            }

            "Pictures/FaceSwapStudio/$fileName"
        }
    }
}
