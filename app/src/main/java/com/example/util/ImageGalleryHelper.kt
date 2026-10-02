package com.example.util

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.provider.DocumentsContract
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileDescriptor
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.math.max

object ImageGalleryHelper {

    private const val MAX_IMAGE_DIMENSION = 1600

    /**
     * Memory-safely decodes a gallery or document image Uri into a software ARGB_8888 Bitmap,
     * applying EXIF orientation and capping the maximum dimension at 1600px.
     *
     * Supports OEM Gallery providers (Vivo, Xiaomi, Samsung, Oppo), Android Photo Picker
     * virtual/typed media URIs, DocumentsProvider URIs, and direct file descriptors.
     */
    fun decodeUriToBitmap(context: Context, uri: Uri): Result<Bitmap> {
        return runCatching {
            // Preserve read permission across coroutine dispatches if the provider supports it
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }

            val candidateUris = buildCandidateUris(context, uri)
            var lastError: Throwable? = null

            for (candidateUri in candidateUris) {
                // Strategy 1: Native ImageDecoder on Android 9+ (handles Photo Picker, HEIC, WebP, EXIF)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val imageDecoderResult = runCatching {
                        decodeWithImageDecoder(context, candidateUri)
                    }
                    if (imageDecoderResult.isSuccess) {
                        return@runCatching imageDecoderResult.getOrThrow()
                    } else {
                        lastError = imageDecoderResult.exceptionOrNull()
                    }
                }

                // Strategy 2: Copy stream once to a temporary cache file using multi-descriptor fallback
                val tempFileResult = runCatching {
                    decodeViaSinglePassTempFile(context, candidateUri)
                }
                if (tempFileResult.isSuccess) {
                    return@runCatching tempFileResult.getOrThrow()
                } else {
                    lastError = tempFileResult.exceptionOrNull()
                }

                // Strategy 3: Direct ParcelFileDescriptor decoding
                val pfdResult = runCatching {
                    decodeViaParcelFileDescriptor(context, candidateUri)
                }
                if (pfdResult.isSuccess) {
                    return@runCatching pfdResult.getOrThrow()
                } else {
                    lastError = pfdResult.exceptionOrNull()
                }
            }

            throw IllegalStateException(
                lastError?.message ?: "Unable to open image stream from Gallery URI ($uri)."
            )
        }
    }

    private fun decodeWithImageDecoder(context: Context, uri: Uri): Bitmap {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
            error("ImageDecoder requires API 28+")
        }
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.isMutableRequired = false
            val w = info.size.width
            val h = info.size.height
            val longest = max(w, h)
            if (longest > MAX_IMAGE_DIMENSION) {
                val scale = MAX_IMAGE_DIMENSION.toFloat() / longest.toFloat()
                decoder.setTargetSize(
                    (w * scale).toInt().coerceAtLeast(1),
                    (h * scale).toInt().coerceAtLeast(1)
                )
            }
        }
        if (bitmap.width <= 0 || bitmap.height <= 0) {
            error("Decoded bitmap has invalid dimensions.")
        }
        return ensureArgb8888(bitmap)
    }

    private fun decodeViaSinglePassTempFile(context: Context, uri: Uri): Bitmap {
        val tempFile = File(context.cacheDir, "picked_photo_${System.nanoTime()}.tmp")
        try {
            val stream = openBestEffortInputStream(context, uri)
                ?: error("Unable to open image stream from Gallery URI.")

            var bytesCopied = 0L
            stream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    bytesCopied = input.copyTo(output, bufferSize = 64 * 1024)
                    output.flush()
                }
            }

            if (bytesCopied <= 0L || !tempFile.exists() || tempFile.length() <= 0L) {
                error("Selected image stream returned 0 bytes.")
            }

            val boundsOpts = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeFile(tempFile.absolutePath, boundsOpts)

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

            val rawBitmap = BitmapFactory.decodeFile(tempFile.absolutePath, decodeOpts)
                ?: error("Failed to decode bitmap pixels from temporary file.")

            val rotationDegrees = readExifRotationFromFile(tempFile)
            val oriented = applyRotationIfNeeded(rawBitmap, rotationDegrees)
            return ensureArgb8888(oriented)
        } finally {
            runCatching { tempFile.delete() }
        }
    }

    private fun decodeViaParcelFileDescriptor(context: Context, uri: Uri): Bitmap {
        val pfd = openBestEffortParcelFileDescriptor(context, uri)
            ?: error("Unable to open ParcelFileDescriptor for Gallery URI.")

        pfd.use { descriptor ->
            val fd: FileDescriptor = descriptor.fileDescriptor
            val boundsOpts = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeFileDescriptor(fd, null, boundsOpts)
            if (boundsOpts.outWidth <= 0 || boundsOpts.outHeight <= 0) {
                error("Unable to decode image bounds from FileDescriptor.")
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

            val rawBitmap = BitmapFactory.decodeFileDescriptor(fd, null, decodeOpts)
                ?: error("Unable to decode bitmap pixels from FileDescriptor.")

            val rotationDegrees = runCatching {
                val exif = ExifInterface(fd)
                exifToDegrees(
                    exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                )
            }.getOrDefault(0f)

            val oriented = applyRotationIfNeeded(rawBitmap, rotationDegrees)
            return ensureArgb8888(oriented)
        }
    }

    private fun openBestEffortInputStream(context: Context, uri: Uri): InputStream? {
        val resolver = context.contentResolver

        // 1. Typed asset file descriptor ("image/*") — required by virtual/transcoded Photo Picker & OEM galleries
        runCatching {
            resolver.openTypedAssetFileDescriptor(uri, "image/*", null)?.createInputStream()
        }.getOrNull()?.let { return it }

        // 2. Standard ContentResolver inputStream
        runCatching {
            resolver.openInputStream(uri)
        }.getOrNull()?.let { return it }

        // 3. AssetFileDescriptor ("r")
        runCatching {
            resolver.openAssetFileDescriptor(uri, "r")?.createInputStream()
        }.getOrNull()?.let { return it }

        // 4. ParcelFileDescriptor AutoCloseInputStream
        runCatching {
            resolver.openFileDescriptor(uri, "r")?.let {
                ParcelFileDescriptor.AutoCloseInputStream(it)
            }
        }.getOrNull()?.let { return it }

        // 5. Wildcard typed asset file descriptor ("*/*")
        runCatching {
            resolver.openTypedAssetFileDescriptor(uri, "*/*", null)?.createInputStream()
        }.getOrNull()?.let { return it }

        // 6. Direct file:// path fallback
        if (uri.scheme.equals("file", ignoreCase = true)) {
            val path = uri.path
            if (!path.isNullOrBlank()) {
                val f = File(path)
                if (f.exists() && f.canRead()) {
                    return runCatching { FileInputStream(f) }.getOrNull()
                }
            }
        }

        return null
    }

    private fun openBestEffortParcelFileDescriptor(
        context: Context,
        uri: Uri
    ): ParcelFileDescriptor? {
        val resolver = context.contentResolver
        runCatching {
            resolver.openTypedAssetFileDescriptor(uri, "image/*", null)?.parcelFileDescriptor
        }.getOrNull()?.let { return it }

        runCatching {
            resolver.openFileDescriptor(uri, "r")
        }.getOrNull()?.let { return it }

        runCatching {
            resolver.openAssetFileDescriptor(uri, "r")?.parcelFileDescriptor
        }.getOrNull()?.let { return it }

        return null
    }

    private fun buildCandidateUris(context: Context, originalUri: Uri): List<Uri> {
        val candidates = LinkedHashSet<Uri>()
        candidates.add(originalUri)

        // Check DocumentsContract URI translations (e.g. image:12345 -> MediaStore EXTERNAL_CONTENT_URI/12345)
        runCatching {
            if (DocumentsContract.isDocumentUri(context, originalUri)) {
                val docId = DocumentsContract.getDocumentId(originalUri)
                if (docId.startsWith("raw:")) {
                    val rawPath = docId.removePrefix("raw:")
                    val rawFile = File(rawPath)
                    if (rawFile.exists()) {
                        candidates.add(Uri.fromFile(rawFile))
                    }
                } else if (docId.contains(":")) {
                    val parts = docId.split(":")
                    val type = parts.getOrNull(0).orEmpty()
                    val idStr = parts.getOrNull(1).orEmpty()
                    val numericId = idStr.toLongOrNull()
                    if (numericId != null && (type.equals("image", ignoreCase = true) ||
                            originalUri.authority?.contains("media") == true)
                    ) {
                        candidates.add(
                            ContentUris.withAppendedId(
                                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                numericId
                            )
                        )
                    }
                    if (type.equals("primary", ignoreCase = true) && idStr.isNotEmpty()) {
                        val extFile = File(Environment.getExternalStorageDirectory(), idStr)
                        if (extFile.exists()) {
                            candidates.add(Uri.fromFile(extFile))
                        }
                    }
                }
            }
        }

        // Check if lastPathSegment contains an embedded media ID (common in Photo Picker / OEM picker URIs)
        runCatching {
            val lastSegment = originalUri.lastPathSegment.orEmpty()
            val numericId = when {
                lastSegment.contains(":") -> lastSegment.substringAfterLast(":").toLongOrNull()
                else -> lastSegment.toLongOrNull()
            }
            if (numericId != null && numericId > 0L) {
                candidates.add(
                    ContentUris.withAppendedId(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                        numericId
                    )
                )
            }
        }

        // Query ContentResolver for _ID or _DATA column if provided by OEM gallery
        runCatching {
            val projection = arrayOf(
                MediaStore.Images.Media._ID,
                MediaStore.MediaColumns.DATA
            )
            context.contentResolver.query(originalUri, projection, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val idIdx = cursor.getColumnIndex(MediaStore.Images.Media._ID)
                    if (idIdx >= 0 && !cursor.isNull(idIdx)) {
                        val mediaId = cursor.getLong(idIdx)
                        if (mediaId > 0L) {
                            candidates.add(
                                ContentUris.withAppendedId(
                                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                    mediaId
                                )
                            )
                        }
                    }
                    val dataIdx = cursor.getColumnIndex(MediaStore.MediaColumns.DATA)
                    if (dataIdx >= 0 && !cursor.isNull(dataIdx)) {
                        val filePath = cursor.getString(dataIdx)
                        if (!filePath.isNullOrBlank()) {
                            val file = File(filePath)
                            if (file.exists() && file.canRead()) {
                                candidates.add(Uri.fromFile(file))
                            }
                        }
                    }
                }
            }
        }

        return candidates.toList()
    }

    private fun readExifRotationFromFile(file: File): Float {
        return runCatching {
            val exif = ExifInterface(file.absolutePath)
            exifToDegrees(
                exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            )
        }.getOrDefault(0f)
    }

    private fun exifToDegrees(exifOrientation: Int): Float {
        return when (exifOrientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
    }

    private fun applyRotationIfNeeded(rawBitmap: Bitmap, rotationDegrees: Float): Bitmap {
        if (rotationDegrees == 0f) return rawBitmap
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
        return rotated
    }

    private fun ensureArgb8888(bitmap: Bitmap): Bitmap {
        return if (bitmap.config != Bitmap.Config.ARGB_8888) {
            val argb = bitmap.copy(Bitmap.Config.ARGB_8888, false)
            bitmap.recycle()
            argb
        } else {
            bitmap
        }
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
