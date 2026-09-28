package com.example.data

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "swap_audit_logs")
data class SwapAuditLog(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val timestampMillis: Long = System.currentTimeMillis(),
    val sourceResolution: String,
    val targetResolution: String,
    val swappedFacesCount: Int,
    val pipelineSummary: String,
    val detectionMs: Long,
    val embeddingMs: Long,
    val inswapperMs: Long,
    val blendingMs: Long,
    val totalMs: Long,
    val colorTransferEnabled: Boolean,
    val watermarkEnabled: Boolean,
    val savedGalleryPath: String? = null
)

@Dao
interface SwapAuditDao {
    @Query("SELECT * FROM swap_audit_logs ORDER BY timestampMillis DESC")
    fun observeAllLogs(): Flow<List<SwapAuditLog>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLog(log: SwapAuditLog): Long

    @Query("UPDATE swap_audit_logs SET savedGalleryPath = :galleryPath WHERE id = :id")
    suspend fun updateSavedPath(id: Int, galleryPath: String)

    @Query("DELETE FROM swap_audit_logs WHERE id = :id")
    suspend fun deleteById(id: Int)

    @Query("DELETE FROM swap_audit_logs")
    suspend fun clearAll()
}

@Database(entities = [SwapAuditLog::class], version = 1, exportSchema = false)
abstract class SwapAuditDatabase : RoomDatabase() {
    abstract fun swapAuditDao(): SwapAuditDao

    companion object {
        @Volatile
        private var INSTANCE: SwapAuditDatabase? = null

        fun getInstance(context: Context): SwapAuditDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    SwapAuditDatabase::class.java,
                    "faceswap_studio_audit.db"
                ).fallbackToDestructiveMigration(true).build()
                INSTANCE = instance
                instance
            }
        }
    }
}

class SwapAuditRepository(private val dao: SwapAuditDao) {
    val allLogs: Flow<List<SwapAuditLog>> = dao.observeAllLogs()

    suspend fun insert(log: SwapAuditLog): Int = dao.insertLog(log).toInt()

    suspend fun markSavedToGallery(id: Int, path: String) = dao.updateSavedPath(id, path)

    suspend fun deleteById(id: Int) = dao.deleteById(id)

    suspend fun clearAll() = dao.clearAll()
}
