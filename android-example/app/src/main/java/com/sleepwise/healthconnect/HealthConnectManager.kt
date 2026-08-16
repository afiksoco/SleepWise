package com.sleepwise.healthconnect

import android.content.Context
import android.util.Log
import androidx.health.connect.client.HealthConnectClient
import androidx.health.connect.client.permission.HealthPermission
import androidx.health.connect.client.records.*
import androidx.health.connect.client.request.ReadRecordsRequest
import androidx.health.connect.client.time.TimeRangeFilter
import java.time.Instant
import java.time.ZonedDateTime
import java.time.temporal.ChronoUnit

/**
 * HealthConnectManager - Reads health data from Samsung Health via Health Connect API
 *
 * This is the core class for SleepWise that handles:
 * - Heart Rate data
 * - Sleep sessions with stages (Light, Deep, REM, Awake)
 * - Heart Rate Variability (HRV)
 * - Oxygen Saturation (SpO2)
 */
class HealthConnectManager(private val context: Context) {

    private val healthConnectClient by lazy { HealthConnectClient.getOrCreate(context) }

    companion object {
        private const val TAG = "HealthConnectManager"

        // All permissions we need for sleep monitoring
        val PERMISSIONS = setOf(
            HealthPermission.getReadPermission(HeartRateRecord::class),
            HealthPermission.getReadPermission(SleepSessionRecord::class),
            HealthPermission.getReadPermission(OxygenSaturationRecord::class),
            HealthPermission.getReadPermission(RestingHeartRateRecord::class),
            HealthPermission.getReadPermission(HeartRateVariabilityRmssdRecord::class),
            HealthPermission.getReadPermission(RespiratoryRateRecord::class),
            HealthPermission.getReadPermission(StepsRecord::class),
        )

        /**
         * Check if Health Connect is available on this device
         */
        fun isAvailable(context: Context): Boolean {
            val status = HealthConnectClient.getSdkStatus(context)
            return status == HealthConnectClient.SDK_AVAILABLE
        }
    }

    // ========================================================================
    // HEART RATE
    // ========================================================================

    /**
     * Read heart rate samples from the last N hours
     * Returns list of (timestamp, bpm) pairs
     */
    suspend fun readHeartRate(hoursBack: Long = 24): List<HeartRateSample> {
        val endTime = Instant.now()
        val startTime = endTime.minus(hoursBack, ChronoUnit.HOURS)

        val request = ReadRecordsRequest(
            recordType = HeartRateRecord::class,
            timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
        )

        val response = healthConnectClient.readRecords(request)

        val samples = mutableListOf<HeartRateSample>()
        for (record in response.records) {
            for (sample in record.samples) {
                samples.add(
                    HeartRateSample(
                        timestamp = sample.time,
                        bpm = sample.beatsPerMinute
                    )
                )
            }
        }

        Log.d(TAG, "Read ${samples.size} heart rate samples")
        return samples.sortedBy { it.timestamp }
    }

    // ========================================================================
    // SLEEP SESSIONS (Most important for SleepWise!)
    // ========================================================================

    /**
     * Read sleep sessions from the last N days
     * Includes sleep stages: AWAKE, LIGHT, DEEP, REM, SLEEPING, OUT_OF_BED
     */
    suspend fun readSleepSessions(daysBack: Long = 7): List<SleepSessionData> {
        val endTime = Instant.now()
        val startTime = endTime.minus(daysBack, ChronoUnit.DAYS)

        val request = ReadRecordsRequest(
            recordType = SleepSessionRecord::class,
            timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
        )

        val response = healthConnectClient.readRecords(request)

        val sessions = response.records.map { record ->
            SleepSessionData(
                startTime = record.startTime,
                endTime = record.endTime,
                title = record.title,
                notes = record.notes,
                stages = record.stages.map { stage ->
                    SleepStage(
                        startTime = stage.startTime,
                        endTime = stage.endTime,
                        stage = mapSleepStageType(stage.stage)
                    )
                }
            )
        }

        Log.d(TAG, "Read ${sessions.size} sleep sessions")
        return sessions.sortedBy { it.startTime }
    }

    private fun mapSleepStageType(stageType: Int): String {
        return when (stageType) {
            SleepSessionRecord.STAGE_TYPE_AWAKE -> "AWAKE"
            SleepSessionRecord.STAGE_TYPE_SLEEPING -> "SLEEPING"
            SleepSessionRecord.STAGE_TYPE_LIGHT -> "LIGHT"
            SleepSessionRecord.STAGE_TYPE_DEEP -> "DEEP"
            SleepSessionRecord.STAGE_TYPE_REM -> "REM"
            SleepSessionRecord.STAGE_TYPE_OUT_OF_BED -> "OUT_OF_BED"
            SleepSessionRecord.STAGE_TYPE_UNKNOWN -> "UNKNOWN"
            else -> "UNKNOWN"
        }
    }

    // ========================================================================
    // HEART RATE VARIABILITY (HRV) - Key for sleep stage detection
    // ========================================================================

    /**
     * Read HRV (RMSSD) samples - important for detecting sleep stages
     */
    suspend fun readHRV(hoursBack: Long = 24): List<HRVSample> {
        val endTime = Instant.now()
        val startTime = endTime.minus(hoursBack, ChronoUnit.HOURS)

        val request = ReadRecordsRequest(
            recordType = HeartRateVariabilityRmssdRecord::class,
            timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
        )

        val response = healthConnectClient.readRecords(request)

        val samples = response.records.map { record ->
            HRVSample(
                timestamp = record.time,
                rmssd = record.heartRateVariabilityMillis
            )
        }

        Log.d(TAG, "Read ${samples.size} HRV samples")
        return samples.sortedBy { it.timestamp }
    }

    // ========================================================================
    // OXYGEN SATURATION (SpO2)
    // ========================================================================

    /**
     * Read SpO2 samples
     */
    suspend fun readOxygenSaturation(hoursBack: Long = 24): List<SpO2Sample> {
        val endTime = Instant.now()
        val startTime = endTime.minus(hoursBack, ChronoUnit.HOURS)

        val request = ReadRecordsRequest(
            recordType = OxygenSaturationRecord::class,
            timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
        )

        val response = healthConnectClient.readRecords(request)

        val samples = response.records.map { record ->
            SpO2Sample(
                timestamp = record.time,
                percentage = record.percentage.value
            )
        }

        Log.d(TAG, "Read ${samples.size} SpO2 samples")
        return samples.sortedBy { it.timestamp }
    }

    // ========================================================================
    // RESTING HEART RATE
    // ========================================================================

    /**
     * Read resting heart rate
     */
    suspend fun readRestingHeartRate(daysBack: Long = 7): List<RestingHRSample> {
        val endTime = Instant.now()
        val startTime = endTime.minus(daysBack, ChronoUnit.DAYS)

        val request = ReadRecordsRequest(
            recordType = RestingHeartRateRecord::class,
            timeRangeFilter = TimeRangeFilter.between(startTime, endTime)
        )

        val response = healthConnectClient.readRecords(request)

        val samples = response.records.map { record ->
            RestingHRSample(
                timestamp = record.time,
                bpm = record.beatsPerMinute
            )
        }

        Log.d(TAG, "Read ${samples.size} resting HR samples")
        return samples.sortedBy { it.timestamp }
    }

    // ========================================================================
    // HELPER: Check if we have all permissions
    // ========================================================================

    suspend fun hasAllPermissions(): Boolean {
        val granted = healthConnectClient.permissionController.getGrantedPermissions()
        return PERMISSIONS.all { it in granted }
    }
}

// ========================================================================
// DATA CLASSES
// ========================================================================

data class HeartRateSample(
    val timestamp: Instant,
    val bpm: Long
)

data class SleepSessionData(
    val startTime: Instant,
    val endTime: Instant,
    val title: String?,
    val notes: String?,
    val stages: List<SleepStage>
) {
    fun durationMinutes(): Long {
        return ChronoUnit.MINUTES.between(startTime, endTime)
    }

    fun getStageSummary(): Map<String, Long> {
        return stages.groupBy { it.stage }
            .mapValues { (_, stages) ->
                stages.sumOf { ChronoUnit.MINUTES.between(it.startTime, it.endTime) }
            }
    }
}

data class SleepStage(
    val startTime: Instant,
    val endTime: Instant,
    val stage: String  // AWAKE, LIGHT, DEEP, REM, SLEEPING, OUT_OF_BED, UNKNOWN
)

data class HRVSample(
    val timestamp: Instant,
    val rmssd: Double  // milliseconds
)

data class SpO2Sample(
    val timestamp: Instant,
    val percentage: Double
)

data class RestingHRSample(
    val timestamp: Instant,
    val bpm: Long
)
