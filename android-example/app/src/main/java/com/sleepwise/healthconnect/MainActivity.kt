package com.sleepwise.healthconnect

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * MainActivity - Example of reading health data from Galaxy Watch via Health Connect
 *
 * This demonstrates:
 * 1. Checking if Health Connect is available
 * 2. Requesting permissions
 * 3. Reading heart rate, sleep sessions, HRV, SpO2
 * 4. Displaying sleep stage breakdown
 */
class MainActivity : AppCompatActivity() {

    private lateinit var healthConnectManager: HealthConnectManager
    private lateinit var statusText: TextView
    private lateinit var dataText: TextView

    // Permission request launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) {
            showStatus("Permissions granted!")
            loadHealthData()
        } else {
            showStatus("Some permissions denied. Cannot read health data.")
        }
    }

    // Health Connect permission launcher
    private val healthPermissionLauncher = registerForActivityResult(
        androidx.health.connect.client.PermissionController.createRequestPermissionResultContract()
    ) { granted ->
        if (granted.containsAll(HealthConnectManager.PERMISSIONS)) {
            showStatus("Health Connect permissions granted!")
            loadHealthData()
        } else {
            showStatus("Health Connect permissions denied.")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Simple UI - in real app, use proper layouts
        val layout = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        statusText = TextView(this).apply {
            textSize = 16f
            text = "Status: Initializing..."
        }
        layout.addView(statusText)

        val requestPermButton = Button(this).apply {
            text = "Request Permissions"
            setOnClickListener { requestHealthPermissions() }
        }
        layout.addView(requestPermButton)

        val loadDataButton = Button(this).apply {
            text = "Load Health Data"
            setOnClickListener { loadHealthData() }
        }
        layout.addView(loadDataButton)

        dataText = TextView(this).apply {
            textSize = 14f
            text = "Health data will appear here..."
        }
        layout.addView(dataText)

        setContentView(layout)

        // Check if Health Connect is available
        if (!HealthConnectManager.isAvailable(this)) {
            showStatus("Health Connect is NOT available on this device!")
            Toast.makeText(this, "Please install Health Connect from Play Store", Toast.LENGTH_LONG).show()
            return
        }

        healthConnectManager = HealthConnectManager(this)
        showStatus("Health Connect is available. Click 'Request Permissions' to start.")
    }

    private fun requestHealthPermissions() {
        healthPermissionLauncher.launch(HealthConnectManager.PERMISSIONS)
    }

    private fun loadHealthData() {
        showStatus("Loading health data...")

        lifecycleScope.launch {
            try {
                // Check permissions first
                if (!healthConnectManager.hasAllPermissions()) {
                    showStatus("Missing permissions. Please grant permissions first.")
                    return@launch
                }

                val output = StringBuilder()

                // ============================================================
                // 1. SLEEP SESSIONS (Last 7 days)
                // ============================================================
                output.append("=== SLEEP SESSIONS (Last 7 days) ===\n\n")

                val sleepSessions = healthConnectManager.readSleepSessions(daysBack = 7)

                if (sleepSessions.isEmpty()) {
                    output.append("No sleep data found.\n")
                    output.append("Make sure Samsung Health is syncing sleep from your Galaxy Watch.\n\n")
                } else {
                    for (session in sleepSessions) {
                        val formatter = DateTimeFormatter.ofPattern("MMM dd, HH:mm")
                            .withZone(ZoneId.systemDefault())

                        output.append("Night: ${formatter.format(session.startTime)}\n")
                        output.append("Duration: ${session.durationMinutes()} minutes\n")

                        // Sleep stage breakdown - THIS IS KEY FOR YOUR ML MODEL!
                        val stageSummary = session.getStageSummary()
                        output.append("Stages:\n")
                        stageSummary.forEach { (stage, minutes) ->
                            output.append("  $stage: $minutes min\n")
                        }
                        output.append("\n")

                        // Show individual stage transitions (for training data)
                        output.append("Stage transitions:\n")
                        for (stage in session.stages.take(10)) {  // First 10 for brevity
                            val time = DateTimeFormatter.ofPattern("HH:mm:ss")
                                .withZone(ZoneId.systemDefault())
                                .format(stage.startTime)
                            output.append("  $time -> ${stage.stage}\n")
                        }
                        if (session.stages.size > 10) {
                            output.append("  ... and ${session.stages.size - 10} more stages\n")
                        }
                        output.append("\n---\n\n")
                    }
                }

                // ============================================================
                // 2. HEART RATE (Last 24 hours)
                // ============================================================
                output.append("=== HEART RATE (Last 24 hours) ===\n\n")

                val heartRates = healthConnectManager.readHeartRate(hoursBack = 24)

                if (heartRates.isEmpty()) {
                    output.append("No heart rate data found.\n\n")
                } else {
                    val avgHR = heartRates.map { it.bpm }.average()
                    val minHR = heartRates.minOf { it.bpm }
                    val maxHR = heartRates.maxOf { it.bpm }

                    output.append("Samples: ${heartRates.size}\n")
                    output.append("Average: ${avgHR.toInt()} bpm\n")
                    output.append("Min: $minHR bpm\n")
                    output.append("Max: $maxHR bpm\n\n")

                    // Show last 5 readings
                    output.append("Latest readings:\n")
                    for (sample in heartRates.takeLast(5)) {
                        val time = DateTimeFormatter.ofPattern("HH:mm:ss")
                            .withZone(ZoneId.systemDefault())
                            .format(sample.timestamp)
                        output.append("  $time: ${sample.bpm} bpm\n")
                    }
                    output.append("\n")
                }

                // ============================================================
                // 3. HRV (Heart Rate Variability)
                // ============================================================
                output.append("=== HRV - RMSSD (Last 24 hours) ===\n\n")

                val hrvSamples = healthConnectManager.readHRV(hoursBack = 24)

                if (hrvSamples.isEmpty()) {
                    output.append("No HRV data found.\n\n")
                } else {
                    val avgHRV = hrvSamples.map { it.rmssd }.average()
                    output.append("Samples: ${hrvSamples.size}\n")
                    output.append("Average RMSSD: ${String.format("%.1f", avgHRV)} ms\n\n")
                }

                // ============================================================
                // 4. SpO2 (Oxygen Saturation)
                // ============================================================
                output.append("=== SpO2 (Last 24 hours) ===\n\n")

                val spo2Samples = healthConnectManager.readOxygenSaturation(hoursBack = 24)

                if (spo2Samples.isEmpty()) {
                    output.append("No SpO2 data found.\n")
                    output.append("(Galaxy Watch may only measure SpO2 during sleep)\n\n")
                } else {
                    val avgSpO2 = spo2Samples.map { it.percentage }.average()
                    output.append("Samples: ${spo2Samples.size}\n")
                    output.append("Average: ${String.format("%.1f", avgSpO2)}%\n\n")
                }

                // Update UI
                runOnUiThread {
                    dataText.text = output.toString()
                    showStatus("Data loaded successfully!")
                }

            } catch (e: Exception) {
                Log.e("MainActivity", "Error loading health data", e)
                runOnUiThread {
                    showStatus("Error: ${e.message}")
                    dataText.text = "Failed to load data.\n\nError: ${e.message}\n\n" +
                            "Make sure:\n" +
                            "1. Health Connect has permissions\n" +
                            "2. Samsung Health is installed and synced\n" +
                            "3. Your Galaxy Watch has recorded data"
                }
            }
        }
    }

    private fun showStatus(message: String) {
        statusText.text = "Status: $message"
        Log.d("MainActivity", message)
    }
}
