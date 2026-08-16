# SleepWise - Health Connect Example

This example shows how to read health data (Heart Rate, Sleep, SpO2) from
Samsung Health via Google's Health Connect API.

## Prerequisites

1. **Android Studio** (latest version)
2. **Galaxy Watch 5 Pro** paired with your Android phone
3. **Samsung Health** app installed and syncing data from your watch
4. **Health Connect** app installed (usually pre-installed on Android 14+)

## Setup Steps

### 1. Install Health Connect (if not already installed)
- Android 14+: Pre-installed
- Android 13: Download from Play Store

### 2. Grant Samsung Health permission to share data
- Open Health Connect app
- Go to "App permissions"
- Find "Samsung Health"
- Enable all data types (Heart rate, Sleep, etc.)

### 3. Create new Android project in Android Studio
- File > New > New Project
- Select "Empty Activity"
- Name: SleepWise
- Package: com.sleepwise.app
- Language: Kotlin
- Minimum SDK: API 28 (Android 9.0)

### 4. Copy the code files from this example

### 5. Build and run on your phone

## What This Example Does

1. Checks if Health Connect is available
2. Requests permissions to read health data
3. Reads:
   - Heart Rate (last 24 hours)
   - Sleep Sessions (last 7 days)
   - SpO2/Oxygen Saturation (last 24 hours)
   - Resting Heart Rate
4. Displays the data in the app

## Files Structure

```
app/src/main/
├── AndroidManifest.xml          # Permissions declaration
├── java/com/sleepwise/healthconnect/
│   ├── HealthConnectManager.kt  # Main Health Connect logic
│   └── MainActivity.kt          # Example UI
└── build.gradle.kts             # Dependencies
```

## Important Notes

- Health Connect requires Android 9+ (API 28)
- Data is only available if Samsung Health has synced it
- First run will prompt user for permissions
- Sleep data includes stages (Awake, Light, Deep, REM) from Galaxy Watch!
