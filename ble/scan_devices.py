"""
BLE Device Scanner - Find your smartwatch and discover its services
"""
import asyncio
from bleak import BleakScanner, BleakClient

# Common health-related BLE service UUIDs
KNOWN_SERVICES = {
    "0000180d-0000-1000-8000-00805f9b34fb": "Heart Rate Service",
    "0000180f-0000-1000-8000-00805f9b34fb": "Battery Service",
    "00001822-0000-1000-8000-00805f9b34fb": "Pulse Oximeter Service",
    "0000181c-0000-1000-8000-00805f9b34fb": "User Data Service",
    "0000181a-0000-1000-8000-00805f9b34fb": "Environmental Sensing",
    "00001816-0000-1000-8000-00805f9b34fb": "Cycling Speed and Cadence",
    "00001814-0000-1000-8000-00805f9b34fb": "Running Speed and Cadence",
    "0000180a-0000-1000-8000-00805f9b34fb": "Device Information",
    "00001800-0000-1000-8000-00805f9b34fb": "Generic Access",
    "00001801-0000-1000-8000-00805f9b34fb": "Generic Attribute",
}

KNOWN_CHARACTERISTICS = {
    "00002a37-0000-1000-8000-00805f9b34fb": "Heart Rate Measurement",
    "00002a38-0000-1000-8000-00805f9b34fb": "Body Sensor Location",
    "00002a39-0000-1000-8000-00805f9b34fb": "Heart Rate Control Point",
    "00002a19-0000-1000-8000-00805f9b34fb": "Battery Level",
    "00002a5e-0000-1000-8000-00805f9b34fb": "PLX Spot-Check Measurement",
    "00002a5f-0000-1000-8000-00805f9b34fb": "PLX Continuous Measurement",
}


async def scan_for_devices(duration=10):
    """Scan for nearby BLE devices"""
    print(f"\n🔍 Scanning for BLE devices ({duration} seconds)...")
    print("   Make sure your watch is nearby and Bluetooth is on!\n")

    devices = await BleakScanner.discover(timeout=duration)

    if not devices:
        print("❌ No devices found. Make sure Bluetooth is enabled.")
        return []

    # Sort by signal strength
    devices_sorted = sorted(devices, key=lambda d: d.rssi, reverse=True)

    print(f"📱 Found {len(devices_sorted)} devices:\n")
    print(f"{'#':<4} {'Name':<30} {'Address':<20} {'RSSI':<8}")
    print("-" * 65)

    watch_keywords = ['watch', 'galaxy', 'samsung', 'pixel', 'wear', 'garmin', 'fitbit', 'amazfit']

    for i, device in enumerate(devices_sorted):
        name = device.name or "Unknown"
        # Highlight potential watches
        is_watch = any(kw in name.lower() for kw in watch_keywords)
        prefix = "⌚" if is_watch else "  "
        print(f"{prefix}{i:<3} {name:<30} {device.address:<20} {device.rssi} dBm")

    return devices_sorted


async def explore_device(address):
    """Connect to a device and explore its services"""
    print(f"\n🔗 Connecting to {address}...")

    try:
        async with BleakClient(address, timeout=20.0) as client:
            if not client.is_connected:
                print("❌ Failed to connect")
                return

            print(f"✅ Connected!\n")
            print("=" * 60)
            print("SERVICES AND CHARACTERISTICS")
            print("=" * 60)

            services = client.services
            health_services_found = []

            for service in services:
                service_uuid = str(service.uuid).lower()
                service_name = KNOWN_SERVICES.get(service_uuid, "Unknown Service")

                # Check if it's a health-related service
                is_health = service_name in ["Heart Rate Service", "Pulse Oximeter Service",
                                             "User Data Service", "Environmental Sensing"]

                marker = "❤️ " if is_health else "   "
                print(f"\n{marker}SERVICE: {service_name}")
                print(f"   UUID: {service.uuid}")

                if is_health:
                    health_services_found.append(service_name)

                for char in service.characteristics:
                    char_uuid = str(char.uuid).lower()
                    char_name = KNOWN_CHARACTERISTICS.get(char_uuid, "Unknown")
                    props = ", ".join(char.properties)
                    print(f"      └─ {char_name}")
                    print(f"         UUID: {char.uuid}")
                    print(f"         Properties: {props}")

                    # Try to read if readable
                    if "read" in char.properties:
                        try:
                            value = await client.read_gatt_char(char.uuid)
                            print(f"         Value: {value.hex()} ({list(value)})")
                        except Exception as e:
                            print(f"         Value: <cannot read: {e}>")

            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)

            if health_services_found:
                print(f"✅ Found health services: {', '.join(health_services_found)}")
                print("   You can potentially read health data from this device!")
            else:
                print("⚠️  No standard health BLE services found.")
                print("   The watch might use proprietary protocols or require a companion app.")

    except Exception as e:
        print(f"❌ Error connecting: {e}")
        print("\nTips:")
        print("  - Make sure the watch is not connected to another device")
        print("  - Try turning Bluetooth off/on on the watch")
        print("  - Some watches require pairing first via system Bluetooth settings")


async def main():
    print("=" * 60)
    print("   SleepWise BLE Scanner")
    print("   Find your smartwatch and discover available services")
    print("=" * 60)

    # Step 1: Scan for devices
    devices = await scan_for_devices(duration=10)

    if not devices:
        return

    # Step 2: Let user select a device
    print("\n" + "-" * 60)
    while True:
        choice = input("\nEnter device number to explore (or 'q' to quit, 'r' to rescan): ").strip()

        if choice.lower() == 'q':
            print("Goodbye!")
            return
        elif choice.lower() == 'r':
            devices = await scan_for_devices(duration=10)
            continue

        try:
            idx = int(choice)
            if 0 <= idx < len(devices):
                await explore_device(devices[idx].address)
            else:
                print(f"Please enter a number between 0 and {len(devices)-1}")
        except ValueError:
            print("Please enter a valid number")


if __name__ == "__main__":
    asyncio.run(main())
