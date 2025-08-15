import asyncio
import bleak
from bleak import BleakClient, BleakScanner

# from ..io_data import UAV_IO_FRAME
import constants


async def discover_ble_devices():
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)


def connect_to_sensor():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_sample_from_device())


async def async_sample_from_device():
    devices = await bleak.BleakScanner.discover()
    target_device = None
    for device in devices:
        print(device.name)
        # if device.name == constants.SERVICE_NAME:
        if "UAV" in str(device.name):
            target_device = device
            print(f"target device found: {target_device.name}")
            break
    if target_device is None:
        print("No Scentience devices found")
        return

    async def notification_handler(sender, data):
        # Decode received data
        ppm = data.decode('utf-8')
        print(f"received data: {ppm}")

    async with BleakClient(target_device.address) as client:
        print(f"Connected to {target_device.name}")

        # Get all services
        for service in client.services:
            print(f"Service: {service.uuid} (Handle: {service.handle})")

            # Get all characteristics within each service
            for char in service.characteristics:
                print(f"\tCharacteristic: {char.uuid} (Handle: {char.handle})")
                print(f"\t\tProperties: {','.join(char.properties)}")


        # Enable notifications for the characteristic
        await client.start_notify(constants.CHARACTERISTIC_UUID, notification_handler)
        # Keep the script running to receive notifications
        # Run for 10 seconds...adjust as needed
        await asyncio.sleep(constants.BLE_SAMPLE_TIME)

"""    
    // Wifi
    var socket = GCDAsyncUdpSocket()
    let sendHost = "192.168.10.1"
    let sendPort: UInt16 = 8889
    let statePort: UInt16 = 8890
"""