import asyncio
import bleak
from bleak import BleakClient, BleakScanner

from ..io_data import UAV_IO_FRAME


# BLE params
SERVICE_UUID: str = "569a1101-b87f-490c-92cb-11ba5ea5167c"
CHARACTERISTIC_UUID: str = "569a2002-b87f-490c-92cb-11ba5ea5167c"
SERVICE_NAME: str = "Scentience"



async def discover_ble_devices():
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)


async def connect_to_ble_device():
    devices = await bleak.BleakScanner.discover()
    target_device = None
    for device in devices:
        if device.name == SERVICE_NAME:
            target_device = device
            break
    if target_device is None:
        print("No Scentience devices found")
        return

    async def notification_handler(sender, data):
        # Decode received data
        io_data = data.decode('utf-8')
        print(f"received data: {io_data}")
        UAV_IO_FRAME.append(dict(io_data))

    async with BleakClient(target_device.address) as client:
        print(f"Connected to {target_device.name}")
        # Enable notifications for the characteristic
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        # Keep the script running to receive notifications
        # Run for 10 seconds...adjust as needed
        await asyncio.sleep(10)

"""    
    // Wifi
    var socket = GCDAsyncUdpSocket()
    let sendHost = "192.168.10.1"
    let sendPort: UInt16 = 8889
    let statePort: UInt16 = 8890
"""