import asyncio
import time
import json
import pandas as pd
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
    # loop.run_until_complete(async_sample_from_device())
    return loop.run_until_complete(async_iterate_devices())


async def async_iterate_devices():
    devices = await bleak.BleakScanner.discover()
    target_device = None
    for device in devices:
        print(device.name)
        # if device.name == constants.SERVICE_NAME:
        if "UAV" in str(device.name):
            target_device = device
            print(f"target device found: {target_device.name}")
            return target_device
    if target_device is None:
        print("No Scentience devices found")
        return None


async def async_sample_from_device(target_device):

    async def notification_handler(sender, data):
        # Decode received data
        da = data.decode('utf-8')
        # print(f'received data:\n\t{da}')
        db: dict = json.loads(da)
        dc = pd.DataFrame([db])

        # Append the new row using append()

        import io_data
        df_old: pd.DataFrame = io_data.UAV_IO_FRAME
        # df_updated: pd.DataFrame = df_old.append(new_row_series, ignore_index=True)
        df_updated = pd.concat([dc, df_old], ignore_index=True)
        io_data.UAV_IO_FRAME = df_updated
        # print(f"received data: {db}")

    async with BleakClient(target_device.address) as client:
        # print(f"Connected to {target_device.name}")

        # # Get all services
        # for service in client.services:
        #     print(f"Service: {service.uuid} (Handle: {service.handle})")
        #
        #     # Get all characteristics within each service
        #     for char in service.characteristics:
        #         print(f"\tCharacteristic: {char.uuid} (Handle: {char.handle})")
        #         print(f"\t\tProperties: {','.join(char.properties)}")


        # Enable notifications for the characteristic
        await client.start_notify(constants.CHARACTERISTIC_UUID, notification_handler)
        # Keep the script running to receive notifications
        # Run for 10 seconds...adjust as needed
        await asyncio.sleep(constants.BLE_SAMPLE_TIME)


def sync_connect_to_device():
    devices = bleak.BleakScanner.discover()
    time.sleep(10)
    target_device = None
    for device in devices:
        print(device.name)
        # if device.name == constants.SERVICE_NAME:
        if "UAV" in str(device.name):
            target_device = device
            print(f"target device found: {target_device.name}")
            return target_device
    if target_device is None:
        print("No Scentience devices found")
        return None


def sync_sample_from_device(target_device):
    def notification_handler(sender, data):
        # Decode received data
        ppm = data.decode('utf-8')
        print(f"received data: {ppm}")

    with BleakClient(target_device.address) as client:
        print(f"Connected to {target_device.name}")

        # Get all services
        for service in client.services:
            print(f"Service: {service.uuid} (Handle: {service.handle})")

            # Get all characteristics within each service
            for char in service.characteristics:
                print(f"\tCharacteristic: {char.uuid} (Handle: {char.handle})")
                print(f"\t\tProperties: {','.join(char.properties)}")


        # Enable notifications for the characteristic
        client.start_notify(constants.CHARACTERISTIC_UUID, notification_handler)
        # Keep the script running to receive notifications
        # Run for 10 seconds...adjust as needed
        # asyncio.sleep(constants.BLE_SAMPLE_TIME)

"""    
    // Wifi
    var socket = GCDAsyncUdpSocket()
    let sendHost = "192.168.10.1"
    let sendPort: UInt16 = 8889
    let statePort: UInt16 = 8890
"""