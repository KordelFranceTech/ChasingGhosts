import asyncio
import bleak
from bleak import BleakClient, BleakScanner
from djitellopy import Tello


# BLE params
SERVICE_UUID: str = "569a1101-b87f-490c-92cb-11ba5ea5167c"
CHARACTERISTIC_UUID: str = "569a2000-b87f-490c-92cb-11ba5ea5167c"
SERVICE_NAME: str = "A00019"


async def scan(address):
    async with BleakClient(address) as client:
        print(f"Connected: {client.is_connected}")

        # Get all services
        for service in client.services:
            print(f"Service: {service.uuid} (Handle: {service.handle})")

            # Get all characteristics within each service
            for char in service.characteristics:
                print(f"\tCharacteristic: {char.uuid} (Handle: {char.handle})")
                print(f"\t\tProperties: {','.join(char.properties)}")

                # Read characteristic value if it has "read" property
                if "read" in char.properties:
                    try:
                        value = await client.read_gatt_char(char.uuid)
                        print(f"\t\tValue: {value.hex()}")  # Print as hex for bytearray
                    except Exception as e:
                        print(f"\t\tError reading characteristic: {e}")

                # Get all descriptors within each characteristic
                for descriptor in char.descriptors:
                    print(f"\t\tDescriptor: {descriptor.uuid} (Handle: {descriptor.handle})")
                    try:
                        value = await client.read_gatt_descriptor(descriptor.handle)
                        print(f"\t\t\tValue: {value.hex()}")  # Print as hex for bytearray
                    except Exception as e:
                        print(f"\t\t\tError reading descriptor: {e}")




async def simple_scan():
    devices = await BleakScanner.discover()
    for d in devices:
        print(d)


async def run():
    devices = await bleak.BleakScanner.discover()
    target_device = None
    for device in devices:
        print(device.name)
        if device.name == SERVICE_NAME:
            target_device = device
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

if __name__ == "__main__":
    # asyncio.run(main())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())

    # tello = Tello()
    #
    # tello.connect()
    # tello.takeoff()
    #
    # tello.move_left(100)
    # tello.rotate_counter_clockwise(90)
    # tello.move_forward(100)
    #
    # tello.land()
