from djitellopy import Tello


"""    
    // Wifi
    var socket = GCDAsyncUdpSocket()
    let sendHost = "192.168.10.1"
    let sendPort: UInt16 = 8889
    let statePort: UInt16 = 8890
"""
tello = Tello()

tello.connect()
tello.takeoff()

tello.move_left(100)
tello.rotate_counter_clockwise(90)
tello.move_forward(100)

tello.land()
