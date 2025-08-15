from djitellopy import Tello
import time

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

# print(f"state: {tello.get_current_state()}")
# time.sleep(1)
tello.move_forward(250)
tello.rotate_counter_clockwise(90)
tello.move_forward(170)
# tello.move_backward(150)
tello.rotate_clockwise(90)
tello.move_forward(750)
tello.sleep(5)
tello.land()


# tello.move_left(100)
# tello.rotate_counter_clockwise(90)
# tello.move_forward(100)
#
# tello.land()
