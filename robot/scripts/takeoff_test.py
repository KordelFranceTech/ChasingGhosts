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
# tello.streamon()
# time.sleep(3)
# tello.streamoff()
# time.sleep(3)
# tello.turn_motor_on()
# time.sleep(3)

tello.takeoff()

# takeoff = False
#
# while not takeoff:
#     try:
#         tello.takeoff()
#         takeoff = True
#     except djitellopy.tello.TelloException:
#         break


# # print(f"state: {tello.get_current_state()}")
# # time.sleep(1)
# tello.move_forward(250)
# tello.rotate_counter_clockwise(90)
# tello.move_forward(170)
# # tello.move_backward(150)
# tello.rotate_clockwise(90)
# tello.move_forward(750)
# time.sleep(5)
tello.land()


# tello.move_left(100)
# tello.rotate_counter_clockwise(90)
# tello.move_forward(100)
#
# tello.land()
