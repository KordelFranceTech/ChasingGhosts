import asyncio
import math
import sys
import time
import pandas as pd
from djitellopy import Tello

from utils import ble_utils, ml_utils
import constants


os_l_list: list = []
os_r_list: list = []

"""
Note this is the most direct path for the course if no plume tracking.
Used to test battery longevity.
"""
course_commands: list = [
        "takeoff",
        "forward 500",  # split from 565
        "forward 65",
        "cw 90",
        "forward 400",
        "ccw 90",
        "forward 343",
        "cw 90",
        "forward 154",
        "forward 500",  # split from 700
        "forward 200",
        "forward 170",
        "ccw 90",
        "forward 145",
        "cw 90",
        "forward 500",
        "ccw 90",
        "forward 120",
        "land",
        ]


"""
Data I/O example:
"""
io_example: dict = {
    "ENV_temperatureC":24,
    "ENV_humidity":62,
    "ENV_pressureHpa":1010,
    "STATUS_opuA":0,
    "BATT_health":98,
    "BATT_v":4.5,
    "BATT_charge":83,
    "BATT_time":2,
    "CO2":1058,
    "CO_A":720,
    "NH3_A":93,
    "NO2_A":5.68,
    "CO_B":720,
    "NH3_B":93,
    "NO2_B":5.68,
    "ToF_mm":100.0
}



def command_loop():
    from io_data import UAV_IO_FRAME
    tello.send_keepalive()
    uav_state = tello.get_current_state()
    print(f"\tcurrent uav state: {uav_state}")
    olf_l_sample = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_A"]
    olf_r_sample = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_B"]
    os_l_list.append(olf_l_sample)
    os_r_list.append(olf_r_sample)

    if olf_l_sample > olf_r_sample:
        # tello.move_left(100)
        tello.rotate_counter_clockwise(90)
        tello.move_forward(10)
    elif olf_r_sample > olf_l_sample:
        # tello.move_right(100)
        tello.rotate_clockwise(90)
        tello.move_forward(10)
    else:
        if float(UAV_IO_FRAME["ToF_mm"]) > 120.0:
            tello.move_forward(10)
        else:
            tello.move_back(10)


def command_loop_with_bout_detection():
    from io_data import UAV_IO_FRAME
    if constants.FLIGHT_MODE:
        tello.send_keepalive()
        uav_state = tello.get_current_state()
        print(f"\tcurrent uav state: {uav_state}")
    else:
        print("Sampling olfaction sensors...")
        time.sleep(constants.STEP_TIME)
    # print(f"io frame: {UAV_IO_FRAME}")
    olf_l_sample = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_B"]
    olf_r_sample = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_A"]
    os_l_list.append(olf_l_sample)
    os_r_list.append(olf_r_sample)
    # os_l_list.pop(0)
    # os_r_list.pop(0)
    os_l_ddx_list: pd.DataFrame = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_A"].diff()
    os_r_ddx_list: pd.DataFrame = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_B"].diff()
    os_l_d2dx2_list: list = os_l_ddx_list.diff().tolist()
    os_r_d2dx2_list: list = os_r_ddx_list.diff().tolist()
    os_l_d2dx2_list.pop(0)
    os_l_d2dx2_list.pop(0)
    os_r_d2dx2_list.pop(0)
    os_r_d2dx2_list.pop(0)
    os_l_d2dx2: float = os_l_d2dx2_list[0]
    os_r_d2dx2: float = os_r_d2dx2_list[0]
    # os_l_d2dx2: float = os_l_ddx_list[0]
    # os_r_d2dx2: float = os_r_ddx_list[0]

    print(f"\n\tbout list d/dx:\n\t\tleft: {os_l_ddx_list.tolist()}\n\t\tright: {os_r_ddx_list.tolist()}\n")
    print(f"\n\tbout list d2/dx2:\n\t\tleft: {os_l_d2dx2_list}\n\t\tright: {os_r_d2dx2_list}\n")

    if (os_r_d2dx2 != 0.0 and os_l_d2dx2 / os_r_d2dx2 > 1.2) or (os_l_d2dx2 != 0.0 and os_r_d2dx2 / os_l_d2dx2 < 0.8):
        print("moving left...")
        # tello.move_left(100)
        if constants.FLIGHT_MODE:
            tello.rotate_counter_clockwise(90)
            time.sleep(constants.STEP_TIME)
        else:
            print("tello.rotate_counter_clockwise(90)")
            time.sleep(constants.STEP_TIME)

    elif (os_l_d2dx2 != 0.0 and os_r_d2dx2 / os_l_d2dx2 > 1.2) or (os_r_d2dx2 != 0.0 and os_l_d2dx2 / os_r_d2dx2 < 0.8):
        print("moving right...")
        # tello.move_right(100)
        if constants.FLIGHT_MODE:
            tello.rotate_clockwise(90)
            time.sleep(constants.STEP_TIME)
        else:
            print("tello.rotate_clockwise(90)")
            time.sleep(constants.STEP_TIME)
    else:
        print("not moving...")

    if float(UAV_IO_FRAME.iloc[0]["TOF_mm"]) > 0.14:
        if constants.FLIGHT_MODE:
            tello.move_forward(10)
            time.sleep(constants.STEP_TIME)
        else:
            print("tello.move_forward(10)")
            time.sleep(constants.STEP_TIME)
    else:
        if constants.FLIGHT_MODE:
            tello.move_back(10)
            time.sleep(constants.STEP_TIME)
        else:
            print("tello.move_back(10)")
            time.sleep(constants.STEP_TIME)


def navigate_to_door():
    from .io_data import UAV_IO_FRAME
    if not tello.stream_on():
        return
    frame_read = tello.get_frame_read()
    yolo_results = ml_utils.infer_doorways(uav_camera_frame=frame_read, is_test=False, should_display=True)
    if yolo_results is not None:
        door = ml_utils.get_nearest_door(yolo_results=yolo_results)
        door_cx, door_cy = ml_utils.get_centroid_of_nearest_door(door_xywhn=door)
        # We are too far right of the door, move left
        if float(door_cx) < 0.4:
            tello.move_left(5)
        # We are too far left of the door, move right
        elif float(door_cx) > 0.6:
            tello.move_right(5)
        # We are nearly aligned with door center, move forward
        else:
            if float(UAV_IO_FRAME["ToF_mm"]) > 0.18:
                tello.move_forward(10)
            else:
                tello.move_back(10)


if __name__ == "__main__":

    # target_device = ble_utils.connect_to_sensor()
    # if target_device is None:
    #     sys.exit()
    time.sleep(constants.STEP_TIME)

    if constants.FLIGHT_MODE:
        tello = Tello()
        tello.connect()
        time.sleep(constants.STEP_TIME)
        battery = tello.get_battery()
        state = tello.get_current_state()
        temp_high = state.get('temph', 0)
        print(f"Battery: {battery}%")
        print(f"Pitch: {state.get('pitch')}  Roll: {state.get('roll')}  Yaw: {state.get('yaw')}")
        print(f"Height: {state.get('h')}  ToF: {state.get('tof')}  Temp: {state.get('templ')}-{temp_high}C")
        if battery < 20:
            print("Battery too low to fly safely. Charge the Tello and try again.")
            sys.exit()
        if temp_high > 85:
            print(f"Tello is overheated ({temp_high}C). Power off and let it cool for 10-15 minutes.")
            sys.exit()
        def stable_rotate(degrees: int, clockwise: bool, step: int = 30, settle_threshold: int = 3, settle_timeout: float = 3.0):
            """Rotate in small increments and wait for IMU to settle between steps.
            Avoids 'no valid IMU' errors caused by payload-induced vibration during
            high-angular-acceleration single-command rotations.
            Args:
                degrees: total rotation in degrees (positive integer)
                clockwise: True for CW, False for CCW
                step: degrees per increment (smaller = gentler, slower)
                settle_threshold: max abs pitch/roll in degrees before proceeding
                settle_timeout: seconds to wait for settle before giving up
            """
            remaining = degrees
            while remaining > 0:
                increment = min(step, remaining)
                if clockwise:
                    tello.rotate_clockwise(increment)
                else:
                    tello.rotate_counter_clockwise(increment)
                remaining -= increment
                deadline = time.time() + settle_timeout
                while time.time() < deadline:
                    s = tello.get_current_state()
                    if abs(s.get('pitch', 99)) <= settle_threshold and abs(s.get('roll', 99)) <= settle_threshold:
                        break
                    time.sleep(0.1)
                time.sleep(0.5)

        tello.takeoff()
        time.sleep(constants.STEP_TIME)
        try:
            tello.move_forward(100)
            time.sleep(constants.STEP_TIME)
            stable_rotate(90, clockwise=True)
            time.sleep(constants.STEP_TIME)
            stable_rotate(270, clockwise=False)
            time.sleep(constants.STEP_TIME)
            tello.move_forward(100)
            time.sleep(constants.STEP_TIME)
            ## Tello move commands are capped at 500 cm — split 565 into two legs
            # tello.move_forward(495)
            # time.sleep(constants.STEP_TIME)
            # tello.move_forward(70)
            # time.sleep(constants.STEP_TIME)
            # tello.rotate_clockwise(90)
            # time.sleep(constants.STEP_TIME)
            # tello.move_forward(400)
            # time.sleep(constants.STEP_TIME)
            # tello.rotate_counter_clockwise(90)
            # time.sleep(constants.STEP_TIME)
            # tello.move_forward(343)
            # time.sleep(constants.STEP_TIME)
        finally:
            try:
                tello.land()
            except Exception:
                # Motors already stopped (e.g. motor stop safety trigger); drone is grounded
                pass
