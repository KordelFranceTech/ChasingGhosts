import asyncio
import time
import pandas as pd
from djitellopy import Tello

from utils import ble_utils, ml_utils
import constants


os_l_list: list = []
os_r_list: list = []


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
    olf_l_sample = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_A"]
    olf_r_sample = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_B"]
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
    # os_l_d2dx2: float = os_l_d2dx2_list[0]
    # os_r_d2dx2: float = os_r_d2dx2_list[0]
    os_l_d2dx2: float = os_l_ddx_list[0]
    os_r_d2dx2: float = os_r_ddx_list[0]

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

    if float(UAV_IO_FRAME.iloc[0]["TOF_mm"]) > 0.18:
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

    target_device = ble_utils.connect_to_sensor()
    time.sleep(constants.STEP_TIME)

    if constants.FLIGHT_MODE:
        tello = Tello()
        tello.connect()
        # tello.stream_on()
        tello.takeoff()
        time.sleep(constants.STEP_TIME)

        # Health check
        tello.rotate_clockwise(90)
        time.sleep(constants.STEP_TIME)
        tello.rotate_counter_clockwise(90)
        time.sleep(constants.STEP_TIME)
        tello.move_forward(10)
        time.sleep(constants.STEP_TIME)
        tello.move_back(10)
        time.sleep(constants.STEP_TIME)
    else:
        print("\n----INIT----")
        print("tello = Tello()")
        print("tello.connect()")
        print("tello.stream_on()")
        print("tello.takeoff()")
        time.sleep(constants.STEP_TIME)

        # Health check
        print("\n------HEALTH CHECK----")
        print("tello.rotate_clockwise(90)")
        print("tello.rotate_counter_clockwise(90)")
        print("tello.move_forward(10)")
        print("tello.move_back(10)")
        time.sleep(constants.STEP_TIME)

    # Sample twice to build enough bank for bout detection
    if target_device is not None:
        asyncio.run(ble_utils.async_sample_from_device(target_device))
        asyncio.run(ble_utils.async_sample_from_device(target_device))


    for i in range(20):
        print(f"\n-----COMMAND LOOP {i + 1}-----")
        # ble_utils.connect_to_sensor()
        if target_device is not None:
            asyncio.run(ble_utils.async_sample_from_device(target_device))
        else:
            break
        if not constants.DEBUG_MODE:
            # # Check for nearest door
            # navigate_to_door()
            # Sample olfaction sensors and make decisions
            # command_loop()
            command_loop_with_bout_detection()
        time.sleep(constants.STEP_TIME)

    if constants.FLIGHT_MODE:
        tello.land()
        # if tello.stream_on:
        #     tello.streamoff()
        tello.end()




