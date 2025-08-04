import asyncio
import time
import pandas as pd
from djitellopy import Tello

from .utils import ble_utils
from . import constants


os_l_list: list = []
os_r_list: list = []


"""
Data I/O example:
"""
io_example: dict ={
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
    from .io_data import UAV_IO_FRAME
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
    from .io_data import UAV_IO_FRAME
    tello.send_keepalive()
    uav_state = tello.get_current_state()
    print(f"\tcurrent uav state: {uav_state}")
    olf_l_sample = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_A"]
    olf_r_sample = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_B"]
    os_l_list.append(olf_l_sample)
    os_r_list.append(olf_r_sample)
    os_l_ddx_list: pd.DataFrame = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_A"].diff()
    os_r_ddx_list: pd.DataFrame = UAV_IO_FRAME[f"{constants.TARGET_COMPOUND}_B"].diff()
    os_l_d2dx2_list: list = os_l_ddx_list.diff().tolist()
    os_r_d2dx2_list: list = os_r_ddx_list.diff().tolist()
    os_l_d2dx2: float = os_l_d2dx2_list[0]
    os_r_d2dx2: float = os_r_d2dx2_list[0]

    if os_l_d2dx2 > os_r_d2dx2:
        # tello.move_left(100)
        tello.rotate_counter_clockwise(90)
        tello.move_forward(10)
    elif os_l_d2dx2 > os_r_d2dx2:
        # tello.move_right(100)
        tello.rotate_clockwise(90)
        tello.move_forward(10)
    else:
        if float(UAV_IO_FRAME["ToF_mm"]) > 120.0:
            tello.move_forward(10)
        else:
            tello.move_back(10)


if __name__ == "__main__":
    asyncio.run(ble_utils.discover_ble_devices())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(ble_utils.connect_to_ble_device())

    tello = Tello()
    tello.connect()
    tello.takeoff()
    tello.rotate_clockwise(90)
    tello.rotate_counter_clockwise(90)
    tello.move_forward(10)

    for _ in range(10):
        command_loop()
        time.sleep(10)

    tello.land()
    if tello.stream_on:
        tello.streamoff()
