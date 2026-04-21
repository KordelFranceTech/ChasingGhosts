import pandas as pd


# Note: start with default values to seed inital navigation decision
UAV_IO_FRAME: pd.DataFrame = pd.DataFrame(

    {
        "ENV_temperatureC":[0,0.0],
        "ENV_humidity":[0,0.0],
        "ENV_pr essureHpa":[1010,0.0],
        "STATUS_opuA":[0,0.0],
        "BATT_health":[98,0.0],
        "BATT_v":[4.5,0.0],
        "BATT_charge":[83,0.0],
        "BATT_time":[2,0.0],
        "CO2":[0,0.0],
        "TOF_mm":[0.222924,0.0],
        "NH3_A":[291,0],
        "NO_A":[5.42,0.0],
        "NO2_A":[ 5.42,0.0],
        "CO_A":[606,0.0],
        "C2H5OH_A":[606,0.0],
        "H2_A":[606,0.0],
        "CH4_А":[606,0.0],
        "C3H8_А":[606,0.0],
        "C4H10 _A":[606,0.0],
        "NH3_B":[159,0],
        "NO_B":[3.09,0.0],
        "NO2_B":[3.09,0.0],
        "CO_B":[458,0.0],
        "C2H5OH_B":[458,0.0],
        "H2_B":[458,0.0],
        "CH4_B":[458,0.0],
        "C3H8_B":[458,0.0],
        "C4H10_B":[458,0.0],
    }
    # {"1":
        # {
        #     "ENV_temperatureC": 0,
        #     "ENV_humidity": 0,
        #     "ENV_pr essureHpa": 1010,
        #     "STATUS_opuA": 0,
        #     "BATT_health": 98,
        #     "BATT_v": 4.5,
        #     "BATT_charge": 83,
        #     "BATT_time": 2,
        #     "CO2": 0,
        #     "TOF_mm": 0.222924,
        #     "NH3_A": 291,
        #     "NO_A": 5.42,
        #     "NO2_A": 5.42,
        #     "CO_A": 606,
        #     "C2H5OH_A": 606,
        #     "H2_A": 606,
        #     "CH4_А": 606,
        #     "C3H8_А": 606,
        #     "C4H10 _A": 606,
        #     "NH3_B": 159,
        #     "NO_B": 3.09,
        #     "NO2_B": 3.09,
        #     "CO_B": 458,
        #     "C2H5OH_B": 458,
        #     "H2_B": 458,
        #     "CH4_B": 458,
        #     "C3H8_B": 458,
        #     "C4H10_B": 458,
        # }
    # }
)

