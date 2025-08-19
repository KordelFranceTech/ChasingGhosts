DEBUG_MODE: bool = False
FLIGHT_MODE: bool = False

# BLE params
SERVICE_UUID: str = "569a1101-b87f-490c-92cb-11ba5ea5167c"
CHARACTERISTIC_UUID: str = "569a2000-b87f-490c-92cb-11ba5ea5167c"
SERVICE_NAME: str = "UAV001"
BLE_SAMPLE_TIME: int = 4
STEP_TIME: int = 1

# Chemical params
TARGET_COMPOUND: str = "NH3"

# ML Models
YOLO_MODEL_DOOR: str = "yolo_doors.pt"
YOLO_MODEL_GENERAL: str = "yolo11n.pt"

