import cv2
from ultralytics import YOLO
from .. import constants




def infer_doorways(uav_camera_frame, is_test:bool=False, should_display:bool=False) -> dict:
    # Load a pretrained YOLO11n model
    model = YOLO(constants.YOLO_MODEL_DOOR)
    # Perform object detection on an image
    if is_test:
        results = model("door_test.jpg")  # Predict on an image
    else:
        results = model(uav_camera_frame)  # Predict on an image
    if should_display:
        results[0].show()  # Display results
    return results


def infer_general(uav_camera_frame, is_test:bool=False, should_display:bool=False) -> dict:
    # Load a pretrained YOLO11n model
    model = YOLO(constants.YOLO_MODEL_DOOR)
    # Perform object detection on an image
    if is_test:
        results = model("test_image.png")  # Predict on an image
    else:
        results = model(uav_camera_frame)  # Predict on an image
    if should_display:
        results[0].show()  # Display results
    return results


