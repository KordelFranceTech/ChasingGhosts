from ultralytics import YOLO
import numpy as np

# from .. import constants


YOLO_MODEL_DOOR: str = "models/yolo_doors.pt"
YOLO_MODEL_GENERAL: str = "models/yolo11n.pt"


def infer_doorways(uav_camera_frame, is_test:bool=False, should_display:bool=False) -> list:
    # Load a pretrained YOLO11n model
    model = YOLO(YOLO_MODEL_DOOR)
    # Perform object detection on an image
    if is_test:
        results = model("images/door_test.jpg")  # Predict on an image
    else:
        results = model(uav_camera_frame)  # Predict on an image
    if should_display:
        results[0].show()  # Display results
    return results


def infer_general(uav_camera_frame, is_test:bool=False, should_display:bool=False) -> list:
    # Load a pretrained YOLO11n model
    model = YOLO(YOLO_MODEL_DOOR)
    # Perform object detection on an image
    if is_test:
        results = model("images/test_image.png")  # Predict on an image
    else:
        results = model(uav_camera_frame)  # Predict on an image
    if should_display:
        results[0].show()  # Display results
    return results


def get_nearest_door(yolo_results: list):
    if len(yolo_results) > 0:
        if yolo_results[0].probs is not None:
            max_prob: float = -1.0
            max_index: int = 0
            for i in range(0, len(yolo_results)):
                    if yolo_results["probs"][i] > max_prob:
                        max_prob = yolo_results["probs"][i]
                        max_index = i
            best_r: dict = yolo_results[max_index]
            best_box = best_r.boxes
            return best_box.tolist()
        else:
            if yolo_results[0].boxes is not None:
                best_indices: list = []
                for i in range(0, len(yolo_results)):
                    xywhn_list: list = yolo_results[i].boxes.xywhn.tolist()
                    for j in range(0, len(xywhn_list)):
                        xywhn: list = xywhn_list[j]
                        # Check width over height aspect ratio - anything else is a window
                        aspect_ratio: float = float(xywhn[2]) / float(xywhn[3])
                        if aspect_ratio > 0.6:
                            best_indices.append(j)
                best_index: int = np.argmax(np.array(best_indices))
                best_box: list = yolo_results[0].boxes.xywhn[best_index]
                return best_box.tolist()
    return None


def get_centroid_of_nearest_door(door_xywhn: list):
    xn: float = door_xywhn[0]
    yn: float = door_xywhn[1]
    wn: float = door_xywhn[2]
    hn: float = door_xywhn[3]
    cx: float = xn + (0.5 * wn)
    cy: float = yn + (0.5 * hn)
    return cx, cy



if __name__ == "__main__":
    results = infer_doorways(uav_camera_frame=None, is_test=True, should_display=True)
    boxes = get_nearest_door(yolo_results=results)
    centroid = get_centroid_of_nearest_door(boxes)
    print(f"best box: {boxes}")
    print(f"centroid: {centroid}")



