from ..utils import ml_utils


if __name__ == "__main__":
    results = ml_utils.infer_doorways(uav_camera_frame=None, is_test=True)
    boxes = ml_utils.get_nearest_door(yolo_results=results)