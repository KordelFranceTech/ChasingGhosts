import cv2
from ultralytics import YOLO
import time

for i in range(8):

    # Load a pretrained YOLO11n model
    # model = YOLO("yolo11n.pt")

    model = YOLO("yolo_doors.pt")  # Load the best model

    # # Train the model on the COCO8 dataset for 100 epochs
    # train_results = model.train(
    #     data="coco8.yaml",  # Path to dataset configuration file
    #     epochs=100,  # Number of training epochs
    #     imgsz=640,  # Image size for training
    #     device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    # )
    #
    # # Evaluate the model's performance on the validation set
    # metrics = model.val()

    # Perform object detection on an image
    # results = model("../images/door_test.jpg")  # Predict on an image
    results = model(f"../images/doors/door_{i}.png")  # Predict on an image
    print(results[0])
    print("****")
    print(results)
    results[0].show()  # Display results

# Export the model to ONNX format for deployment
# path = model.export(format="onnx")  # Returns the path to the exported model

# tello.land()

