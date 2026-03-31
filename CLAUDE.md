# Chasing Ghosts - UAV Navigation Project

## Goal
This project is based on the paper "Chasing Ghosts" from the following URL: https://arxiv.org/pdf/2602.19577.
I am the author.
The goal of this project is to command a UAV to navigate a course using vision and chemical sensing.
The UAV has a camera, some downward depth sensors for leveling, a forward-looking infrared depth sensor, and two chemical sensors that track a target compound and stick out from the airframe like antennae.
A Tello UAV has an olfaction processing unit (OPU) attached to its belly as payload.
This OPU coordinates the computation needed to process the data from the chemical sensors to facilitate scent-based navigation.
In `uav_main.py`, I instruct the Tello UAV to connect with the OPU over Bluetooth and stream the data so that the UAV can make decisions based on this data, and navigate itsel to the source.
The code sends and receives control commands to the Tello over local area network connection.
Through this connection, it can also stream images from the camera as well as other telemetry.
It sends and recieves olfaction data over Bluetooth connection.

The COLIP embeddings (or olfaction-vision-language model) associated with the paper can be found here on HuggingFace: https://huggingface.co/kordelfrance/Olfaction-Vision-Language-Embeddings.
The specific variant of the model I use from this HuggingFace repository can be found here: https://huggingface.co/kordelfrance/Olfaction-Vision-Language-Embeddings/tree/main/model/colip-small/base

The YOLO model I use for fast updates mentioned in the paper can be found within the `./models` subdirectory of this repository here.

The reinforcement learning model is trained in navigation/paper/simple/plume_env_v4_1.py, and a policy is output that is used in the paper.
