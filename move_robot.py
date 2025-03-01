#!/usr/bin/env python
"""
Main algorithm script for the real robot.  
Assumptions:
  - The robot’s base pose and current joint configuration are provided by the controller.
  - Camera data is acquired using PyRealSense.
  - No segmentation is available.
  - The selected grasp pose will be returned (or printed) so that the controller can execute the motion.
"""

import argparse
import datetime
import json
import numpy as np
import cv2
import sys

# Import the modified CGN agent for the real robot.
from spa.cgn_agent import CGNAgent

def get_camera_data():
    """Capture one RGB-D frame (and point cloud) from local image files."""
    # Hardcoded file paths for the RGB and depth images.
    rgb_file = "/home/jz/Desktop/cs4278_data/rgbd/rgb.jpg"       # Change to your RGB image file path.
    depth_file = "/home/jz/Desktop/cs4278_data/rgbd/d.jpg"     # Change to your depth image file path.
    
    # Load images from disk.
    rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        print("Error reading images from disk.", file=sys.stderr)
        return None

    intrinsic_cv = np.array([[600.8057861328125, 0.0, 325.5308532714844], 
                             [0.0, 600.634033203125, 252.90933227539062], 
                             [0.0, 0.0, 1.0]])
    extrinsic_cv = np.array([[0.99900709,-0.00422515,-0.0443507,0.1974],
                                     [-0.01567638,-0.96516695,-0.26116473,0.076],
                                     [-0.04170237,0.26160068,-0.96427486,0.609],
                                     [0.0,0.0,0.0,1.0]])

    # Convert depth image to meters.
    # (Assuming the depth image is in 16-bit PNG format with depth in millimeters)
    depth_m = depth.astype(np.float32) / 1000.0

    # Get image dimensions.
    height, width = depth_m.shape

    # Create a grid of (u,v) pixel coordinates.
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Compute the 3D coordinates for each pixel using the pinhole camera model.
    cx, cy = intrinsic_cv[0, 2], intrinsic_cv[1, 2]
    fx, fy = intrinsic_cv[0, 0], intrinsic_cv[1, 1]
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m

    # Only consider pixels with a valid depth value.
    valid = depth_m > 0
    x = x[valid]
    y = y[valid]
    z = z[valid]
    points = np.stack([x, y, z], axis=-1)

    # Add a dummy homogeneous coordinate (w = 1) to each 3D point.
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    pointcloud_xyzwh = np.hstack([points, ones])
    
    # Get corresponding RGB values for valid depth pixels.
    rgb_valid = rgb[valid]

    pointcloud = {
        'xyzw': pointcloud_xyzwh,
        'rgb': rgb_valid
    }

    # Build the observation dictionary.
    obs = {
        'image': {
            'hand_camera': {
                'rgb': rgb,
                'depth': depth  # Original depth image (in raw units)
            }
        },
        'camera_param': {
            'hand_camera': {
                'intrinsic_cv': intrinsic_cv,
                'extrinsic_cv': extrinsic_cv.tolist()
            }
        },
        'pointcloud': pointcloud,
        # Hardcoded current joint configuration (example values).
        'current_qpos': np.array([0.1974360455588751, 0.07597034947681322, 0.6089763035481176, -0.7043158661916847, -0.6972020477573383, -0.07785831859184095, -0.10856586691811596])
    }
    return obs

def main(selected_model, selected_tasks, task, visual_method):
    # For logging and unique run identification.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting run: {timestamp}")

    # (2) Acquire camera data from the RealSense camera.
    print("Acquiring camera data...")
    obs = get_camera_data()
    if obs is None:
        print("Failed to capture camera data. Exiting.", file=sys.stderr)
        return

    # (4) Initialize the CGN agent.
    agent = CGNAgent()
    
    # (5) Get the selected grasp pose.
    print("Querying CGN Agent for a grasp pose...")
    grasp_result = agent.get_grasp_traj(
        selected_model=selected_model,
        selected_tasks=selected_tasks,
        task=task,
        visual_method=visual_method,
        obs=obs,
        eval_eps=timestamp
    )
    
    if False:
    # if grasp_result.get('status') != 'Success':
        print(f"Grasp generation failed: {grasp_result.get('status')}")
    else:
        grasp_pose_world = grasp_result.get('grasp_pose_world')
        print("Selected Grasp Pose (world frame):")
        print(grasp_pose_world)
        # (6) You can now pass this grasp pose back to your controller for execution.

    # (Optional) Save the result to a JSON file.
    results = {
        "timestamp": timestamp,
        "selected_model": selected_model,
        "selected_tasks": selected_tasks,
        "task": task,
        "visual_method": visual_method,
        "grasp_result": grasp_result
    }
    print(results)
    # with open(f"results_{selected_tasks}_{timestamp}.json", "w") as f:
    #     json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Robot: Grasp Generation Algorithm")
    parser.add_argument("--task", required=True,
                        choices=["task_1", "task_2", "task_3", "task_4", "task_5a", "task_5b"],
                        help="Task name to run")
    parser.add_argument("--task_idx", type=int, required=True,
                        help="Task index to run")
    parser.add_argument("--visual_method", required=True,
                        choices=["gripper_one", "gripper_all", "contact_one", "contact_all", "distance"],
                        help="Visual method to use")
    parser.add_argument("--model", required=True,
                        choices=["molmo", "qwen_7B", "gemini", "qwen_72B"],
                        help="Model name to use (molmo, qwen_7B, qwen_72B)")
    args = parser.parse_args()

    # Example task dictionary (replace or expand as needed).
    tasks = {
        "task_1": [
            "Grip the toy by the front (head or front paws).",
            "Grip the toy by the tail to keep the head visible."
        ],
        "task_2": [
            "Retrieve the object by gripping the handle to ensure a secure hold.",
            "Retrieve the object by gripping a part other than the handle to make it easier for me to handle."
        ],
        "task_3": [
            "Retrieve the tools by gripping the handle to ensure a secure hold.",
            "Retrieve the tools by gripping a part other than the handle to make it easier for me to handle."
        ],
        "task_4": [
            "Retrieve the bottle by gripping the body to allow easy twisting.",
            "Retrieve the bottle by gripping the head part to make it easier for me to handle."
        ],
        "task_5a": [
            "Among all the bottles here, identify and retrieve the ketchup bottle.",
            "Among all the bottles here, identify and retrieve the milk bottle.",
            "Among all the bottles here, identify and retrieve the honey bottle."
        ],
        "task_5b": [
            "Among all the animal toys here, identify and retrieve the white sheep.",
            "Among all the animal toys here, identify and retrieve the brown cow.",
            "Among all the animal toys here, identify and retrieve the pink pig."
        ]
    }

    selected_tasks = args.task
    idx = args.task_idx
    t = tasks[selected_tasks][idx]
    model = args.model
    visual_method = args.visual_method

    main(selected_model=model,
         selected_tasks=f"{selected_tasks}_{idx}",
         task=t,
         visual_method=visual_method)
