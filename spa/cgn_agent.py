import logging
import numpy as np
import mplib
from pyquaternion import Quaternion
import io
import requests
import time
import base64
import re
from PIL import Image
import os
import cv2
from third_party.contact_graspnet.contact_graspnet.utils.ipc import CGNClient
import json
import google.generativeai as genai

# import sapien.core as sapien
logging.basicConfig(level=logging.INFO)

class CGNAgent():
    def __init__(self) -> None:
        self.cgn_client = CGNClient()
        self.planner = None
        self.state = 0
        self.step = 0
        self.init_planner()

        self.grasp_steps = 10
        self.move_to_goal_steps = 20

    def reset(self):
        self.state = 0
        self.step = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cgn_client.close()

    def init_planner(self):
        link_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_link8', 'panda_hand', 'panda_hand_tcp', 'panda_leftfinger', 'panda_rightfinger']
        joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
        from mani_skill2 import PACKAGE_ASSET_DIR
        planner = mplib.Planner(
            urdf=f"{PACKAGE_ASSET_DIR}/descriptions/panda_v2.urdf",
            srdf=f"{PACKAGE_ASSET_DIR}/descriptions/panda_v2.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))    
        self.planner = planner
        
    def get_grasp_traj(self, selected_model, selected_tasks, task, visual_method, obs, eval_eps=None):

        rgb = obs['image']['hand_camera']['rgb']  # h, w, 3
        depth = obs['image']['hand_camera']['depth']  # h, w, 1
        camera_k = obs['camera_param']['hand_camera']['intrinsic_cv']  # in CV convention
        camera_extrinsic = obs['camera_param']['hand_camera']['extrinsic_cv']  # 4x4 matrix
        point_cloud = obs['pointcloud']
        robot_qpos = obs['current_qpos']

        data_to_send = {
            'command': 'generate_grasps',
            'visualization': visual_method,
            'task': task,
            'selected_tasks': selected_tasks,
            'selected_model': selected_model,
            'data': {
                'rgb': rgb,
                'depth': depth,
                'cam_K': camera_k,
            }
        }

        rgb_image = Image.fromarray(rgb.astype('uint8'))  # Convert NumPy array to image
        rgb_image.save('rgb_image.png')
        depth = np.squeeze(depth)  # Now depth has shape (h, w)

        # Normalize depth for visualization (optional)
        depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

        # Convert depth to 8-bit image format (values between 0-255)
        depth_image = Image.fromarray((depth_normalized * 255).astype('uint8'))

        # Save the depth image
        depth_image.save('depth_image.png')

        import trimesh

        # print(point_cloud)
        vertices = point_cloud['xyzw'][:, :3]  # Take only the first 3 columns (x, y, z)
        # Get the RGB colors
        colors = point_cloud['rgb']  # Assuming the colors are already in uint8 format (0-255)
        # Create a Trimesh PointCloud object
        pcd_mesh = trimesh.PointCloud(vertices, colors=colors)

        # Save the point cloud to a .ply file
        pcd_mesh.export('point_cloud_trimesh_1.ply')

        print("Point cloud saved successfully using trimesh!")

        # Send request to server to generate grasps
        response = self.cgn_client.send_request(data_to_send)
        if response is None:
            logging.error("Failed to receive grasps from the server.")
            return {'status': 'No grasps received'}

        pred_grasps_cam = response['pred_grasps_cam']
        scores = response['scores']
        pointcloud = response['point_cloud']
        
        vertices = np.array(pointcloud['xyzw'])[:, :3]  # Convert to NumPy array and take only (x, y, z)
        colors = pointcloud['rgb']  # Assuming the colors are already in uint8 format (0-255)
        pcd_mesh = trimesh.PointCloud(vertices, colors=colors)
        pcd_mesh.export('point_cloud_trimesh_2.ply')

        # segmented_pc = response['segmented_pc']

        # Perform motion planning to filter out infeasible grasps
        feasible_grasps = []
        feasible_scores = []
        results_list = []

        for obj_id in pred_grasps_cam:
            grasps = [np.array(g) for g in pred_grasps_cam[obj_id]]
            obj_scores = scores[obj_id]

            for grasp_cam, score in zip(grasps, obj_scores):
                grasp_cam_real = grasp_cam

                # Debug 10cm
                # Step 1: Extract rotation and position from grasp_cam
                rot_matrix_cam = grasp_cam[:3, :3]
                pos_cam = grasp_cam[:3, 3]
                # Step 2: Compute approach direction in camera frame
                local_approach_direction = np.array([0, 0, 1])  # Adjust based on your gripper
                approach_direction_cam = rot_matrix_cam @ local_approach_direction
                # Step 3: Normalize the approach direction
                approach_direction_cam = approach_direction_cam / np.linalg.norm(approach_direction_cam)
                # Step 4: Compute the translation vector along the approach direction
                translation_vector = -0.2034 * approach_direction_cam
                # Step 5: Update the position with the translation
                pos_cam_new = pos_cam + translation_vector
                # Step 6: Construct the new grasp_cam
                grasp_cam = np.eye(4)
                grasp_cam[:3, :3] = rot_matrix_cam  # Keep the original rotation
                grasp_cam[:3, 3] = pos_cam_new  

                # # Debug 180
                # # Step 1: Extract rotation and position from grasp_cam
                # rot_matrix_cam = grasp_cam[:3, :3]
                # pos_cam = grasp_cam[:3, 3]
                # # Step 2: Compute approach direction in camera frame
                # local_approach_direction = np.array([0, 0, 1])  # Adjust based on your gripper
                # approach_direction_cam = rot_matrix_cam @ local_approach_direction
                # # Step 3: Create rotation matrix for 180-degree rotation around approach direction
                # angle_rad = np.pi  # 180 degrees in radians
                # axis = approach_direction_cam / np.linalg.norm(approach_direction_cam)
                # adjust_rot_cam = R.from_rotvec(angle_rad * axis).as_matrix()
                # # Step 4: Apply the rotation to the original rotation matrix
                # rot_matrix_cam_new = adjust_rot_cam @ rot_matrix_cam
                # # Step 5: Construct the new grasp_cam
                # grasp_cam_new = np.eye(4)
                # grasp_cam_new[:3, :3] = rot_matrix_cam_new
                # grasp_cam_new[:3, 3] = pos_cam

                # # Debug 180 real
                # grasp_cam_new_real = np.eye(4)
                # grasp_cam_new_real[:3, :3] = rot_matrix_cam_new
                # grasp_cam_new_real[:3, 3] = grasp_cam_real[:3, 3]

                # Cam frame to world frame to base frame
                grasp_world_new = np.linalg.inv(camera_extrinsic) @ grasp_cam
                # grasp_base_new = self.base_pose_inv @ grasp_world_new  # Grasp pose in the base frame
                # grasp_world_new_real = np.linalg.inv(camera_extrinsic) @ grasp_cam_new_real
                # grasp_base_new_real = self.base_pose_inv @ grasp_world_new_real  # Grasp pose in the base frame

                # Fixed 90 degree rotation
                rot = Quaternion(matrix=grasp_world_new[:3, :3], atol=1e-3, rtol=0)
                fix = Quaternion(axis=[0.0, 0.0, 1.0], degrees=-90)
                rot = rot * fix
                pos = grasp_world_new[:3, 3]
                target_pose = np.concatenate([pos, rot.elements])

                print("Hereererereeree")

                # rot_real = Quaternion(matrix=grasp_base_new_real[:3, :3], atol=1e-3, rtol=0)
                # rot_real = rot_real * fix
                # pos_real = grasp_base_new_real[:3, 3]
                # target_pose_real = np.concatenate([pos_real, rot_real.elements])

                print(f"target_pose: {target_pose}\n qpos: {robot_qpos}")
                # Plan the trajectory to the grasp pose
                # result = self.planner.plan(
                #     target_pose,
                #     robot_qpos,
                #     time_step=0.05,
                #     use_point_cloud=False,
                #     verbose=True
                # )
                result = {"target_pose": target_pose}

                print("Thererererereeree")

                # if result['status'] == 'Success':
                if True:
                    feasible_grasps.append(grasp_cam_real)
                    feasible_scores.append(float(score)) 
                    results_list.append(result)
        if not feasible_grasps:
            logging.error("No feasible grasps found after planning.")
            return {'status': 'No feasible grasps'}

        # Debug
        # print(f"feasible grasps cam: {feasible_grasps}")
        # print(f"feasible grasps planning result list: {results_list}")

        # Prepare data to send back to the server for visualization and selection
        data_to_send = {
            'command': 'visualize_and_select',
            'visualization': visual_method,
            'task': task,
            'selected_tasks': selected_tasks,
            'selected_model': selected_model,
            'data': {
                'rgb': rgb,
                'depth': depth,
                'cam_K': camera_k,
                'camera_extrinsic': camera_extrinsic,
                'grasps_cam': [g.tolist() for g in feasible_grasps],
                'scores': feasible_scores,
                'point_cloud': pointcloud, #point_cloud,  # Assuming this contains 'xyzw' and 'rgb'
                'eval_eps': eval_eps,
            }
        }
        print(rgb.shape)
        # Get the selected grasp from the server
        response = self.cgn_client.send_request(data_to_send)
        if response is None:
            logging.error("Failed to receive selected grasp from the server.")
            return {'status': 'No grasp selected'}
        
        def parse_response_for_index(response, max_num):
            # Find all numbers in the response
            matches = re.findall(r'\b\d+\b', response)
            if matches:
                # Take the last number found and convert it to an integer
                index = int(matches[-1])
                print(f"Parsed index from the last occurrence: {index}")
                
                # Set index to 0 if it is greater than 3
                if index > 3 or index > max_num:
                    return 0
                return index - 1
            else:
                # Default return if no number is found
                return 0
   
        def parse_response_for_color(response, max_num):
            # Color dictionary with indices
            color_map = {
                "red": 0,
                "green": 1,
                "blue": 2,
                "yellow": 3,
                "cyan": 4,
                "magenta": 5,
                "orange": 6,
                "purple": 7,
                "pink": 8,
                "lime": 9,
                "brown": 10,
                "white": 11
            }

            # Reverse mapping to use in the regex for reverse string search
            color_names_reversed = "|".join([name[::-1] for name in color_map.keys()])

            # Reverse the response to match reversed color names
            reversed_response = response[::-1]
            
            # Search for any of the reversed color names in the response
            match = re.search(rf'({color_names_reversed})\b', reversed_response, re.IGNORECASE)
            # color = None
            if match:
                # Get the matched color name, reverse it back, and convert to lowercase
                color = match.group(0)[::-1].lower()
                print(f"Parsed color: {color}")
                index = color_map.get(color)
                if index > max_num:
                    return 0, color
                return index, color
            else:
                return 0,"red"
    
        def extract_points(molmo_output, image_w, image_h):
            all_points = []
            
            # Primary regex pattern (simple coordinates like 0.5, 0.5)
            primary_pattern = r'\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*'
            # Secondary regex pattern (handles formats like (x: 0.5, y: 0.5))
            secondary_pattern = r'[xyXY]?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*[xyXY]?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)'

            # Try the primary pattern first
            matches = list(re.finditer(primary_pattern, molmo_output))
            
            # If no matches are found, try the secondary pattern
            if not matches:
                matches = list(re.finditer(secondary_pattern, molmo_output))

            # Process matched coordinates
            for match in matches:
                try:
                    # Extract x and y coordinates
                    point = [float(match.group(i)) for i in range(1, 3)]
                except ValueError:
                    print("Error parsing point")
                    continue
                else:
                    point = np.array(point)
                    if np.max(point) > 100:
                        # Treat as an invalid output
                        continue
                    if np.min(point) < 1:
                        point = point * np.array([image_w, image_h])
                    else:
                        point /= 100.0
                        point = point * np.array([image_w, image_h])
                    all_points.append(point)

            return all_points[0] if all_points else np.array([240., 320.])


        def find_nearest_point_from_single_point(coordinates_2D, generated_point, image_np, path):
            # Debug'
            print(f"coordinates_2D type: {type(coordinates_2D)}")    # List
            print(f"generated_point type: {type(generated_point)}")  # numpy.ndarray     
            print(f"--> coordinates_2D: {coordinates_2D}")           # e.g. [(319, 260), (306, 256), (288, 255)]
            print(f"--> generated_point: {generated_point}")         # e.g. [224. 264.]

            distances = np.linalg.norm(coordinates_2D - generated_point, axis=1)
            nearest_index = np.argmin(distances)

            nearest_point = coordinates_2D[nearest_index]
            image_np_with_circle = cv2.circle(image_np, (nearest_point[0], nearest_point[1]), radius=5, color=(0, 0, 255), thickness=-1)
            image_with_circle = Image.fromarray(image_np_with_circle)
            image_with_circle.save(path, format='PNG')
            print(f"Image saved at {path}")
            
            return nearest_index

        def encode_image_to_base64(image_path):
            encoded_string = base64.b64encode(image_path.getvalue()).decode('utf-8')
            return encoded_string
        
        def call_gemini(images, text):
            genai.configure(api_key="")
            model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
            
            image_parts = []
            for image_bytes_io in images:
                image_bytes_io.seek(0)  # Reset the stream to the beginning
                img_byte_arr = image_bytes_io.read()  # Read the image bytes from BytesIO
                image_parts.append({
                    "mime_type": "image/png",  # Assumes PNG, as per the image preparation. Adjust if needed.
                    "data": img_byte_arr
                })

            contents = [{"role": "user", "parts": [text, *image_parts]}]

            response = model.generate_content(contents)
            return {"response": response.text}

        def call_vlm(selected_model, host, image: str, text: str, max_retries=100) -> str:
            image_base64 = [encode_image_to_base64(image)]
            text = text + "\nHere is the image:\n"

            if selected_model == "gpt":
                pass
            elif selected_model == "gemini":
                return call_gemini([image], text)
            else:
                data = {
                    "image_base64": image_base64,  # base64 encoded image
                    "text": text,
                    "model_name": selected_model
                }

                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"Attempt {attempt}: Sending request to {host}")
                        res = requests.post(f"{host}/generate", json=data)
                        
                        # Check if request was successful
                        if res.status_code == 200:
                            print("Request successful")
                            return res.json()
                        else:
                            logging.error(f"Attempt {attempt} failed with status code {res.status_code}: {res.text}")
                    
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Attempt {attempt} failed with exception: {e}")
                    
                    # Optional: Add a short delay before retrying
                    time.sleep(1)
                
                # If all attempts fail, raise an exception
                raise Exception("Failed to call VLM after multiple attempts.")

        def call_vlm_multiple_image(selected_model, host, image_list, text, max_retries=100):
            
            num_images = len(image_list)
            images = [encode_image_to_base64(image) for image in image_list]
            text = text + f"\nHere are the {num_images} images:\n"

            if selected_model == "gpt":
                pass
            elif selected_model == "gemini":
                return call_gemini(image_list, text)
            else:
                # Prepare the data payload with the number of images
                data = {
                    "text": text,
                    "image_base64": images,  # base64 encoded images
                    "model_name": selected_model
                }

                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"Attempt {attempt}: Sending request to {host}")
                        res = requests.post(f"{host}/generate", json=data)
                        
                        # Check if request was successful
                        if res.status_code == 200:
                            print("Request successful")
                            return res.json()
                        else:
                            logging.error(f"Attempt {attempt} failed with status code {res.status_code}: {res.text}")
                    
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Attempt {attempt} failed with exception: {e}")
                    
                    # Optional: Add a short delay before retrying
                    time.sleep(1)
                
                # If all attempts fail, raise an exception
                raise Exception("Failed to call VLM after multiple attempts.")

        img = response['img']
        coordinates_2D = response['coordinates_2D']
        original_index_list = response['original_index_list']
        color_list = response['color_list']
        visualization = response['visualization']
        # task = "Pick up the tool or cutlery, and hand it to the user in the easiest way for them to grasp and use."
        host = "http://172.28.177.20:8000"

        num_images = len(original_index_list)
        assert num_images == len(color_list)

        index_list = ["1", "1 or 2", "1, 2 or 3", "1, 2, 3 or 4", "1, 2, 3, 4 or 5", "1, 2, 3, 4, 5 or 6", "1, 2, 3, 4, 5, 6 or 7", "1, 2, 3, 4, 5, 6, 7 or 8", "1, 2, 3, 4, 5, 6, 7, 8 or 9", "1, 2, 3, 4, 5, 6, 7, 8, 9 or 10", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10 or 11", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 or 12"]
        rgb_list = ["red", "red or green", "red, green or blue", "red, green, blue or yellow", "red, green, blue, yellow or cyan", "red, green, blue, yellow, cyan or magenta", "red, green, blue, yellow, cyan, magenta or orange", "red, green, blue, yellow, cyan, magenta, orange or purple", "red, green, blue, yellow, cyan, magenta, orange, purple or pink", "red, green, blue, yellow, cyan, magenta, orange, purple, pink or lime", "red, green, blue, yellow, cyan, magenta, orange, purple, pink, lime or brown", "red, green, blue, yellow, cyan, magenta, orange, purple, pink, lime, brown or white"]
        rgb_list2 = ["red", "red and green", "red, green and blue", "red, green, blue and yellow", "red, green, blue, yellow and cyan", "red, green, blue, yellow, cyan and magenta", "red, green, blue, yellow, cyan, magenta and orange", "red, green, blue, yellow, cyan, magenta, orange and purple", "red, green, blue, yellow, cyan, magenta, orange, purple and pink", "red, green, blue, yellow, cyan, magenta, orange, purple, pink and lime", "red, green, blue, yellow, cyan, magenta, orange, purple, pink, lime and brown", "red, green, blue, yellow, cyan, magenta, orange, purple, pink, lime, brown and white"]
        
        
        # /home/stevenliudw/jiaming/LIPP/results/molmo/task_5a_0_contact_one/images/eps_0/eps0_grasp_0.png
        output_dir = f'./results/{selected_model}/{selected_tasks}_{visualization}/images/eps_{eval_eps}'
        os.makedirs(output_dir, exist_ok=True)


        if visualization == "gripper_one":
            # Convert images to bytes and then to base64 format
            image_bytes_list = []
            for image in img:
                if isinstance(image, io.BytesIO):
                    image_bytes_list.append(image)
                else:
                    # Convert PIL Image to BytesIO and save it
                    image_bytes = io.BytesIO()  # Create an in-memory bytes buffer
                    image.save(image_bytes, format='PNG')  # Save the PIL image to the buffer
                    image_bytes_list.append(image_bytes)  # Store the bytes object
            text_mul_img_percentange_com = f"""You are an intelligent system that can specify the best grasping point for a robot to pick up an object from multiple images.
            You are given multiple images, each showing the object with a different grasp candidate highlighted in red, and a task description.
            Your task is to analyze the grasp candidates in each image and choose the index of the grasp that would best accomplish the task.
            Note that there may be multiple valid grasping points, but you should select the one that is most suitable for the given task.
            Consider factors like stability, object orientation, and the requirements of the task in your decision.
            The task is: {task}
            Explain your reasoning clearly, then state the chosen grasp index ({index_list[num_images-1]}) at the end of your response."""

            text_highest_suc_rate = f"""You are an intelligent system tasked with selecting the best grasp for a robot from multiple images. 
            Each image presents a different grasp candidate. Your goal is to choose the grasp that offers the highest success rate, ensuring it is robust and stable for the robot to pick up the object securely. 
            Consider factors such as stability, and likelihood of a successful grasp. 
            Explain your reasoning clearly, then state the chosen grasp index ({index_list[num_images-1]}) at the end of your response."""
            print(selected_model)
            res = call_vlm_multiple_image(selected_model, host, image_bytes_list, text_mul_img_percentange_com)
            print(text_mul_img_percentange_com)
            print(res['response'])
            index_sorted = parse_response_for_index(res['response'], num_images)
            index_original = original_index_list[index_sorted]

            chosen_image_bytes = image_bytes_list[index_sorted]
            chosen_image = Image.open(chosen_image_bytes)  # Convert BytesIO to PIL Image
            chosen_image_path = os.path.join(output_dir, "chosen_image.png")
            os.makedirs(os.path.dirname(chosen_image_path), exist_ok=True)

            chosen_image.save(chosen_image_path, format='PNG')

            # Save the JSON file with parsed and original response
            response_data = {
                "prompts": text_mul_img_percentange_com,
                "original_response": res['response'],
                "parsed_response": index_sorted
            }
            response_json_path = os.path.join(output_dir, "chosen_response.json")
            os.makedirs(os.path.dirname(response_json_path), exist_ok=True)

            with open(response_json_path, 'w') as json_file:
                json.dump(response_data, json_file, indent=4)

            print(f"Saved chosen image at {chosen_image_path}")
            print(f"Saved response JSON at {response_json_path}")

            # index_original = original_index_list[0]
        
        elif visualization == "gripper_all":
            image = img[0]
            text_img_gripper_one_repeat = f"""You are an intelligent system that can specify the best grasping point for a robot to pick up an object from a single image.
            You are given an image showing the object with different grasp candidates highlighted in {rgb_list2[num_images-1]}.
            Your task is to analyze these grasp candidates based on the task description and choose the best one.
            Note that there may be multiple valid grasping points, but you should select the one that is most suitable for the given task.
            Consider factors like stability, object orientation, and the requirements of the task in your decision.
            The task is: {task}
            Explain your reasoning clearly, then repeat the chosen color({rgb_list[num_images-1]}) at the end of your response."""
            
            text_highest_suc_rate = f"""You are an intelligent system tasked with selecting the best grasp for a robot from one image. 
            You are given an image showing the object with different grasp candidates highlighted in {rgb_list2[num_images-1]}.
            Your goal is to choose the grasp that offers the highest success rate, ensuring it is robust and stable for the robot to pick up the object securely. 
            Consider factors such as stability, and likelihood of a successful grasp. 
            Explain your reasoning clearly, then repeat the chosen color({rgb_list[num_images-1]}) at the end of your response."""
            
            res = call_vlm(selected_model, host, image, text_img_gripper_one_repeat)
            print(text_img_gripper_one_repeat)
            print(res['response'])
            index_sorted,parsed_color = parse_response_for_color(res['response'], num_images)
            index_original = original_index_list[index_sorted]

            # Save the JSON file with parsed and original response
            response_data = {
                "prompts": text_img_gripper_one_repeat,
                "original_response": res['response'],
                "parsed_response": parsed_color
            }
            response_json_path = os.path.join(output_dir, "chosen_response.json")
            os.makedirs(os.path.dirname(response_json_path), exist_ok=True)

            with open(response_json_path, 'w') as json_file:
                json.dump(response_data, json_file, indent=4)

            print(f"Saved response JSON at {response_json_path}")

            # index_original = original_index_list[0]

        elif visualization == "contact_one":
            image_bytes_list = []
            for image in img:
                if isinstance(image, io.BytesIO):
                    image_bytes_list.append(image)
                else:
                    # Convert PIL Image to BytesIO and save it
                    image_bytes = io.BytesIO()  # Create an in-memory bytes buffer
                    image.save(image_bytes, format='PNG')  # Save the PIL image to the buffer
                    image_bytes_list.append(image_bytes)  # Store the bytes object
            text_mul_img_percentange_com = f"""You are an intelligent system that can specify the best grasping point for a robot to pick up an object from a set of images.
            You are given multiple images, each showing the object with a different grasp candidate with a red dot, and a task description.
            Your task is to analyze the grasp candidates in each image and choose the index of the grasp that would best accomplish the task.
            Note that there may be multiple valid grasping points, but you should select the one that is most suitable for the given task.
            Consider factors like stability, object orientation, and the requirements of the task in your decision.
            The task is: {task}
            Explain your reasoning clearly, then state the chosen grasp index ({index_list[num_images-1]}) at the end of your response."""

            text_highest_suc_rate = f"""You are an intelligent system tasked with selecting the best grasp for a robot from multiple images. 
            Each image presents a different grasp candidate (represented as a colored dot). Your goal is to choose the grasp that offers the highest success rate, ensuring it is robust and stable for the robot to pick up the object securely. 
            Consider factors such as stability, and likelihood of a successful grasp. 
            Explain your reasoning clearly, then state the chosen grasp index ({index_list[num_images-1]}) at the end of your response."""


            res = call_vlm_multiple_image(selected_model, host, image_bytes_list, text_mul_img_percentange_com)
            print(text_mul_img_percentange_com)
            print(res['response'])
            index_sorted = parse_response_for_index(res['response'], num_images)
            index_original = original_index_list[index_sorted]

            chosen_image_bytes = image_bytes_list[index_sorted]
            chosen_image = Image.open(chosen_image_bytes)  # Convert BytesIO to PIL Image
            chosen_image_path = os.path.join(output_dir, "chosen_image.png")
            os.makedirs(os.path.dirname(chosen_image_path), exist_ok=True)

            chosen_image.save(chosen_image_path, format='PNG')

            # Save the JSON file with parsed and original response
            response_data = {
                "prompts": text_mul_img_percentange_com,
                "original_response": res['response'],
                "parsed_response": index_sorted
            }
            response_json_path = os.path.join(output_dir, "chosen_response.json")
            os.makedirs(os.path.dirname(response_json_path), exist_ok=True)

            with open(response_json_path, 'w') as json_file:
                json.dump(response_data, json_file, indent=4)

            print(f"Saved chosen image at {chosen_image_path}")
            print(f"Saved response JSON at {response_json_path}")

            # index_original = original_index_list[0]

        elif visualization == "contact_all":
            image = img[0]
            # image = "/home/stevenliudw/jiaming/Server/molmo-serve-master/data/gripper_all/eps1_grasp_0.png"
            # print(image[0])
            text_img_gripper_one_repeat = f"""You are an intelligent system that can specify the best grasping point for a robot to pick up an object from a single image.
            You are given an image showing the object with different grasp candidates highlighted in {rgb_list2[num_images-1]} dots.
            Your task is to analyze these grasp candidates based on the task description and choose the best one.
            Note that there may be multiple valid grasping points, but you should select the one that is most suitable for the given task.
            Consider factors like stability, object orientation, and the requirements of the task in your decision.
            The task is: {task}
            Explain your reasoning clearly, then repeat the chosen color({rgb_list[num_images-1]}) at the end of your response."""
            
            text_highest_suc_rate = f"""You are an intelligent system tasked with selecting the best grasp for a robot from from one image. 
            You are given an image showing the object with different grasp candidates (represented as colored dots) highlighted in {rgb_list2[num_images-1]}.
            Your goal is to choose the grasp that offers the highest success rate, ensuring it is robust and stable for the robot to pick up the object securely. 
            Consider factors such as stability, and likelihood of a successful grasp. 
            Explain your reasoning clearly, then repeat the chosen color({rgb_list[num_images-1]}) at the end of your response."""
            
            res = call_vlm(selected_model, host, image, text_img_gripper_one_repeat)
            print(text_img_gripper_one_repeat)
            print(res['response'])
            index_sorted, parsed_color = parse_response_for_color(res['response'], num_images)

            index_original = original_index_list[index_sorted]

            # Save the JSON file with parsed and original response
            response_data = {
                "prompts": text_img_gripper_one_repeat,
                "original_response": res['response'],
                "parsed_response": parsed_color
            }
            response_json_path = os.path.join(output_dir, "chosen_response.json")
            os.makedirs(os.path.dirname(response_json_path), exist_ok=True)

            with open(response_json_path, 'w') as json_file:
                json.dump(response_data, json_file, indent=4)

            # print(f"Saved chosen image at {chosen_image_path}")
            print(f"Saved response JSON at {response_json_path}")

            # index_original = original_index_list[0]


        elif visualization == "distance":
            print(f"coordinates_2D: {coordinates_2D}")
            image = img[0]
            image_bytes = io.BytesIO()  # Create an in-memory bytes buffer
            image.save(image_bytes, format='PNG')  # Save the PIL image to the buffer

            save_dir = f"./results/{selected_model}/{selected_tasks}_{visual_method}/images/eps_{eval_eps}/"
            # Debug: Save for verification
            save_path = os.path.join(save_dir, f"eps_{eval_eps}_for_VLM.png")  # Update with your desired path
            image.save(save_path, format='PNG')
            print(f"Image saved at {save_path}")

            print(image)
            text_percentage = f"""You are an intelligent system tasked with predicting the best grasp for a robot from from one image.
            Given an image of the scene and a task description, 
            you should provide a point on the image where the robot should grasp the object to best accomplish the task. 
            Note that there may be multiple valid grasping points, but you should select the one that is most suitable for the given task.
            Consider factors like stability, object orientation, and the requirements of the task in your decision.
            The task is: {task}
            Explain your reasoning clearly, then repeat the answer with a point coordintate in percentage(eg. (x,y)) on the image at the end of your response."""
            
            text_highest_suc_rate = f"""You are an intelligent system tasked with predicting the best grasp for a robot from from one image. 
            Given an image of the scene, you should provide a grasping point on the image that offers the highest grasping success rate, ensuring it is robust and stable for the robot to pick up the object securely. 
            Consider factors such as stability, and likelihood of a successful grasp. 
            Explain your reasoning clearly, then repeat the answer with a point coordintate in percentage(eg. (x,y)) on the image at the end of your response."""
            
            res = call_vlm(selected_model, host, image_bytes, text_percentage)
            print(text_percentage)
            print(res['response'])
            # np.array(Image.open(path).convert('RGB'))
            image = np.array(Image.open(save_path).convert('RGB'))
            generated_point = extract_points(res['response'], 640, 480)
            index_sorted = find_nearest_point_from_single_point(coordinates_2D, generated_point, image, os.path.join(save_dir, f"eps_{eval_eps}_VLM_chosen.png"))
            index_original = original_index_list[index_sorted]


            response_data = {
                "prompts": text_percentage,
                "original_response": res['response'],
                "parsed_response": generated_point.tolist()  # Convert ndarray to list
            }
            response_json_path = os.path.join(output_dir, "chosen_response.json")
            os.makedirs(os.path.dirname(response_json_path), exist_ok=True)

            with open(response_json_path, 'w') as json_file:
                json.dump(response_data, json_file, indent=4)

            # print(f"Saved chosen image at {chosen_image_path}")
            print(f"Saved response JSON at {response_json_path}")
            # index_original = original_index_list[0]


        selected_grasp_idx = np.array(index_original)
        final_result = results_list[selected_grasp_idx]
        # Transform the selected grasp to world coordinates
        grasp_world = np.linalg.inv(camera_extrinsic) @ feasible_grasps[selected_grasp_idx]
        # grasp_base = self.base_pose_inv @ grasp_world  # Grasp pose in the base frame

        # if final_result['status'] == 'Success':
        if True:
            final_result['grasp_pose_world'] = grasp_world
            final_result['grasp_pose_base'] = grasp_world
            return final_result
        else:
            logging.error(f"Planning failed for selected grasp: {final_result['status']}")
            return {'status': 'Planning failed for selected grasp'}
