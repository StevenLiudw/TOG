import os
import json
import requests
from PIL import Image
import base64
import time
import logging
import re

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def parse_grasp_color(text):
    # Define patterns in the specified order of priority
    patterns = [
        r"red, green or blue",   # Look for "red, green or blue" first
        r"red or green",         # Then look for "red or green"
        r"red"                   # Finally, look for "red"
    ]
    
    # Iterate over patterns and return the first match found
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
    
    # If No pattern is found, return "colored" as default
    return "colored"
def call_vlm(selected_model, host, image, text, max_retries=100):
    data = {
        "image_base64": [encode_image_to_base64(image)],
        "text": text,
        "model_name": selected_model
    }
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(f"{host}/generate", json=data)
            if res.status_code == 200:
                return res.json()
            else:
                logging.error(f"Attempt {attempt} failed with status code {res.status_code}: {res.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt} failed with exception: {e}")
        time.sleep(1)
    raise Exception("Failed to call VLM after multiple attempts.")

def call_vlm_multiple_image(selected_model, host, image_list, text, max_retries=100):
    images = [encode_image_to_base64(image) for image in image_list]
    data = {
        "image_base64": images,
        "text": text,
        "model_name": selected_model
    }
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(f"{host}/generate", json=data)
            if res.status_code == 200:
                return res.json()
            else:
                logging.error(f"Attempt {attempt} failed with status code {res.status_code}: {res.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt} failed with exception: {e}")
        time.sleep(1)
    raise Exception("Failed to call VLM after multiple attempts.")

def parse_vlm_response(response_text):
    """
    Parses the VLM response text to extract the final "Yes" or "No" answer.
    - Searches from the end of the response for the first occurrence of "Yes" or "No" possibly preceded by punctuation.
    - Returns "Cannot parse" if No clear answer is found.
    """
    response_text = response_text.strip().lower()[::-1]  # Normalize case, trim whitespace, and reverse text
    
    # Regular expression to find "Yes" or "No" at the start of the reversed text, preceded by optional punctuation
    match = re.search(r"^[\s.]*?(sey|on)\b", response_text)
    if match:
        # Reverse back the matched result to get "Yes" or "No" in the correct order
        return match.group(1)[::-1].capitalize()
    
    return "Cannot parse"

# Task descriptions
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
# Main function
def main(base_folder, selected_model, host):
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        split_path = subfolder.split("_")
        if len(split_path) < 3:
            print(f"Skipping folder {subfolder_path} due to unexpected name format.")
            continue
        
        task_name = "_".join(split_path[:2])
        try:
            subtask_index = int(split_path[2])
        except ValueError:
            print(f"Skipping folder {subfolder_path} as subtask index is not an integer.")
            continue
        evaluation_type = "_".join(split_path[3:])
        
        print(f"Processing Task: {task_name}, Subtask Index: {subtask_index}, Evaluation Type: {evaluation_type}")
        
        task_description = tasks.get(task_name, [])[subtask_index]
        image_folder = os.path.join(subfolder_path, "images")
        
        if not os.path.exists(image_folder):
            print(f"No images folder found in {subfolder_path}")
            continue
        
        for eps_folder in os.listdir(image_folder):
            eps_path = os.path.join(image_folder, eps_folder)
            parent_folder_name = eps_folder.replace("_", "")
            response_path = os.path.join(eps_path, "chosen_response.json")
            
            if not os.path.exists(response_path):
                continue
            
            with open(response_path, "r") as f:
                # Check if file is empty
                if os.stat(response_path).st_size == 0:
                    print("Error: The file is empty.")
                    chosen = None
                    original_res = None
                else:
                    try:
                        data = json.load(f)
                        chosen = data.get("parsed_response")
                        original_res = data.get("prompts")
                    except json.JSONDecodeError:
                        print("Error: Failed to decode JSON. The file may be corrupted or contain invalid JSON.")
                        chosen = None
                        original_res = None

            # Optionally, add logic to handle the case when chosen or original_res is None
            if chosen is None or original_res is None:
                print("Failed to retrieve 'parsed_response' or 'prompts' from the JSON data.")

            if evaluation_type == "contact_all":
                image_name = f"{parent_folder_name}_grasp_0.png"
                image_path = os.path.join(eps_path, image_name)
                parsed_original_res = parse_grasp_color(original_res)

                prompt = (
                    f"The robot is considering multiple grasp candidates to achieve the task.\n"
                    f"The task description is: '{task_description}'.\n"
                    f"Each of the colored dots (highlighted in {parsed_original_res}) shows a different grasp candidate."
                    f"To accomplish the given task effectively, the robot has chosen to grasp the object at a point highlighted in {chosen}.\n"
                    f"Please evaluate if this chosen grasp is the most appropriate one among the candidates for effectively accomplishing the task.\n"
                    f"Explain your reasoning clearly, then conclude with 'Yes' or 'No' at the end of your response."
                )
                print(prompt)
                response = call_vlm(selected_model, host, image_path, prompt)
                print(response)
            elif evaluation_type == "gripper_all":
                image_name = f"{parent_folder_name}_grasp_0.png"
                image_path = os.path.join(eps_path, image_name)
                parsed_original_res = parse_grasp_color(original_res)

                prompt = (
                    f"The robot is considering multiple grasp candidates to achieve the task.\n"
                    f"The task description is: '{task_description}'.\n"
                    f"Each of the colored grip poses (highlighted in {parsed_original_res}) shows a different grasp candidate."
                    f"To accomplish the given task effectively, the robot has chosen to grasp the object at a grip pose highlighted in {chosen}.\n"
                    f"Please evaluate if this chosen grip pose is the most appropriate one among the candidates for effectively accomplishing the task.\n"
                    f"Explain your reasoning clearly, then conclude with 'Yes' or 'No' at the end of your response."
                )
                print(prompt)
                response = call_vlm(selected_model, host, image_path, prompt)
                print(response)
                
            elif evaluation_type == "contact_one":
                image_paths = []
                num_image = 0
                for i in range(3):
                    image_name = f"{parent_folder_name}_grasp_{i}.png"
                    image_path = os.path.join(eps_path, image_name)
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        num_image += 1
                chosen_image_path = os.path.join(eps_path, "chosen_image.png")
                if image_paths and os.path.exists(chosen_image_path):
                    # Add chosen image as the last image in the list
                    image_paths.append(chosen_image_path)

                    # Updated prompt for multiple candidates and the final chosen grasp
                    prompt = (
                        f"The robot is considering multiple grasp candidates to achieve the task.\n"
                        f"The task description is: '{task_description}'.\n"
                        f"Each of the first {num_image} images shows a different grasp candidate highlighted in red dot, and the last image shows the chosen grasp (also highlighted in red).\n"
                        f"Please evaluate if this chosen grasp is the most appropriate one among the candidates for effectively accomplishing the task.\n"
                        f"Explain your reasoning clearly, then conclude with 'Yes' or 'No' at the end of your response."
                    )
                print(prompt)
                response = call_vlm_multiple_image(selected_model, host, image_paths, prompt)
                print(response)

            elif evaluation_type == "gripper_one":
                image_paths = []
                num_image = 0
                for i in range(3):
                    image_name = f"{parent_folder_name}_grasp_{i}.png"
                    image_path = os.path.join(eps_path, image_name)
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        num_image += 1
                chosen_image_path = os.path.join(eps_path, "chosen_image.png")
                if image_paths and os.path.exists(chosen_image_path):
                    # Add chosen image as the last image in the list
                    image_paths.append(chosen_image_path)

                    # Updated prompt for multiple candidates and the final chosen grasp
                    prompt = (
                        f"The robot is considering multiple grasp candidates to achieve the task.\n"
                        f"The task description is: '{task_description}'.\n"
                        f"Each of the first {num_image} images shows a different grasp candidate's pose highlighted in red, and the last image shows the chosen grasp (also highlighted in red).\n"
                        f"Please evaluate if this chosen grasp is the most appropriate one among the candidates for effectively accomplishing the task.\n"
                        f"Explain your reasoning clearly, then conclude with 'Yes' or 'No' at the end of your response."
                    )
                print(prompt)
                response = call_vlm_multiple_image(selected_model, host, image_paths, prompt)
                print(response)

            elif evaluation_type == "distance":
                image_name = f"{parent_folder_name}_VLM_chosen.png"
                image_path = os.path.join(eps_path, image_name)
                
                if os.path.exists(chosen_image_path):
                    prompt = (
                        f"The robot has autonomously chosen a grasp point on the object in the image to accomplish the task.\n"
                        f"The task description is: '{task_description}'.\n"
                        f"Please evaluate if the robot's chosen grasp point is appropriate and fulfills the task requirements.\n"
                        f"Explain your reasoning clearly, then conclude with 'Yes' or 'No' at the end of your response."
                    )
                    print(prompt)
                    response = call_vlm(selected_model, host, chosen_image_path, prompt)
                    print(response)
            else:
                print("No such visualization method!!!!!!!")
                continue

            evaluation_result = parse_vlm_response(response["response"])
            print(evaluation_result)
            evaluation_data = {
                "prompt": prompt,
                "original_response": response["response"],
                "evaluation_result": evaluation_result
            }
            eval_response_path = os.path.join(eps_path, "evaluation_response_72b.json")
            with open(eval_response_path, "w") as f:
                json.dump(evaluation_data, f, indent=4)
            print(f"Saved evaluation response in {eval_response_path}")

if __name__ == "__main__":
    main("./all_images/qwen_7B", "qwen_72B", "http://172.28.177.20:8000")








