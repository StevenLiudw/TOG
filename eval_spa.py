import torch
import gym
import json
from spa.cgn_agent import CGNAgent
from mani_skill2.utils.wrappers import RecordEpisode
from envs.pick_only import RGBDPCObservationWrapper
import argparse
import gc
import os
import datetime
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.2, torch.cuda.current_device())
visual_methods = ["gripper_one", "gripper_all", "contact_one", "contact_all", "distance"]
# visual_methods = ["contact_one"]


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
num_tasks = {
    "task_1": 8,
    "task_2": 6,
    "task_3": 9,
    "task_4": 4,
    "task_5a": 1,
    "task_5b": 1
}

def main(selected_model, selected_tasks, task, visual_method, exp_name, random_seed, task_name):
    # Generate a timestamp for the results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    ############### setup env #####################
    # Configure the camera, including the pose
    camera_cfg = {
        'add_segmentation': True,
        'width': 640,
        'height': 480,
    }

    # Use the correct environment ID
    control_mode = "pd_joint_pos_vel"
    env = gym.make(
        "PicOnlykYCB-v0",
        obs_mode='image',
        camera_cfgs=camera_cfg,
        control_mode=control_mode
    )
    env = RGBDPCObservationWrapper(env)
    # Update the results directory path to include the timestamp
    results_dir = f'results_{selected_tasks}_{timestamp}/{exp_name}/{selected_model}/{selected_tasks}_{visual_method}'
    os.makedirs(results_dir, exist_ok=True)
    # Update the env to use the new results directory
    env = RecordEpisode(
        env,
        os.path.join(results_dir, 'videos'),
        save_trajectory=False,
        save_video=True,
        info_on_video=True,
    )

    env.seed(random_seed)  # specify a seed for randomness
    eval_eps = num_tasks[task_name]
    eval
    max_steps = 1000
    ############### setup env #####################

    # Define a unique checkpoint file name
    checkpoint_file = f"checkpoint_{selected_model}_{selected_tasks}_{visual_method}.json"

    # Initialize the accumulated results
    success = []
    name = []
    failed_count = {}
    failed_eps = {}

    last_completed_episode = -1

    ############### main loop #####################
    with CGNAgent(env) as agent:
        for eps in range(last_completed_episode + 1, eval_eps):
            obs, nameid = env.reset(mdid=eps)
            agent.reset()
            reward = 0  # Initialize reward
            done = False  # Initialize done
            info = {'success': False}  # Initialize info
            is_success = True
            for step in range(max_steps):
                plan = agent.act(selected_model, selected_tasks, task, visual_method, obs, eps)
                if plan['info'] != 'Success':
                    is_success = False
                    failed_count[plan['info']] = failed_count.get(plan['info'], 0) + 1
                    failed_eps[plan['info']] = failed_eps.get(plan['info'], []) + [eps]
                    break
                obs, reward, done, info = env.step(plan['action'])

                if done:
                    break
            if is_success:
                name.append(info['name'])
            else:
                name.append(nameid)
            print("model: ", selected_model, "episode: ", eps, "task: ", task, "name: ", name[-1], "visual_method: ", visual_method,
                  "reward: ", reward, "done: ", done, "success: ", info['success'])
            if info['success']:
                success.append(eps)

            # Save the current episode number and accumulated results to the checkpoint file
            checkpoint_data = {
                'last_completed_episode': eps,
                'success': success,
                'name': name,
                'failed_count': failed_count,
                'failed_eps': failed_eps
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)

            # Clear GPU cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # After completing all episodes, delete the checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    success_rate = len(success) / eval_eps
    print(f"Success rate for task '{task}' with visualization '{visual_method}': ", success_rate)
    ############### main loop #####################
    stat = {
        "model": selected_model,
        "task": task,
        "name": name,
        "success": success,
        "failed_count": failed_count,
        "failed_eps": failed_eps,
        "success_rate": success_rate,
    }
    with open(f"{results_dir}/{selected_model}_{selected_tasks}_{visual_method}_results.json", 'w') as f:
        json.dump(stat, f)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tasks for robot.")
    parser.add_argument("--task", required=True, choices=tasks.keys(), help="Task name to run (e.g., task_1, task_2, etc.)")
    parser.add_argument("--task_idx", type=int, required=True, help="Task index to run")
    parser.add_argument("--visual_method", required=True, choices=visual_methods, help="Visual method to use")
    parser.add_argument("--model", required=True, choices=["molmo", "qwen_7B", "qwen_72B", "gemini"], help="Model name to use (molmo, qwen_7B, qwen_72B)")

    args = parser.parse_args()

    selected_tasks = args.task
    idx = args.task_idx
    t = tasks[selected_tasks][idx]
    model = args.model
    visual_method = args.visual_method
    random_seed = 2

    # Create a unique identifier for each task variant
    task_variant_name = f"{selected_tasks}_{idx}"
    main(model, task_variant_name, t, visual_method, 'filter_then_select_col_cam', random_seed, selected_tasks)
