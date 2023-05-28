"""
SPA policy using Contact GraphNet

https://github.com/NVlabs/contact_graspnet.git
"""

from typing import Any
import gym
import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
import matplotlib.pyplot as plt
from mani_skill2.utils.sapien_utils import look_at
import socket
import numpy as np


def plot_img(img, title=None):
    plt.figure(figsize=(10,6))
    if title is not None:
        plt.title(title)
    plt.imshow(img)


import os
import sys
# assumes the cwd is LIPP
BASE_DIR = os.path.abspath(
    os.path.join(os.getcwd(), "third_party/contact_graspnet")
)
sys.path.append(BASE_DIR)
from contact_graspnet.utils.ipc import CGNClient

def main():
    pose = look_at([1, -1, 0.5], [0, 0, 0])

    camera_cfg = {
        'add_segmentation': True,
        'width': 640,
        'height': 480,
        }
    env = gym.make("PickSingleYCB-v0",
                obs_mode='rgbd', 
                camera_cfgs=camera_cfg, 
                control_mode="pd_ee_delta_pose") # this is the best performing mode for RL

    env = RecordEpisode(
                    env,
                    'videos',
                    save_trajectory=False,
                    save_video=True,
                    info_on_video=True,
                    # render_mode="cameras",
                )

    env.seed(0)  # specify a seed for randomness
    obs = env.reset()
    done = False
    max_steps = 10

    camera_k = obs['camera_param']['base_camera']['intrinsic_cv']
    # camera_k = camera_k.flatten()
    camera_pose = obs['camera_param']['base_camera']['extrinsic_cv']

    with CGNClient() as client:

        for step in range(max_steps):    
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            rgb = obs['image']['base_camera']['rgb'] # h, w, 3
            depth = obs['image']['base_camera']['depth'] # h, w, 1
            seg = obs['image']['base_camera']['Segmentation'][:,:,1:2] # h, w ,1, this is the actor-level segmentation
            grasps, scores = client.get_grasps([rgb, depth, seg, camera_k])

            if done:
                print("Episode finished after {} timesteps".format(step + 1))
                break
        env.close()
    
    
if __name__ == "__main__":
    main()
    # with CGNClient() as client:
    #     data = np.array([1,2,3,4,5,6])
    #     re = client.get_grasps(data)
    #     print(re)

    #     re = client.get_grasps(data)
    #     print(re)

        
        