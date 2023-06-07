"""
SPA policy using Contact GraphNet

https://github.com/NVlabs/contact_graspnet.git
"""

from typing import Any
import gym
import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.sapien_utils import look_at
import numpy as np
import mplib
from pyquaternion import Quaternion
from spa.utils import get_rotation_between_vecs, plot_img
from spa.cgn_agent import CGNAgent

import os
import sys
import envs
from envs.pick_only import RGBDPCObservationWrapper
import json
def main(exp_name):

    ############### setup env #####################
    pose = look_at([1, -1, 0.5], [0, 0, 0])
    camera_cfg = {
        'add_segmentation': True,
        'width': 640,
        'height': 480,
        }
    # control_mode = "pd_ee_delta_pose" # used for RL
    control_mode = "pd_joint_pos_vel" # so that we can use motion planner to drive the robot, [q(7dim), qdot(7dim)]
    env = gym.make("PicOnlykYCB-v0",
                obs_mode='image', 
                camera_cfgs=camera_cfg, 
                control_mode=control_mode) # this is the best performing mode for RL
    env = RGBDPCObservationWrapper(env)
    env = RecordEpisode(
                    env,
                    f'results/{exp_name}/videos',
                    save_trajectory=False,
                    save_video=True,
                    info_on_video=True,
                    # render_mode="cameras",
                )

    env.seed(0)  # specify a seed for randomness
    eval_eps = 500
    max_steps = 1000
    ############### setup env #####################


    ############### main loop #####################
    success = []
    failed_count = {}
    failed_eps = {}
    with CGNAgent(env) as agent:
        for eps in range(eval_eps):
            obs = env.reset()
            agent.reset()
            for step in range(max_steps):    
                plan = agent.act(obs)
                if plan['info'] != 'Success':
                    failed_count[plan['info']] = failed_count.get(plan['info'], 0) + 1
                    failed_eps[plan['info']] = failed_eps.get(plan['info'], []) + [eps]

                    break
                obs, reward, done, info = env.step(plan['action'])
                # print("step: ", step, "reward: ", reward, "done: ", done, "success: ", info['success'])
                
                if done:
                    break
            print("episode: ", eps, "reward: ", reward, "done: ", done, "success: ", info['success'])
            if info['success']:
                success.append(eps)
    
    print("success rate: ", len(success)/eval_eps)
    ############### main loop #####################
    stat = {
        "success": success,
        "failed_count": failed_count,
        "failed_eps": failed_eps,
        "success_rate": len(success)/eval_eps,
    }
    with open(f"results/{exp_name}/stat.json", 'w') as f:
        json.dump(stat, f)
    env.close()
    
    
if __name__ == "__main__":
    main('filter_then_select_col')


        
        