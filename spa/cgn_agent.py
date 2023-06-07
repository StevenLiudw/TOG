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

import os
import sys

# # assumes the cwd is LIPP
# BASE_DIR = os.path.abspath(
#     os.path.join(os.getcwd(), "third_party/contact_graspnet")
# )
# sys.path.append(BASE_DIR)
# from contact_graspnet.utils.ipc import CGNClient

from third_party.contact_graspnet.contact_graspnet.utils.ipc import CGNClient

    ############### setup planner #####################
class CGNAgent():
    def __init__(self,env) -> None:
        self.env = env
        self.robot = self.env.agent.robot
        self.cgn_client = CGNClient()
        self.planner = None
        self.state = 0
        self.step = 0
        self.init_planner()
        # self.control_time_step = env.sim_timestep # use the same frequency as the simulator
        self.control_time_step = 1 / env.control_freq # use the same frequency as the agent in sim
        # self.action_space = env.action_space 
        base_pose = self.env.agent.robot.get_root_pose()
        base_t = base_pose.p
        base_q = Quaternion(base_pose.q)
        base_q_mat = base_q.rotation_matrix
        self.base_pose = np.eye(4)
        self.base_pose[:3, :3] = base_q_mat
        self.base_pose[:3, 3] = base_t
        self.base_pose_inv = np.linalg.inv(self.base_pose) # base pose in the world frame

        self.grasp_steps = 10
        self.move_to_goal_steps = 20

    def reset(self):
        self.state = 0
        self.step = 0
        self.robot = self.env.agent.robot
        # self.init_planner()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cgn_client.close()

    def init_planner(self):
        robot = self.robot
        link_names = [link.get_name() for link in robot.get_links()]
        joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        from  mani_skill2 import PACKAGE_ASSET_DIR
        planner = mplib.Planner(
            urdf=f"{PACKAGE_ASSET_DIR}/descriptions/panda_v2.urdf",
            srdf=f"{PACKAGE_ASSET_DIR}/descriptions/panda_v2.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))
        self.planner = planner

    def get_grasp_traj(self,obs):
        rgb = obs['image']['base_camera']['rgb'] # h, w, 3
        depth = obs['image']['base_camera']['depth'] # h, w, 1
        seg = obs['image']['base_camera']['Segmentation'][:,:,1:2] # h, w ,1, this is the actor-level segmentation
        camera_k = obs['camera_param']['base_camera']['intrinsic_cv'] # in CV convention
        # camera2world = obs['camera_param']['base_camera']['cam2world_gl']

        ############## convert point cloud to base frame ################
        pc_world = obs['pointcloud']['xyzw']
        target_seg = obs['pointcloud']['robot_seg'].squeeze(1) # 1 for robot points
        target_seg = target_seg == 1
        pc_world = pc_world[target_seg] # remove robot points

        pc_world = pc_world[pc_world[:,3] > 0] # remove invalid points
        pc_base = pc_world @ self.base_pose_inv.T # pc in the base frame
        self.planner.update_point_cloud(pc_base[:,:3]) 
        ############################################

        camera_extrinsic = obs['camera_param']['base_camera']['extrinsic_cv']
        target_object_actor_ids = [x.id for x in self.env.get_actors() if x.name not in ['ground', 'goal_site']]
        seg_id = np.array(target_object_actor_ids)
        grasps, scores = self.cgn_client.get_grasps([rgb, depth, seg, camera_k,seg_id])
        
        obj_id = seg_id[0]
        scores_sorted = np.argsort(scores[obj_id])[::-1] # descending order

        for idx in scores_sorted:
            grasp = grasps[obj_id][idx]
            score = scores[obj_id][idx]

            # we need to use the inv of extrinsic_cv since the pc is in cv coordinates (instead of gl coordinates)
            grasp_world = np.linalg.inv(camera_extrinsic) @ grasp 
            grasp_base = self.base_pose_inv @ grasp_world # grasp pose in the base frame
            
            # we need to change the orientation of the grasp pose since it uses a different frame system
            rot = Quaternion(matrix=grasp_base,atol=1e-3,rtol=0)
            fix = Quaternion(axis=[0.0, 0.0, 1.0], degrees=90)
            rot = rot * fix
            pos = grasp_base[:3, 3]
            target_pose = np.concatenate([pos, rot.elements]) # [xyz, wxyz]

            result = self.planner.plan(target_pose, 
                                       self.robot.get_qpos(), 
                                       time_step=self.control_time_step,
                                       use_point_cloud=True,
                                       verbose=False)
            if result['status'] == 'Success':
                return result
        msg = 'No valid grasp found' if len(grasps[obj_id]) > 0 else 'Planning failed'
        result = {'status': msg}
        return result

    def get_goal_traj(self,obs):
        
        # base = obs['agent']['base_pose'][:3]
        # goal_pos = obs['extra']['goal_pos']

        # base[2] = goal_pos[2] 
        # target_z = (goal_pos - base) / np.linalg.norm(goal_pos - base)
        # cur_rot = obs['extra']['tcp_pose'][3:] # use current rot
        # cur_rot = Quaternion(cur_rot)
        # cur_rot_mat = cur_rot.rotation_matrix
        # cur_z = cur_rot_mat[:3,2]
        # rot_cur_tartz = get_rotation_between_vecs(cur_z, target_z)
        # tart_rot = rot_cur_tartz @ cur_rot_mat
        
        # # transform goal pos to the base frame
        # goal_pose = np.eye(4)
        # goal_pose[:3, :3] = tart_rot
        
        # goal_pose[:3, 3] = goal_pos
        # goal_pose_base = self.base_pose_inv @ goal_pose
        # goal_pos_base = goal_pose_base[:3,3]
        # goal_rot_base = goal_pose_base[:3, :3]
        # goal_rot_base_quat = Quaternion(matrix=goal_rot_base,atol=1e-3,rtol=0).elements
        
        # target_pose = np.concatenate([goal_pos_base, goal_rot_base_quat]) # [xyz, wxyz]

        # lift 0.2m
        cur_pose = obs['extra']['tcp_pose'] 
        cur_pose[2] += 0.3
        target_pose_mat = np.eye(4)
        target_pose_mat[:3,:3] = Quaternion(cur_pose[3:]).rotation_matrix
        target_pose_mat[:3,3] = cur_pose[:3]
        target_pose_base = self.base_pose_inv @ target_pose_mat
        target_pose = np.concatenate([target_pose_base[:3,3], Quaternion(matrix=target_pose_base[:3,:3],atol=1e-3,rtol=0).elements]) # [xyz, wxyz]
        
        result = self.planner.plan(target_pose, self.robot.get_qpos(), time_step=self.control_time_step)
        return result

    def act(self,obs):
        """
        Act with the following steps:
            State 0: generate grasps and trajectory, follow the trajectory
            state 1: grasp the object
            state 2: lift the object

        """
        info = None
        if self.state == 0:
            if self.step == 0:
            # generate grasps and trajectory accordingly
                
                self.trj = self.get_grasp_traj(obs)
                if self.trj['status'] != 'Success':
                    info = f'Grasp generation failed: {self.trj["status"]}'
                    print(info)
                    return {'info': info}
                print('Got grasp trajectory. Moving to the grasp pose...')
            
            gripper_action = np.array([1]) # 1 for open
            action = np.concatenate([self.trj['position'][self.step], self.trj['velocity'][self.step],gripper_action])
            self.step += 1

            if self.step >= self.trj['position'].shape[0]:
                self.state = 1
                self.step = 0
                print('Will open gripper...')
            
        
        elif self.state == 1:
            # move to the object and close the gripper
            gripper_action = np.array([-1]) # -1 for close
            action = np.concatenate([self.trj['position'][-1], self.trj['velocity'][-1],gripper_action])
            self.step += 1

            if self.step >= self.grasp_steps:
                self.state = 2
                self.step = 0
        
        elif self.state == 2:
            if self.step == 0:
                self.trj = self.get_goal_traj(obs)
                if self.trj['status'] != 'Success':
                    info = f'Goal trajectory generation failed: {self.trj["status"]}'
                    print(info)
                    return {'info': info}
                print('Goal trajectory generated. Moving to the goal pose...')

            gripper_action = np.array([-1]) # -1 for close
            action = np.concatenate([self.trj['position'][self.step], self.trj['velocity'][self.step],gripper_action])
            self.step += 1
                
            # lift the object
            if self.step >= self.trj['position'].shape[0]:
                self.state = 3
                self.step = 0

        elif self.state == 3:
            # stay at the goal pose for 2 steps
            gripper_action = np.array([-1]) # -1 for close
            action = np.concatenate([self.trj['position'][-1], self.trj['velocity'][-1],gripper_action])
            self.step += 1
            if self.step >= 10:
                info = 'Plan finished.'
                return {'info': info}
        
        plan_result = {}
        plan_result['action'] = action
        plan_result['info'] = 'Success' if info is None else info
        return plan_result