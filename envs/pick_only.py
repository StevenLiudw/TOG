from mani_skill2.envs.pick_and_place.pick_single import PickSingleYCBEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at

from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import gym
import numpy as np
from gym import spaces

from mani_skill2.utils.common import (
    flatten_dict_keys,
    flatten_dict_space_keys,
    merge_dicts,
)

@register_env("PicOnlykYCB-v0", max_episode_steps=200, override=True)
class PickOnlyYCB(PickSingleYCBEnv):
    # def _register_cameras(self):
    #     pose = look_at([0.3, 0.2, 0.6], [-0.1, 0, 0.1])
    #     left_camera = CameraConfig(
    #         "left_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
    #     )

    #     pose = look_at([0.3, -0.2, 0.6], [-0.1, 0, 0.1])
    #     right_camera = CameraConfig(
    #         "right_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
    #     )
    #     return [left_camera, right_camera]
    
    # def _register_render_cameras(self):
    #     return []
    def _initialize_task(self):
        super()._initialize_task()
        self._init_obj_z = self.obj_pose.p[2]
    def evaluate(self, **kwargs):
        # we only care about the object height
        init_z = self._init_obj_z
        obj_to_goal_pos = abs(init_z - self.obj_pose.p[2])
        is_obj_placed = obj_to_goal_pos >= 0.1
        is_robot_static = self.check_robot_static()
        return dict(
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed,
        )
    


class RGBDPCObservationWrapper(gym.ObservationWrapper):
    """ RGBD and Point Cloud Observation Wrapper"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = deepcopy(env.observation_space)
        self.update_observation_space(self.observation_space)
        self._buffer = {}
        self.replace = True
        # Cache robot link ids
        self.robot_link_ids = self.env.robot_link_ids

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.robot_link_ids = self.env.robot_link_ids
        return self.observation(observation)
    
    @staticmethod
    def update_observation_space(space: spaces.Dict):
        # Update image observation space
        image_space: spaces.Dict = space.spaces["image"]
        for cam_uid in image_space:
            ori_cam_space = image_space[cam_uid]
            new_cam_space = OrderedDict()
            for key in ori_cam_space:
                if key == "Color":
                    height, width = ori_cam_space[key].shape[:2]
                    new_cam_space["rgb"] = spaces.Box(
                        low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                    )
                elif key == "Position":
                    height, width = ori_cam_space[key].shape[:2]
                    new_cam_space["depth"] = spaces.Box(
                        low=0, high=np.inf, shape=(height, width, 1), dtype=np.float32
                    )
                else:
                    new_cam_space[key] = ori_cam_space[key]
            image_space.spaces[cam_uid] = spaces.Dict(new_cam_space)

    def observation(self, observation: dict):
        
        #################### process point cloud ####################
        image_obs = observation.get("image")
        camera_params = observation.get("camera_param")
        pointcloud_obs = OrderedDict()

        for cam_uid, images in image_obs.items():
            cam_pcd = {}

            # Each pixel is (x, y, z, z_buffer_depth) in OpenGL camera space
            position = images["Position"]
            # position[..., 3] = position[..., 3] < 1
            position[..., 3] = position[..., 2] < 0

            # Convert to world space
            cam2world = camera_params[cam_uid]["cam2world_gl"]
            xyzw = position.reshape(-1, 4) @ cam2world.T
            cam_pcd["xyzw"] = xyzw

            # Extra keys
            if "Color" in images:
                rgb = images["Color"][..., :3]
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                cam_pcd["rgb"] = rgb.reshape(-1, 3)
            if "Segmentation" in images:
                cam_pcd["Segmentation"] = images["Segmentation"].reshape(-1, 4)

            pointcloud_obs[cam_uid] = cam_pcd

        pointcloud_obs = merge_dicts(pointcloud_obs.values())
        for key, value in pointcloud_obs.items():
            buffer = self._buffer.get(key, None)
            pointcloud_obs[key] = np.concatenate(value, out=buffer)
            self._buffer[key] = pointcloud_obs[key]

        observation["pointcloud"] = pointcloud_obs

        ########### process RGBD
        image_obs = observation["image"]
        for cam_uid, ori_images in image_obs.items():
            new_images = OrderedDict()
            for key in ori_images:
                if key == "Color":
                    rgb = ori_images[key][..., :3]  # [H, W, 4]
                    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                    new_images["rgb"] = rgb  # [H, W, 4]
                elif key == "Position":
                    depth = -ori_images[key][..., [2]]  # [H, W, 1]
                    new_images["depth"] = depth
                else:
                    new_images[key] = ori_images[key]
            image_obs[cam_uid] = new_images

        observation = self.observation_pointcloud(observation)

        return observation

    def observation_pointcloud(self, observation: dict):
        target_object_actor_ids = [x.id for x in self.env.get_actors() if x.name not in ['ground', 'goal_site']]
        pointcloud_obs = observation["pointcloud"]
        if "Segmentation" not in pointcloud_obs:
            return observation
        seg = pointcloud_obs["Segmentation"]
        # robot_seg = np.isin(seg[..., 1:2], self.robot_link_ids)
        robot_seg = np.isin(seg[..., 1:2], target_object_actor_ids) # only care about the object
        if self.replace:
            pointcloud_obs.pop("Segmentation")
        pointcloud_obs["robot_seg"] = robot_seg
        return observation