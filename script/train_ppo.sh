
# to continue training DADPG agent using PPO for 5 mil steps
python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn.py \
--work-dir logs/ppo-pickycb --gpu-ids 0 1 2 3 \
--cfg-options "env_cfg.env_name=PickSingleYCB-v0" "env_cfg.obs_mode=pointcloud" \
"env_cfg.n_points=1200" "env_cfg.control_mode=pd_ee_delta_pose" \
"env_cfg.obs_frame=ee" "env_cfg.n_goal_points=50" \
"env_cfg.reward_mode=dense" "rollout_cfg.num_procs=15" \
"eval_cfg.num=500" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"eval_cfg.num_procs=5" \
--resume-from maniskill2_learn_pretrained_models_videos/PickSingleYCB-v0/dapg_pointcloud/model_25000000.ckpt



# to evaluate the trained agent
python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn.py \
            --work-dir logs/ppo-pickycb-eval --gpu-ids 0 \
            --cfg-options "env_cfg.env_name=PickSingleYCB-v0" "env_cfg.obs_mode=pointcloud" "env_cfg.n_points=1200" \
            "env_cfg.control_mode=pd_ee_delta_pose" "env_cfg.obs_frame=ee" "env_cfg.n_goal_points=50" \
            "eval_cfg.num=500" "eval_cfg.num_procs=1" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
            "eval_cfg.seed=0" \
            --evaluation --resume-from logs/ppo-pickycb/models/model_final.ckpt



# to continue training DADPG agent using PPO for 50 mil steps using sparse reward
python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn.py \
--work-dir logs/ppo-pickycb-sparse --gpu-ids 0 1 2 3 \
--cfg-options "env_cfg.env_name=PickSingleYCB-v0" "env_cfg.obs_mode=pointcloud" \
"env_cfg.n_points=1200" "env_cfg.control_mode=pd_ee_delta_pose" \
"env_cfg.obs_frame=ee" "env_cfg.n_goal_points=50" \
"env_cfg.reward_mode=sparse" "rollout_cfg.num_procs=15" \
"eval_cfg.num=500" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"eval_cfg.num_procs=5" "train_cfg.total_steps=50000000" \
--resume-from maniskill2_learn_pretrained_models_videos/PickSingleYCB-v0/dapg_pointcloud/model_25000000.ckpt