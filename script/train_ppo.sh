

python maniskill2_learn/apis/run_rl.py configs/mfrl/ppo/maniskill2_pn.py \
--work-dir logs/ppo-pickycb --gpu-ids 0 1 2 3 \
--cfg-options "env_cfg.env_name=PickSingleYCB-v0" "env_cfg.obs_mode=pointcloud" \
"env_cfg.n_points=1200" "env_cfg.control_mode=pd_ee_delta_pose" \
"env_cfg.obs_frame=ee" "env_cfg.n_goal_points=50" \
"env_cfg.reward_mode=dense" "rollout_cfg.num_procs=10" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"eval_cfg.num_procs=5" \
--resume-from maniskill2_learn_pretrained_models_videos/PickSingleYCB-v0/dapg_pointcloud/model_25000000.ckpt