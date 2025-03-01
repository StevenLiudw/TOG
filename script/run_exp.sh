#! /usr/bin/env bash

SEED=(0 1 2)
ENV="PickSingleYCB-v0"
for seed in ${SEED[@]}; do
    python rl/ppo_state.py --exp-name ppo-state-$ENV-seed-$seed -n 8 --seed $seed --total-timesteps 1000000 --env-id $ENV
    python rl/ppo_rgbd.py --exp-name ppo-rgbd-$ENV-seed-$seed -n 8 --seed $seed --total-timesteps 1000000 --env-id $ENV
done