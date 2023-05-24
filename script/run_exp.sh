#! /usr/bin/env bash

SEED=(0 1 2)
ENV="PickSingleYCB-v0"
for seed in ${SEED[@]}; do
    python lipp/ppo_state.py --exp-name ppo-state-$ENV-seed-$seed -n 32 --seed $seed --total-timesteps 24000000 --env-id $ENV
    python lipp/ppo_rgbd.py --exp-name ppo-rgbd-$ENV-seed-$seed -n 32 --seed $seed --total-timesteps 24000000 --env-id $ENV
done