"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class TakeoffAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import matplotlib.pyplot as plt
import logging
import time
import argparse
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from drone.env.LiftoffAviary import LiftoffAviary
from drone.env.LiftoffAviary import ActionType

if __name__ == "__main__":

    env = LiftoffAviary(gui=False,
                        record=False,
                        act=ActionType.RPM)

    attitudes = []
    obs = env.reset()
    start = time.time()
    for i in range(100000):
        action = np.array([.10001, .10001, .1, .1])
        obs, reward, done, info = env.step(action)
        attitudes.append(obs[2])
        if i % env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
            break
    env.close()
    plt.plot(attitudes)
    plt.show()
