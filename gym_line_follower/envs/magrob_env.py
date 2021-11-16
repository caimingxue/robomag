#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : Example.py
# @Time : 2021/11/6 2:26 下午
# @Author : Mingxue Cai
# @Email : im_caimingxue@163.com
# @github : https://github.com/caimingxue/magnetic-robot-simulation
# @notice ：
import gym
from gym import spaces
import numpy as np
from random import random
from gym.envs.classic_control import rendering
import time
import math
from loguru import logger


class MagRob_Env(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    def __init__(self):
        self.min_mag_freq = -5.
        self.max_mag_freq = 5.
        self.min_mag_amp = 0.
        self.max_mag_amp = 5.
        self.min_mag_angle = -np.pi
        self.max_mag_angle = np.pi

        self.min_x_diff = -8.
        self.max_x_diff = 8.
        self.min_y_diff = -8.
        self.max_y_diff = 8.
        self.min_dist_error = 0.
        self.max_dist_error = 8 * np.sqrt(2)
        self.min_orien_diff = -np.pi
        self.max_orien_diff = np.pi
        self.min_vel = -1.
        self.max_vel = 1.

        self.magrob_position_bound = [-4, 4]
        self.magrob_yaw_bound = [-np.pi, np.pi]
        self.action_init = np.array([0, 0, 0])


        # state space: x_diff, y_diff, dist_error, orien_diff, vel
        self.observation_space = spaces.Box(
            low=np.array([self.min_x_diff, self.min_y_diff, self.min_dist_error, self.min_orien_diff, self.min_vel]),
            high=np.array([self.max_x_diff, self.max_y_diff, self.max_dist_error, self.max_orien_diff, self.max_vel]),
            dtype=np.float32
        )
        # action space: frequency, amplitude, angle
        self.action_space = spaces.Box(
            low=np.array([self.min_mag_freq, self.min_mag_amp, self.min_mag_angle]),
            high=np.array([self.max_mag_freq, self.max_mag_amp, self.max_mag_angle]),
            dtype=np.float32
        )
        # x, y, yaw
        self.robot_pose_space = spaces.Box(
            low=np.array([self.magrob_position_bound[0], self.magrob_position_bound[0], self.magrob_yaw_bound[0]]),
            high=np.array([self.magrob_position_bound[1], self.magrob_position_bound[1], self.magrob_yaw_bound[1]]),
            dtype=np.float32
        )

        self.state = None
        self.current_pose = None
        self.map_size = np.array([5, 5])

        self.dt = 0.1
        self.vel_coeff = 0.2

        self.viewer = None
        self.reset()
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        assert self.observation_space.contains(self.state), "%r (%s) invalid" % (self.state, type(self.state))
        # Bot position updated here so it must be first!
        next_robot_pose = self.update_pose(action)

        self.current_pose = np.clip(next_robot_pose, a_min=self.robot_pose_space.low,
                                                     a_max=self.robot_pose_space.high)

        self.state = self.get_obs(action)

        # reward = 30 * np.exp(1 - self.dist_error / self.max_dist_error) - 20 * abs(self.alpha) - 1 * abs(self.beta)
        assert abs(self.alpha) <= np.pi
        assert abs(self.beta) <= np.pi
        # reward = 10 * np.exp(1 - self.dist_error / self.max_dist_error) \
        #          + 10 * np.exp(1 - abs(self.alpha) / np.pi) + 1 * np.exp(1 - abs(self.beta) / np.pi) \
                 # + np.exp(1 - abs(action[0]) / 5) + 1 * np.exp(1 - abs(action[1]) / np.pi)

        # reward = - 3 * self.dist_error**2 - 4 * abs(self.alpha)**2
        # reward_dist = -np.linalg.norm(self.dist_error)
        # reward_ctrl = -np.square(action).sum()
        # reward = 3 * reward_dist + reward_ctrl

        reward = -5 * self.dist_error - 2 * abs(self.beta)

        done = False
        if self.dist_error < 0.02:
            done = True
            logger.warning("################## REACH SUCCESSFULLY #################")
        elif self.step_num > 1000:
            done = True
            logger.error("=================== STEP NUM LIMITS =======================")
        else:
            done = False

        self.step_num += 1

        info = {"action": action, "state": self.state, "reward": reward, "done": done}
        print(info)

        return self.state, reward, done, info

    # 用于在每轮开始之前重置智能体的状态，把环境恢复到最开始
    # 在训练的时候，可以不指定start_state，随机选择初始状态，以便能尽可能全的采集到的环境中所有状态的数据反馈
    def reset(self, start_pose=None):
        if start_pose == None:
            self.current_pose = self.robot_pose_space.sample()
        else: # 在训练完成测试的时候，可以根据需要指定从某个状态开始
            if not self.robot_pose_space.contains(start_pose):
                self.current_pose = start_pose
        # give a random goal pose
        self.target_pose = self.robot_pose_space.sample()
        # clear
        self.step_num = 0

        action = self.action_init

        self.state = self.get_obs(action)

        assert self.observation_space.contains(self.state), "%r (%s) invalid" % (self.state, type(self.state))

        return self.state

    # render()绘制可视化环境的部分都写在这里，gym中的这个color，（x, y, z）中的每一位应该取[0, 1]之间的值
    def render(self, mode='human', close=False):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.map_size[0], self.map_size[0], -self.map_size[0], self.map_size[0])
            # length, width
            self.magrob_length = 0.6
            self.magrob_width = 0.4
            magrob = rendering.make_capsule(self.magrob_length, self.magrob_width)
            magrob.set_color(0.8, 0.3, 0.3)
            # 默认中心不在原点
            self.magrob_transform = rendering.Transform(translation=(-self.magrob_length/2, 0))
            magrob.add_attr(self.magrob_transform)
            self.viewer.add_geom(magrob)

            magrob_axle = rendering.make_circle(0.1)
            magrob_axle.set_color(0, 0, 0)
            self.magrob_transform_point = rendering.Transform(translation=(0, 0))
            magrob_axle.add_attr(self.magrob_transform_point)
            self.viewer.add_geom(magrob_axle)

            # Target Setting
            magrob_target = rendering.make_capsule(self.magrob_length, self.magrob_width)
            magrob_target.set_color(0.0, 0.0, 0.0)
            self.magrob_transform_target = rendering.Transform(translation=(-self.magrob_length/2,
                                                                            0))
            magrob_target.add_attr(self.magrob_transform_target)
            self.viewer.add_geom(magrob_target)

            axle = rendering.make_circle(0.1)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)


        self.magrob_transform.set_translation(self.current_pose[0]-self.magrob_length/2,
                                              self.current_pose[1])
        self.magrob_transform.set_rotation(self.current_pose[2])

        self.magrob_transform_point.set_translation(self.current_pose[0], self.current_pose[1])

        self.magrob_transform_target.set_translation(self.target_pose[0] - self.magrob_length / 2,
                                              self.target_pose[1])
        self.magrob_transform_target.set_rotation(self.target_pose[2])


        return self.viewer.render(return_rgb_array=mode == 'human')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def update_pose(self, action):

        current_yaw = self.current_pose[2]
        self.alpha = self.angle_normalize(np.arctan2(self.state[1], self.state[0]+0.0001) - current_yaw)

        target_yaw = self.target_pose[2]
        self.beta = self.angle_normalize(target_yaw - current_yaw - self.alpha)

        # action: freq, amplitude, angle
        vel = action[0] * self.vel_coeff + action[1] * self.vel_coeff
        angular_vel = action[2]

        ## 保持只前进，不后退
        # if self.alpha > np.pi / 2 or self.alpha < -np.pi / 2:
        #     vel = -vel

        self.next_pose = np.zeros_like(self.current_pose)
        self.next_pose[2] = self.angle_normalize(current_yaw + angular_vel * self.dt)
        self.next_pose[0] = self.current_pose[0] + vel * np.cos(self.next_pose[2]) * self.dt
        self.next_pose[1] = self.current_pose[1] + vel * np.sin(self.next_pose[2]) * self.dt

        return self.next_pose


    def get_obs(self, action):
        ...
        # state space: x_diff, y_diff, dist_error, orien_diff, vel
        self.x_diff = self.target_pose[0] - self.current_pose[0]
        self.y_diff = self.target_pose[1] - self.current_pose[1]
        self.dist_error = np.sqrt(self.x_diff ** 2 + self.y_diff ** 2)

        current_yaw = self.current_pose[2]
        self.alpha = self.angle_normalize(np.arctan2(self.y_diff, self.x_diff + 0.0001) - current_yaw)

        target_yaw = self.target_pose[2]
        self.beta = self.angle_normalize(target_yaw - current_yaw - self.alpha)

        self.vel = action[0] * self.vel_coeff

        self.state = np.array([self.x_diff, self.y_diff, self.dist_error, self.beta, self.vel])

        return self.state

    def angle_normalize(self, angle):
        return (((angle + np.pi) % (2 * np.pi)) - np.pi)

if __name__ == '__main__':

    import gym
    from stable_baselines3 import SAC
    env = MagRob_Env()
    from stable_baselines3.common.env_checker import check_env
    check_env(env)
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("./pose_track.pkl")

    model = SAC.load("./pose_track.pkl")

    obs = env.reset()
    for i in range(100000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()



