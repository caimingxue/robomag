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
from loguru import logger


class PoseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    # 将会初始化动作空间与状态空间，便于强化学习算法在给定的状态空间中搜索合适的动作
    # 环境中会用的全局变量可以声明为类（self.）的变量
    def __init__(self):
        # state space: x, y, theta
        self.observation_space = spaces.Box(
            low=np.array([0, 0., -np.pi]),
            high=np.array([500., 500., np.pi]),
            dtype=np.float32
        )
        # action space: v, w
        self.action_space = spaces.Box(
            low=np.array([-5, -1]),
            high=np.array([5, 1]),
            dtype=np.float32
        )
        self.state = None   # current state

        self.dt = 0.05
        self.pre_dist = 0.
        self.dist = 0.
        self.delta_dist = 0.

        self.viewer = None
        self.out_border = False
        self.reset()
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        assert self.observation_space.contains(self.state), "%r (%s) invalid" % (self.state, type(self.state))
        assert self.observation_space.contains(self.target), "%r (%s) invalid" % (self.target, type(self.target))

        x_diff = self.target[0] - self.state[0]
        y_diff = self.target[1] - self.state[1]
        self.dist = np.hypot(x_diff, y_diff)

        temp = np.arctan2(y_diff, x_diff) - self.state[2]
        self.alpha = self.angle_normalize(temp)

        temp = self.target[2] - self.state[2] - self.alpha
        self.beta = self.angle_normalize(temp)

        v = action[0]
        w = action[1]

        if self.alpha > np.pi / 2 or self.alpha < -np.pi / 2:
            v = -v
        self.next_state = np.zeros_like(self.state)
        self.next_state[2] = self.state[2] + w * self.dt
        self.next_state[2] = (self.next_state[2] + np.pi) % (2 * np.pi) - np.pi
        self.next_state[0] = self.state[0] + v * np.cos(self.next_state[2]) * self.dt
        self.next_state[1] = self.state[1] + v * np.sin(self.next_state[2]) * self.dt

        # 在这里做一下限定，如果下一个动作导致智能体越过了环境边界（即不在状态空间中），则无视这个动作
        if self.observation_space.contains(self.next_state):
            self.state = self.next_state
        else:
            logger.info("next_state: {}", self.next_state)
            self.out_border = True
        self.step_num += 1

        reward = self.get_reward()

        self.delta_dist = self.pre_dist - self.dist
        self.pre_dist = self.dist

        done = self.get_done()
        info = {"action": action, "reward": reward, "distance": self.dist, "done": done}
        # print(info)

        return self.state, reward, done, info

    # 用于在每轮开始之前重置智能体的状态，把环境恢复到最开始
    # 在训练的时候，可以不指定start_state，随机选择初始状态，以便能尽可能全的采集到的环境中所有状态的数据反馈
    def reset(self, start_state=None):
        if start_state==None:
            self.state = self.observation_space.sample()
            logger.info("init state: {}", self.state)
        else: # 在训练完成测试的时候，可以根据需要指定从某个状态开始
            if self.observation_space.contains(start_state):
                self.state = start_state
            else:
                self.state = self.observation_space.sample()
        # give a random goal pose
        self.x_goal = 500 * random()
        self.y_goal = 500 * random()
        self.theta_goal = 2 * np.pi * random() - np.pi
        self.target = np.array([self.x_goal, self.y_goal, self.theta_goal])

        # clear the counts
        self.step_num = 0

        logger.info("@@@@@@@@@@@@@@@ Next episode to use reset @@@@@@@@@@@@@@@@@@@@")

        self.out_border = False
        return self.state

    # render()绘制可视化环境的部分都写在这里，gym中的这个color，（x, y, z）中的每一位应该取[0, 1]之间的值
    def render(self, mode='rgb_array', close=False):
        from gym.envs.classic_control import rendering
        # #向右为x轴正方向，向上为y轴正方向，左下角为坐标原点
        # self.viewer = rendering.Viewer(400, 400)    # 初始化一张画布
        # # length, width, 默认中心画在原点
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)

        self.viewer.draw_circle(
            20, 20, True, color=(0, 1, 0)
        ).add_attr(rendering.Transform((self.state[0], self.state[1])))

        self.viewer.draw_circle(
            20, 20, True, color=(0, 0, 1)
        ).add_attr(rendering.Transform((self.target[0], self.target[1])))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def get_obs(self):
        ...
        self.state = np.array([self.x, self.y, self.theta])
        return self.state

    def get_reward(self):
        ...
        Rp = 100 * self.delta_dist
        # Rp = np.exp(-self.distance)

        if self.out_border:
            R_punish = -10000000
        else:
            R_punish = 0

        if self.dist < 5:
            Rr = 20000
        else:
            Rr = 0.

        # R_action = -0.1 * u**2  # consider the action punishment

        Rt = -1
        reward = Rp + R_punish + Rr + Rt
        return reward

    def get_done(self):
        ...
        if self.dist < 5:
            done = True
            logger.warning("################## Success Reach#################")
        elif self.step_num > 10000:
            done = True
            logger.warning("################## Step Num Limits #################")
        elif self.out_border:
            done = True
            logger.error("************ Out of Border *****************")
        else:
            done = False
        return done

    def angle_normalize(self, angle):
        return (((angle + np.pi) % (2 * np.pi)) - np.pi)

if __name__ == '__main__':

    import gym
    from stable_baselines3 import SAC
    env = PoseEnv()
    # from stable_baselines3.common.env_checker import check_env
    # check_env(env)
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("./pose_track.pkl")

    # model = SAC.load("./pose_track.pkl")

    obs = env.reset()
    for i in range(100000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()

    # Kp_rho = 9
    # Kp_alpha = 15
    # Kp_beta = -3
    # from stable_baselines3.common.env_checker import check_env
    # check_env(env)
    # action = [0.1, 0.1]
    # while 1:
    #     env.step(action)
    #     env.render()
    #     v = Kp_rho * env.distance
    #     w = Kp_alpha * env.alpha + Kp_beta * env.beta
    #     action = [v, w]
    #     print("====================================")
    #     time.sleep(1)

