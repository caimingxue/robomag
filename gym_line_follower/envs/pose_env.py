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
            low=np.array([-4, -4., -np.pi]),
            high=np.array([4., 4., np.pi]),
            dtype=np.float32
        )
        # action space: v, w
        self.action_space = spaces.Box(
            low=np.array([-5, -np.pi]),
            high=np.array([5, np.pi]),
            dtype=np.float32
        )
        self.state = None   # current state

        self.map_size = np.array([5, 5])
        self.max_dist_error = np.sqrt(np.sum(self.map_size ** 2))

        self.dt = 0.01
        self.pre_dist_error = 0.
        self.dist_error = 0.
        self.delta_dist_error = 0.

        self.viewer = None
        self.out_border = False
        self.reset()
        self.seed()

    def step(self, action):

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        assert self.observation_space.contains(self.state), "%r (%s) invalid" % (self.state, type(self.state))
        assert self.observation_space.contains(self.target), "%r (%s) invalid" % (self.target, type(self.target))
        # Bot position updated here so it must be first!
        next_state = self.update_state(action)
        # 在这里做一下限定，如果下一个动作导致智能体越过了环境边界（即不在状态空间中），则无视这个动作
        if self.observation_space.contains(next_state):
            self.state = next_state
        else:
            self.out_border = True

        # reward = 30 * np.exp(1 - self.dist_error / self.max_dist_error) - 20 * abs(self.alpha) - 1 * abs(self.beta)
        assert abs(self.alpha) <= np.pi
        assert abs(self.beta) <= np.pi
        reward = 6 * np.exp(1 - self.dist_error / self.max_dist_error) \
                 + 10 * np.exp(1 - abs(self.alpha) / np.pi) + 1 * np.exp(1 - abs(self.beta) / np.pi) \
                 # + np.exp(1 - abs(action[0]) / 5) + 1 * np.exp(1 - abs(action[1]) / np.pi)

        # # Time penalty
        # reward -= 0.2

        done = False
        if self.dist_error < 0.02:
            reward = 100.
            done = True
            logger.warning("################## REACH SUCCESSFULLY #################")
        elif self.step_num > 5000:
            reward = -100.
            done = True
            logger.warning("=================== STEP NUM LIMITS =======================")
        elif self.out_border:
            reward = -100.
            done = True
            logger.error("************ OUT OF BORDER *****************")
        else:
            done = False

        self.delta_dist_error = self.pre_dist_error - self.dist_error
        self.pre_dist_error = self.dist_error
        self.step_num += 1

        info = {"action": action, "reward": reward, "distance": self.dist_error, "done": done}
        # print(info)

        return self.state, reward, done, info

    # 用于在每轮开始之前重置智能体的状态，把环境恢复到最开始
    # 在训练的时候，可以不指定start_state，随机选择初始状态，以便能尽可能全的采集到的环境中所有状态的数据反馈
    def reset(self, start_state=None):
        if start_state==None:
            self.state = self.observation_space.sample()
            # logger.info("init state: {}", self.state)
        else: # 在训练完成测试的时候，可以根据需要指定从某个状态开始
            if self.observation_space.contains(start_state):
                self.state = start_state
            else:
                self.state = self.observation_space.sample()
        # give a random goal pose
        self.x_goal = 4 * random()
        self.y_goal = 4 * random()
        self.theta_goal = 2 * np.pi * random() - np.pi
        self.target = np.array([self.x_goal, self.y_goal, self.theta_goal])

        # clear
        self.step_num = 0
        self.out_border = False

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
            # Target Setting
            magrob_target = rendering.make_capsule(self.magrob_length, self.magrob_width)
            magrob_target.set_color(0.0, 0.0, 0.0)
            self.magrob_transform_target = rendering.Transform(translation=(self.target[0]-self.magrob_length/2, self.target[1]))
            magrob_target.add_attr(self.magrob_transform_target)
            self.viewer.add_geom(magrob_target)

            axle = rendering.make_circle(0.1)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

        self.magrob_transform.set_translation(self.state[0]-self.magrob_length/2, self.state[1])
        self.magrob_transform.set_rotation(self.state[2])


        return self.viewer.render(return_rgb_array=mode == 'human')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def update_state(self, action):
        x_diff = self.target[0] - self.state[0]
        y_diff = self.target[1] - self.state[1]
        self.dist_error = np.hypot(x_diff, y_diff)

        current_yaw = self.state[2]
        self.alpha = self.angle_normalize(np.arctan2(y_diff, x_diff) - current_yaw)

        target_yaw = self.target[2]
        self.beta = self.angle_normalize(target_yaw - current_yaw - self.alpha)

        vel = action[0]
        angular_vel = action[1]

        if self.alpha > np.pi / 2 or self.alpha < -np.pi / 2:
            vel = -vel
        self.next_state = np.zeros_like(self.state)
        self.next_state[2] = self.angle_normalize(current_yaw + angular_vel * self.dt)
        self.next_state[0] = self.state[0] + vel * np.cos(self.next_state[2]) * self.dt
        self.next_state[1] = self.state[1] + vel * np.sin(self.next_state[2]) * self.dt
        return self.next_state

        # throttle = a[0]
        # steer = a[1]
        #
        # state[0] = state[0] + throttle * math.cos(state[2]) * DT
        # state[1] = state[1] + throttle * math.sin(state[2]) * DT
        # state[2] = state[2] + throttle / WB * math.tan(steer) * DT

        # return state

    def get_obs(self):
        ...
        self.state = np.array([self.x, self.y, self.theta])
        return self.state

    def angle_normalize(self, angle):
        return (((angle + np.pi) % (2 * np.pi)) - np.pi)

    # def get_img(self, obs):
    #     obs = self.obs_to_pixel(obs)
    #     goal = self.obs_to_pixel(self.env.goal)
    #     img_state = np.zeros((self.resolution, self.resolution, 3), np.uint8)
    #     img_state.fill(255)  # create white img
    #     img_state = cv2.circle(img_state, tuple(obs), 2, (255, 0, 0), -1)
    #     self.img_state = cv2.drawMarker(img_state, tuple(goal), (0, 0, 255), cv2.MARKER_STAR, 10)
    #     return self.img_state.copy().transpose(2, 0, 1)
    #
    # def obs_to_pixel(self, obs):
    #     pixel_x = int((obs[0] / self.env.map_size[0]) * self.resolution)
    #     pixel_y = self.resolution - int((obs[1] / self.env.map_size[0]) * self.resolution)
    #
    #     return [pixel_x, pixel_y]

if __name__ == '__main__':

    import gym
    from stable_baselines3 import SAC
    env = PoseEnv()
    from stable_baselines3.common.env_checker import check_env
    check_env(env)
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
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
    # Kp_alpha = 12
    # Kp_beta = -1
    # from stable_baselines3.common.env_checker import check_env
    # check_env(env)
    # action = [0.3, 0.3]
    # while 1:
    #     env.step(action)
    #     env.render()
    #     v = Kp_rho * env.dist_error
    #     w = Kp_alpha * env.alpha + Kp_beta * env.beta
    #
    #     # if v > 5:
    #     #     v = 5
    #     # if v < -5:
    #     #     v = -5
    #     # if w > np.pi:
    #     #     w = np.pi
    #     # if w < -np.pi:
    #     #     w = -np.pi
    #     action = [v, w]
    #     print("====================================")
    #     time.sleep(0.01)

