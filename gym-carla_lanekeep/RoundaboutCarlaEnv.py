import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import gym
import sys
try:
  sys.path.append('/home/yq/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')
except IndexError:
    pass
import carla
from gym_carla.envs.carla_env import  CarlaEnv

class RoundaboutCarlaEnv(gym.Env):
    def __init__(self):
        carla_params = {
            'number_of_vehicles': 0,  #环境中的车辆
            'number_of_walkers': 0,   #环境中的行人
            'display_size': 512,  # screen size of bird-eye render, 显示窗口的大小
            'max_past_step': 1,  # the number of past steps to draw  #要绘制的过去步骤数？？？？？？？？
            'dt': 0.1,  # time interval between two frames
            'discrete': False,  # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
            'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
            'port': 2000,  # connection port
            'town': 'Town03',  # which town to simulate
            'task_mode': 'lanekeep',  # mode of the task, [acc]
            'max_time_episode': 1000,  # maximum timesteps per episode
            'max_waypt': 12,  # maximum number of waypoints
            'obs_range': 32,  # observation range (meter)
            'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
            'd_behind': 12,  # distance behind the ego vehicle (meter)
            'out_lane_thres': 2.0,  # threshold for out of lane  #偏离的距离
            'desired_speed': 8,  # desired speed (m/s)
            'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
            'display_route': True,  # whether to render the desired route
            'pixor_size': 64,  # size of the pixor labels
            'pixor': False,  # whether to output PIXOR observation
            'learning_rate': 0.1,
            'discount': 0.9,
            'epsilon': 0.8,
        }
        self.out_lane_thres = 10
        self.desired_speed = 30
        self.max_timestep = 1000
        
        """
        状态空间：
        1.车辆与车道线的位置
        2.车辆与车道线的夹角
        3.车辆的速度
        4.与前方车辆的距离
        """
        self.observation_space = gym.spaces.Box(low = np.array([-3,-1,0,0]),high = np.array([3,1,30,1]))


        """
        动作空间:
        1.油门 0,1
        2.方向盘转角 -1,1
        3.刹车 0,1
        """
        self.action_space = gym.spaces.Box(low=np.array([-3, -0.3]), high=np.array([3, 0.3]))

        self.lateral_dis = 0
        self.delta_yaw = 0
        self.speed = 0
        self.front_vehicle_dis = 0
        self.front_vehicle_ang = 0
        self.lspeed_lon = 0
        self.target = []
        self.isCollision = False
        self.dest = False

        self.timestep = 0
        self.env = CarlaEnv(params=carla_params)
       
    def reset(self):
        state = self.env.reset()
        self.timestep = 0
        if state is None:
            print('未初始化。。')
        # print('reset_state==',state[:5])
        return state['state']

    def step(self, action):
        obs_p, reward, done, info= self.env.step(action)

        self.timestep += 1
        state = [obs_p['state'][0], obs_p['state'][1], obs_p['state'][2], obs_p['state'][3]]
        if (self.timestep% 50 == 0):
             print("step:action======", action,'reward:',reward)
        return state, reward,done,{}