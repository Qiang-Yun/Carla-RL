#!/usr/bin/env python3

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
  sys.path.append('/home/yq/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
except IndexError:
    pass
import carla
from gym_carla.envs.carla_env import  CarlaEnv


class RoundaboutCarlaEnv(gym.Env):
    def __init__(self):
        carla_params = {
            'number_of_vehicles': 0,
            'number_of_walkers': 0,
            'display_size': 512,  # screen size of bird-eye render, 显示窗口的大小
            'max_past_step': 1,  # the number of past steps to draw
            'dt': 0.1,  # time interval between two frames
            'discrete': False,  # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
            'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
            'port': 2000,  # connection port
            'town': 'Town03',  # which town to simulate
            'task_mode': 'lanekeep',  # mode of the task, [lanekeep]
            'max_time_episode': 1000,  # maximum timesteps per episode
            'max_waypt': 12,  # maximum number of waypoints
            'obs_range': 32,  # observation range (meter)
            'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
            'd_behind': 12,  # distance behind the ego vehicle (meter)
            'out_lane_thres': 4.0,  # threshold for out of lane
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
        5.与前方车辆的夹角
        """
        self.observation_space = gym.spaces.Box(low = np.array([-3,-1,0,0]),high = np.array([3,1,30,1]))

        """
        动作空间:
        1.油门 -3,3
        2.方向盘转角 -0.3,0.3
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
        print("step:action======", action,'reward:',reward)
        return state, reward,done,{}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="LaneKeep-V1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--training-num", type=int, default=16)
    parser.add_argument("--test-num", type=int, default=1)
    # ppo special
    parser.add_argument("--rew-norm", type=int, default=True) 
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--save-interval", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default="/home/yq/CARLA_0.9.6/CarlaRL/gym-carla_lanekeep/log/LaneKeep-V1/ppo/0/220830-154911/policy.pth")
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    # parser.add_argument(
    #     "--watch",
    #     default=True,
    #     action="store_true",
    #     help="watch the play of pre-trained policy only",
    # )
    return parser.parse_args()


def run_ppo(args=get_args()):
    env = RoundaboutCarlaEnv()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr
    )

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        print("Loaded agent from: ", args.resume_path)

    # collector
    # if args.training_num > 1:
    #     buffer = VectorReplayBuffer(args.buffer_size, len(env))
    # else:
    #     buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(
        policy, 
        env, 
        VectorReplayBuffer(args.buffer_size, 1), 
        exploration_noise=True
    )
    test_collector = Collector(policy, env)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    # if args.logger == "wandb":
    #     logger = WandbLogger(
    #         save_interval=1,
    #         name=log_name.replace(os.path.sep, "__"),
    #         run_id=args.resume_id,
    #         config=args,
    #         project=args.wandb_project,
    #     )
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)
    # writer.add_text("args", str(args))
    # if args.logger == "tensorboard":
    #     logger = TensorboardLogger(writer)
    # else:  # wandb
    #     logger.load(writer)



    def save_best_fn(policy):
        state = {"model": policy.state_dict()}
        torch.save(state, os.path.join(log_path, "policy.pth"))

    # if not args.watch:
    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        save_best_fn=save_best_fn,
        logger=logger,
        # resume_from_log=args.resume,
        # save_checkpoint_fn=save_checkpoint_fn,
    )
    pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    env.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    run_ppo()
