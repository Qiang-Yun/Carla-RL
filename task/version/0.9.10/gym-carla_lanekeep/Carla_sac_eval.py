from Carla_sac import get_args
import argparse
import datetime
import os
import pprint

import numpy as np
import torch
# from mujoco_env import make_mujoco_env
# from tianshou.env import SubprocVectorEnv

from RoundaboutCarlaEnv import RoundaboutCarlaEnv 
# from ac_zhy.wrappers_missile import WraSingleAgent
# from config import Algo_config, Env_config_missile
# from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
# from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


def eval_sac(args=get_args()):
    # env, train_envs, test_envs = make_mujoco_env(
    #     args.task, args.seed, args.training_num, args.test_num, obs_norm=False
    # )
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
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # log
    # now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # now = "2022-well-done"
    # args.algo_name = "sac"
    # log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    # log_path = os.path.join(args.logdir, log_name, "policy_20221012.pth")
    log_path =  r"/home/yq/CARLA_0.9.6/CarlaRL/gym-carla_lanekeep/log/Carla-KeepLane-v3/sac/0/221024-151433/policy.pth" # 偏右行驶
    # log_path =  r"/home/yq/CARLA_0.9.6/CarlaRL/gym-carla_lanekeep/log/Carla-KeepLane-v3/sac/0/221024-151433/policy.pth"
    print()
    # model_dir = os.path.join(project_path, args.logdir, args.task, Algorithm_name, args.modeldir, 'model.pth')
    policy.load_state_dict(torch.load(log_path, map_location=torch.device(args.device))['model'])
    # Let's watch its performance!
    policy.eval()
    # policy.actor.forward()
    env.seed(args.seed)
    collector = Collector(policy, env)
    # args.render = 0.001
    # result = collector.collect(n_episode=100, render=args.render)
    result = collector.collect(n_episode=100, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    eval_sac()
