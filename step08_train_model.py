"""

进行训练

"""
import sys

from elegantrl.agent import AgentDQN, AgentDoubleDQN, AgentD3QN, AgentDiscretePPO
from elegantrl.config import get_gym_env_args, Arguments
from elegantrl.run import train_and_evaluate, train_and_evaluate_mp

import gym
from gym import register
from gym.vector.utils import CloudpickleWrapper

from tiny_market import profits_or_loss_reward, linear_fine

register(
    id='TinyMarketGymEnvRandom-v0',
    entry_point='tiny_market:GymEnvRandom',
    max_episode_steps=3600,  # 一个episode最大步数
    reward_threshold=1000000.0,
)

gym.logger.set_level(40)  # Block warning


def try_train(file_path="data/dominant_processed_data_20170103_20220215.h5",
              cwd_suffix=None):
    env_args = get_gym_env_args(gym.make("TinyMarketGymEnvRandom-v0"), if_print=False)

    def make_env_func(**kwargs):
        env = gym.make(env_args['env_name'])
        env.init(capital=20000,
                 file_path=file_path,
                 date_start="20211201", date_end="20211231",
                 reward_func=profits_or_loss_reward,
                 fine_func=linear_fine(0.1)
                 )
        return env

    args = Arguments(AgentDQN, env_func=CloudpickleWrapper(make_env_func), env_args=env_args)

    # DQN, DoubleDQN, D3QN, PPO-Discrete for discrete actions
    # AgentDQN
    # AgentDoubleDQN
    # AgentD3QN
    # AgentDiscretePPO

    args.target_step = args.max_step

    # 经过测试：
    # 每个worker 大概消耗 2200M的显存（估计是当显存的模型放到GPU上时？）
    # 当worker_num=1 时 显存消耗约为 6900M显存
    # 当为多线程情况时 会有三类进程：
    #   一个进程用来Learn
    #   一个进程用来Evaluation
    #   N个Worker进程用来和环境交互获得trajectory
    args.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)

    args.gamma = 0.99
    args.eval_times = 2**5
    args.if_remove = False
    if cwd_suffix is not None:
        args.cwd = f'./{args.env_name}_{args.agent.__name__[5:]}_{args.learner_gpus}_{cwd_suffix}'

    train_and_evaluate_mp(args)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        argv_file_path = sys.argv[1]
        try_train(argv_file_path)
    elif len(sys.argv) == 3:
        argv_file_path = sys.argv[1]
        argv_cwd_suffix = sys.argv[2]
        try_train(argv_file_path, argv_cwd_suffix)
    else:
        try_train()
