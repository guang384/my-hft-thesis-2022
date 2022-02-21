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

ENV_NAME = 'TinyMarketGymEnvRandom-v0'

register(
    id=ENV_NAME,
    entry_point='tiny_market:GymEnvRandom',
    max_episode_steps=3600,  # 一个episode最大步数
    reward_threshold=1000000.0,
)

gym.logger.set_level(40)  # Block warning


def try_train(file_path="data/dominant_processed_data_20170103_20220215.h5",
              agent_name='dqn',
              cwd_suffix=None,
              train_and_evaluate_func=train_and_evaluate_mp,
              env_name=ENV_NAME):
    agent_name = agent_name.lower()
    if agent_name == 'dqn':
        agent = AgentDQN
        print('use agent AgentDQN')
    elif agent_name == 'discreteppo' or agent_name == 'ppo':
        agent = AgentDiscretePPO
        print('use agent AgentDiscretePPO')
    elif agent_name == 'doubledqn' or agent_name == 'ddqn':
        agent = AgentDoubleDQN
        print('use agent AgentDoubleDQN')
    elif agent_name == 'd3qn':
        agent = AgentD3QN
        print('use agent AgentD3QN')
    else:
        raise RuntimeError('unknown agent : ', agent_name)

    # AgentDQN
    # AgentDoubleDQN
    # AgentD3QN
    # AgentDiscretePPO

    def make_env_func(**kwargs):
        env = gym.make(env_name,
                       capital=20000,
                       file_path=file_path,
                       date_start="20211201", date_end="20211231",
                       reward_func=profits_or_loss_reward,
                       fine_func=linear_fine(0.1)
                       )
        return env

    tmp_env = make_env_func()
    env_args = get_gym_env_args(tmp_env, if_print=False)
    del tmp_env

    '''
    env_args = {
        'env_num': 1,
        'env_name': 'TinyMarketGymEnvRandom-v0',
        'max_step': 3600,
        'state_dim': 32,
        'action_dim': 3,
        'if_discrete': True,
        'target_return': 1000000.0,
    }
    '''

    args = Arguments(agent, env_func=CloudpickleWrapper(make_env_func), env_args=env_args)

    # DQN, DoubleDQN, D3QN, PPO-Discrete for discrete actions
    # AgentDQN
    # AgentDoubleDQN
    # AgentD3QN
    # AgentDiscretePPO

    # 经过测试：
    # 每个worker 大概消耗 2200M的显存（估计是当显存的模型放到GPU上时？）
    # 当worker_num=1 时 显存消耗约为 6900M显存
    # 当为多线程情况时 会有三类进程：
    #   一个进程用来Learn
    #   一个进程用来Evaluation
    #   N个Worker进程用来和环境交互获得trajectory
    # 文档注释：  rollout workers number pre GPU (adjust it to get high GPU usage)
    args.worker_num = 2  # for PC
    # args.worker_num = 8  # for Nvidia 3090 24G

    '''
     优化目标让训练用时和探索用时接近 这样在异步探索情况下CPU和GPU利用率最高
     (异步情况下，探索占很少时间（不占时间可能因为训练耗时太多了，或者也可用同步训练模式下调参确认训练和探索两个阶段用时接近后 再改成异步的)）
    '''

    # 这个参数是在训练过程中 update_net阶段每批次处理数据量（影响训练用时）
    # 默认是  self.net_dim * 2  用不满GPU 可以适当放大提高资源利用率
    args.batch_size = 2 ** 8  # for PC
    # args.batch_size = 2 ** 12  # for Nvidia 3090 24G

    # 单轮探索的最低步数（影响探索用时）
    # 当一个episode走完了会reset了继续走直到到达target_step，到达target_step会继续走完没走完的episode
    # 根据情况设置:让探索和训练次数达到平衡
    args.target_step = 300  # for PC
    # args.target_step = 500  # for Nvidia 3090 24G

    # 控制reply buffer 总容量
    # 每次探索得到数量不多时，总容量太大容易导致达到容量上限后新探索的结果reply buffer不能及时得到反应
    args.max_memo = 2 ** 20  # capacity of replay buffer

    args.gamma = 0.99
    args.eval_times = 2 ** 5
    args.if_remove = False
    if cwd_suffix is not None:
        args.cwd = f'./{args.env_name}_{args.agent.__name__[5:]}_{args.learner_gpus}_{cwd_suffix}'

    train_and_evaluate_func(args)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        argv_file_path = sys.argv[1]
        try_train(file_path=argv_file_path)
    elif len(sys.argv) == 3:
        argv_file_path = sys.argv[1]
        argv_cwd_suffix = sys.argv[2]
        try_train(file_path=argv_file_path, cwd_suffix=argv_cwd_suffix)
    elif len(sys.argv) == 4:
        argv_agent_name = sys.argv[1]
        argv_file_path = sys.argv[2]
        argv_cwd_suffix = sys.argv[3]

        try_train(agent_name=argv_agent_name, file_path=argv_file_path, cwd_suffix=argv_cwd_suffix)
    else:
        try_train()
