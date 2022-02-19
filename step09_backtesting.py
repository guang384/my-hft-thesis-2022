"""

回测训练结果
给定训练好的模型和时间周期，按天为单位打印收益曲线

python step9_backtesting.py data/actor_01568563_00003.500.pth    20220101    20220110
 |<--   run command    -->|   |<--   path to model file    -->|  |<-start->|  |<-end->|

"""
import sys
import random

import gym
import torch
from elegantrl.agent import AgentDQN, AgentDiscretePPO, AgentDoubleDQN, AgentD3QN
from elegantrl.config import get_gym_env_args, Arguments
from gym import register

from elegantrl import config
from tiny_market import linear_fine

register(
    id='TinyMarketGymEnvDaily-v0',
    entry_point='tiny_market:GymEnvDaily',
    max_episode_steps=50000,  # 一个episode最大步数
    reward_threshold=1000000.0,
)

gym.logger.set_level(40)  # Block warning


def test_model(
        actor_path,
        agent_name='DQN',
        start_date="20220101",
        end_date="20220110",
        file_path="data/dominant_processed_data_20170103_20220215.h5"):
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

    env_args = get_gym_env_args(gym.make("TinyMarketGymEnvDaily-v0"), if_print=False)

    def make_env_func(**kwargs):
        env_inner = gym.make(env_args['env_name'])
        env_inner.init(capital=20000,
                       file_path=file_path,
                       date_start=start_date, date_end=end_date, )
        return env_inner

    # 初始化
    args = Arguments(agent, env_func=make_env_func, env_args=env_args)
    env = config.build_env(env_func=make_env_func, env_args=env_args)

    # 加载模型
    act = agent(args.net_dim, env.state_dim, env.action_dim, gpu_id=args.learner_gpus, args=args).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    torch.no_grad()
    print('Model loaded -> ', actor_path)

    # 测试开始
    device = next(act.parameters()).device  # net.parameters() is a Python generator.
    max_step = 50000

    state = env.reset()
    while not env.if_all_days_done():
        episode_step = None
        info = None
        episode_return = 0.0  # sum of rewards in an episode
        for episode_step in range(max_step):
            s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            a_tensor = act(s_tensor)
            a_tensor = a_tensor.argmax(dim=1)  # discrete
            action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
            state, reward, done, info = env.step(action)
            episode_return += reward
            if done:
                break
        episode_return  # 这是最后的收益
        episode_step += 1  # 这是总步数
        # 展示结果
        print("The final gain is %.2f. The final total account balance is %.2f RMB, Date: %s "
              % (episode_return, info['amount'], env.current_day()))
        # 下一天
        env.reset()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        argv_actor_path = sys.argv[1]
        test_model(argv_actor_path)
    elif len(sys.argv) == 3:
        argv_actor_path = sys.argv[1]
        argv_agent_name = sys.argv[2]
        test_model(actor_path=argv_actor_path,
                   agent_name=argv_agent_name)
    elif len(sys.argv) == 4:
        argv_actor_path = sys.argv[1]
        argv_test_start_date = sys.argv[2]
        argv_test_end_date = sys.argv[3]
        test_model(actor_path=argv_actor_path,
                   start_date=argv_test_start_date,
                   end_date=argv_test_end_date)
    elif len(sys.argv) == 5:
        argv_actor_path = sys.argv[1]
        argv_agent_name = sys.argv[2]
        argv_test_start_date = sys.argv[3]
        argv_test_end_date = sys.argv[4]
        test_model(actor_path=argv_actor_path,
                   agent_name=argv_agent_name,
                   start_date=argv_test_start_date,
                   end_date=argv_test_end_date)

    else:
        test_model("data_sample/tiny_market_actor_avgR_4.00.pth")
