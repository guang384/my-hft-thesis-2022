# 尝试训练
import sys
import random

import gym
import torch
from elegantrl.agent import AgentDQN
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
        file_path="data/dominant_processed_data_20170103_20220215.h5"):
    env_args = get_gym_env_args(gym.make("TinyMarketGymEnvDaily-v0"), if_print=False)

    def make_env_func(**kwargs):
        env_inner = gym.make(env_args['env_name'])
        env_inner.init(capital=20000,
                       file_path=file_path,
                       date_start="20220101", date_end="20220110", )
        return env_inner

    # 初始化
    args = Arguments(AgentDQN, env_func=make_env_func, env_args=env_args)
    env = config.build_env(env_func=make_env_func, env_args=env_args)

    # 加载模型
    act = AgentDQN(args.net_dim, env.state_dim, env.action_dim, gpu_id=args.learner_gpus, args=args).act
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
    else:
        test_model("data_sample/tiny_market_actor_avgR_4.00.pth")
