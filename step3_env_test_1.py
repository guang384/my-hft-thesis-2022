# 模拟市场环境测试
# 场景1：日常环境 给定范围内的第一个交易日 随机动作输出图

import random

import gym
from gym import register

from tiny_market import profits_or_loss_with_fine_reward

register(
    id='TinyMarketGymEnvDaily-v0',
    entry_point='tiny_market:GymEnvDaily',
    max_episode_steps=50000,  # 一个episode最大步数
    reward_threshold=1000000.0,
)


def train():
    env = gym.make('TinyMarketGymEnvDaily-v0')
    env.init(capital=20000,
             file_path="dominant_processed_data_20170101_20220214.h5",
             date_start="20211001", date_end="20211231",
             reward_func=profits_or_loss_with_fine_reward)

    env.reset()
    print("Day : ", env.current_day())

    done = False
    while not done:
        rand = random.randint(0, 100)
        action = 0
        if rand > 95:
            action = 1
        elif rand < 5:
            action = 2
        ob, reward, done, info = env.step(action)
    env.render()


if __name__ == '__main__':
    gym.logger.setLevel(gym.logger.WARN)
    train()
    print("Done.")
