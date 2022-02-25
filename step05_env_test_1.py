"""

TinyMarket 基本功能验证
场景1：日常环境 给定范围内的第一个交易日 随机动作输出图

"""

import sys
import random

import gym
from gym import register, envs
from tiny_market import GymEnvDaily, GymEnvWaitAndSeeWillResultInFines

ENV_NAME = 'TinyMarketGymEnvDailyFineWatching-v0'
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
if ENV_NAME not in env_ids:
    register(
        id=ENV_NAME,
        entry_point='step05_env_test_1:GymEnvDailyFineWatching',
        max_episode_steps=50000,  # 一个episode最大步数
        reward_threshold=10000.0,
    )


class GymEnvDailyFineWatching(GymEnvDaily, GymEnvWaitAndSeeWillResultInFines):
    pass


def run_test(file_path="data/dominant_processed_data_20170103_20220215.h5"):
    env = gym.make(ENV_NAME,
                   capital=20000,
                   fine_pre_seconds=1,
                   file_path=file_path,
                   date_start="20211001", date_end="20211231")

    env.reset()
    print("Day : ", env.current_day())

    actions = []
    rewards = []
    total_reward = 0
    done = False
    while not done:
        rand = random.randint(0, 100)
        action = 0
        if rand > 95:
            action = 1
        elif rand < 5:
            action = 2
        actions.append(action)
        ob, reward, done, info = env.step(action)
        rewards.append(reward)
        total_reward = round(total_reward + reward, 1)
    print(info)
    print(total_reward)
    env.render()
    env.close()


if __name__ == '__main__':
    gym.logger.setLevel(gym.logger.WARN)
    if len(sys.argv) == 2:
        argv_file_path = sys.argv[1]
        run_test(argv_file_path)
    else:
        run_test()
    print("Done.")
