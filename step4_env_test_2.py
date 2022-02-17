# 模拟市场环境测试
# 场景2：随机环境 无限次执行 观察内存占用和执行时间

import random

import gym
from gym import register
import timeit
import psutil

register(
    id='TinyMarketGymEnvRandom-v0',
    entry_point='tiny_market:GymEnvRandom',
    max_episode_steps=3600,  # 一个episode最大步数
    reward_threshold=1000000.0,
)


def train():
    env = gym.make('TinyMarketGymEnvRandom-v0')
    env.init(capital=20000,
             file_path="dominant_processed_data_20170103_20220215.h5",
             date_start="20211001", date_end="20211231")

    env.reset()
    count = 0
    last_time = timeit.default_timer()
    total_reward = 0
    reset_count = 0
    while True:
        rand = random.randint(0, 100)
        action = 0
        if rand > 95:
            action = 1
        elif rand < 5:
            action = 2
        ob, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            env.reset()
            reset_count += 1
        count += 1
        if count % 10000 == 0:
            avg_reward = total_reward / 10000
            total_reward = 0
            mem = psutil.virtual_memory()
            time_diff = timeit.default_timer() - last_time
            print("Step Count : %d , Mem : %d / %d (%.4f), Time diff : %f, Avg Reward : %f, Reset pre step : %f"
                  % (count, mem.used, mem.total, mem.used / mem.total, time_diff, avg_reward,
                     10000 / (reset_count + 1e-9)))
            last_time = timeit.default_timer()
            reset_count = 0


if __name__ == '__main__':
    gym.logger.setLevel(gym.logger.WARN)
    train()
    print("Done.")
