"""

TinyMarket 稳定性测试
场景3：日常环境 指定日期之间范围内执行 观察内存占用和执行时间

"""

import sys
import random

import gym
from gym import register
import timeit
import psutil

from tiny_market import linear_fine

register(
    id='TinyMarketGymEnvDaily-v0',
    entry_point='tiny_market:GymEnvDaily',
    max_episode_steps=50000,  # 一个episode最大步数
    reward_threshold=1000000.0,
)


def run_test(file_path="data/dominant_processed_data_20170103_20220215.h5"):
    env = gym.make('TinyMarketGymEnvDaily-v0')
    env.init(capital=20000,
             file_path=file_path,
             date_start="20211201", date_end="20211231",
             fine_func=linear_fine(1))

    env.reset()
    print("Day start : ", env.current_day())
    count = 0
    last_time = timeit.default_timer()
    total_reward = 0
    reset_count = 0
    last_day = env.current_day()
    random.seed(timeit.default_timer())
    while True:
        count += 1
        rand = random.randint(0, 100)
        action = 0
        if rand > 95:
            action = 1
        elif rand < 5:
            action = 2
        ob, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Reward is : ", total_reward)
            print("Info is : ", info)
            total_reward = 0
            env.reset()
            print("Day change to : ", env.current_day())
            reset_count += 1
            current_day = env.current_day()
            if current_day == last_day:
                break
            else:
                last_day = current_day

        if count % 10000 == 0:
            mem = psutil.virtual_memory()
            time_diff = timeit.default_timer() - last_time
            print("Step Count : %d , Mem : %d / %d (%.4f), Time diff : %f"
                  % (count, mem.used, mem.total, mem.used / mem.total, time_diff))
            last_time = timeit.default_timer()
            reset_count = 0


if __name__ == '__main__':
    gym.logger.setLevel(gym.logger.WARN)
    if len(sys.argv) == 2:
        argv_file_path = sys.argv[1]
        run_test(argv_file_path)
    else:
        run_test()
    print("Done.")
