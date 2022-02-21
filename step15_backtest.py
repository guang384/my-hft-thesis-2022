"""
基于新数据进行训练
"""
import sys

import numpy as np
import pandas as pd
from gym import spaces, register

from tiny_market import GymEnvRandom
from step08_train_model import try_train
from step10_statistics_execution_time import train_and_evaluate_mp_async_timed
from gym import envs
from step09_backtesting import test_model
from step14_train import custom_observation
from tiny_market import GymEnvDaily
from gym import logger
import matplotlib.pyplot as plt


ENV_NAME = 'TinyMarketGymEnvDailyInd-v0'

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

if ENV_NAME not in env_ids:
    register(
        id=ENV_NAME,
        entry_point='step15_backtest:GymEnvDailyIndData',
        max_episode_steps=50000,  # 一个episode最大步数
        reward_threshold=10000.0,
    )


class GymEnvDailyIndData(GymEnvDaily):

    def _observation(self):
        return custom_observation(self)

    def render(self, mode="human"):
        logger.info("You render the env")
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)
        ax1.plot(self.transaction_data['last_price'])
        ax2.plot(self.records['amount'])
        ax3.plot(self.records['position'])
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 5:
        argv_actor_path = sys.argv[1]
        argv_agent_name = sys.argv[2]
        argv_test_start_date = sys.argv[3]
        argv_test_end_date = sys.argv[4]
        test_model(actor_path=argv_actor_path,
                   agent_name=argv_agent_name,
                   start_date=argv_test_start_date,
                   end_date=argv_test_end_date,
                   if_continue=True,
                   env_name=ENV_NAME,
                   file_path="data/dominant_reprocessed_data_202111_ind.h5")
    else:
        test_model(actor_path='data/modelDQN/actor_23688289_-0003.281.pth',
                   agent_name='dqn',
                   start_date="20211122",
                   end_date="20211127",
                   if_continue=True,
                   env_name=ENV_NAME,
                   file_path="data/dominant_reprocessed_data_202111_ind.h5")

    print("end")
