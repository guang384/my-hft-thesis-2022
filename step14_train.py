"""
基于新数据进行训练
"""
import numpy as np
import pandas as pd
from gym import spaces, register

from tiny_market import GymEnvRandom, GymEnvDaily
from step08_train_model import try_train
from step10_statistics_execution_time import train_and_evaluate_mp_async_timed
from gym import envs

ENV_NAME = 'TinyMarketGymEnvRandomInd-v0'

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

if ENV_NAME not in env_ids:
    register(
        id=ENV_NAME,
        entry_point='step14_train:GymEnvRandomIndData',
        max_episode_steps=3600,  # 一个episode最大步数
        reward_threshold=10000.0,
    )


def custom_observation(self):
    # 获取交易数据
    transaction_state = self.transaction_data.iloc[self.current_observation_index].copy()  # 前两位是日期数据不要
    # 缩放量价信息数据到10以内
    transaction_state['last_price'] = transaction_state['last_price'] / 1000
    transaction_state['ask'] = transaction_state['ask'] / 1000
    transaction_state['bid'] = transaction_state['bid'] / 1000
    transaction_state['ask_volume'] = transaction_state['ask_volume'] / 100
    transaction_state['bid_volume'] = transaction_state['bid_volume'] / 100
    # 获取仓位数据
    position_state = pd.Series(self._position_info())
    # 拼接数据并返回结果
    return np.array(tuple(transaction_state)[2:] + (position_state['position'], position_state['risk']))


class GymEnvRandomIndData(GymEnvRandom):

    def _observation(self):
        return custom_observation(self)

    def _calculate_the_fine(self):  # 直接按照时间惩罚
        seconds_of_watching = np.timedelta64(self.time - self.start_time, 's').astype('int')
        return seconds_of_watching


if __name__ == '__main__':
    try_train(file_path="data/dominant_reprocessed_data_202111_ind.h5",
              date_start='20211115', date_end='20211121', agent_name='dqn',
              env_name=ENV_NAME, train_and_evaluate_func=train_and_evaluate_mp_async_timed)
    print("end")
