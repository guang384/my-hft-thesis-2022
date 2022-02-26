"""
基于新数据进行训练
"""
import numpy as np
import pandas as pd
from gym import register

from tiny_market import GymEnvRandom
from step08_train_model import try_train
from step10_statistics_execution_time import train_and_evaluate_mp_async_timed
from gym import envs

ENV_NAME = 'TinyMarketGymEnvRandomRocData-v0'

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

if ENV_NAME not in env_ids:
    register(
        id='TinyMarketGymEnvRandomRocData-v0',
        entry_point='step12_train:GymEnvRandomRocData',
        max_episode_steps=3600,  # 一个episode最大步数
        reward_threshold=1000000.0,
    )


class GymEnvRandomRocData(GymEnvRandom):

    # 通过复写修改状态数据
    def _observation(self):
        # 获取交易数据
        transaction_state = self.transaction_data.iloc[self.current_observation_index]  # 前两位是日期数据不要
        # 获取仓位数据
        position_state = pd.Series(self._position_info())
        # 拼接数据并返回结果
        return np.array(tuple(transaction_state)[2:] + (position_state['position'], float(position_state['risk'])))


if __name__ == '__main__':
    try_train(file_path="data/dominant_reprocessed_data_20211001_roc.h5",
              env_name='TinyMarketGymEnvRandomRocData-v0',
              train_and_evaluate_func=train_and_evaluate_mp_async_timed)
    print("end")
