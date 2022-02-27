"""
基于新数据进行训练
"""

import numpy as np
import pandas as pd
from gym import register
from step08_train_model import try_train
from step10_statistics_execution_time import train_and_evaluate_mp_async_timed
from gym import envs

from tiny_market import GymEnvFeatureScaling, GymEnvRandom, GymEnvLongAndShortImbalanceWillResultInFines, \
    GymEnvWaitAndSeeWillResultInFines

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


# 依赖关系是从左往右（先调用自己内部的然后遇到super从左往右，直到没有super为止
class GymEnvRandomIndData(GymEnvLongAndShortImbalanceWillResultInFines,
                          GymEnvWaitAndSeeWillResultInFines,
                          GymEnvFeatureScaling, GymEnvRandom):
    pass


if __name__ == '__main__':
    try_train(file_path="data/dominant_reprocessed_data_202111_ind.h5",
              date_start='20211115', date_end='20211121', agent_name='dqn',
              env_name=ENV_NAME, train_and_evaluate_func=train_and_evaluate_mp_async_timed)
    print("end")
