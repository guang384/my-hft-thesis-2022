import os
from decimal import Decimal

import numpy as np
import pandas as pd
from gym import logger

from .gym_env_base import GymEnvBase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pd.set_option('display.float_format', lambda x: '%.10f' % x)  # 为了直观的显示数字，不采用科学计数法


# 观望罚金 随着观望时间增长线性增加
class GymEnvWaitAndSeeWillResultInFines(GymEnvBase):

    def _pick_day_when_reset(self):
        return super(GymEnvWaitAndSeeWillResultInFines, self)._pick_day_when_reset()

    def _pick_start_index_and_time_when_reset(self):
        return super(GymEnvWaitAndSeeWillResultInFines, self)._pick_start_index_and_time_when_reset()

    def _if_done_when_step(self):
        return super(GymEnvWaitAndSeeWillResultInFines, self)._if_done_when_step()

    # 观望罚金 随着观望时间增长线性增加
    def __init__(self, **kwargs):
        assert 'fine_pre_seconds' in kwargs.keys(), 'Parameter [fine_pre_seconds] must be specified.'
        self.fine_pre_seconds = float(kwargs['fine_pre_seconds'])
        self.total_fine = 0
        super().__init__(**kwargs)

    def reset(self):
        self.total_fine = 0
        return super(GymEnvWaitAndSeeWillResultInFines, self).reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info['total_fine'] = self.total_fine
        return observation, reward, done, info

    def _observation(self):
        ob = super()._observation()
        return np.append(ob, float(self.total_fine/10))

    def _calculate_reward(self):
        reward = super()._calculate_reward()
        current_fine = 0
        if len(self.order_list) == 0:
            seconds_of_watching = int(self.seconds_from_start)
            total_watching_fine = round((self.fine_pre_seconds * seconds_of_watching) ** 1.5, 2)  # 1.5次幂，等的越久扣的越多
            current_fine = total_watching_fine - self.total_fine
            self.total_fine = total_watching_fine
        return reward - Decimal(str(current_fine))
