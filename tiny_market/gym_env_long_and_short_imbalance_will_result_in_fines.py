import os
from decimal import Decimal

import numpy as np
import pandas as pd
from gym import logger

from .gym_env_base import GymEnvBase

"""
智能体多空操作应该趋于一致 （超时会平仓 保证金不足会自自动限仓
"""


class GymEnvLongAndShortImbalanceWillResult(GymEnvBase):

    def _pick_day_when_reset(self):
        return super(GymEnvLongAndShortImbalanceWillResult, self)._pick_day_when_reset()

    def _pick_start_index_and_time_when_reset(self):
        return super(GymEnvLongAndShortImbalanceWillResult, self)._pick_start_index_and_time_when_reset()

    def _if_done_when_step(self):
        return super(GymEnvLongAndShortImbalanceWillResult, self)._if_done_when_step()

    def __init__(self, **kwargs):
        assert 'fine_pre_seconds' in kwargs.keys(), 'Parameter [fine_pre_seconds] must be specified.'
        self.fine_pre_seconds = float(kwargs['fine_pre_seconds'])
        self.action_1 = 0
        self.action_2 = 0
        self.current_imbalance = 0
        super().__init__(**kwargs)

    def reset(self):
        self.action_1 = 0
        self.action_2 = 0
        self.current_imbalance = 0
        return super().reset()

    def step(self, action):
        action = int(action)
        if action == 1:
            self.action_1 += 1
        elif action == 2:
            self.action_2 += 1
        self.current_imbalance = Decimal(np.abs(self.action_1 - self.action_2) / 100)
        return super().step(action)

    def _observation(self):
        ob = super()._observation()
        return np.array(ob + float(self.current_imbalance))

    def _calculate_reward(self):
        reward = super()._calculate_reward()
        # 只要不平衡就一直扣分
        return reward - self.current_imbalance
