import os
from decimal import Decimal

import numpy as np
import pandas as pd
from gym import logger

from .gym_env_base import GymEnvBase


class GymEnvDoneIfUndermargined(GymEnvBase):

    def _observation(self):
        ob = super()._observation()

        # 保证金充足率
        adequacy = 1
        if self.current_position_info is not None:
            adequacy = self.current_position_info['free_margin'] / self.margin_pre_lot

        return np.append(ob, float(adequacy))

    def _if_done_when_step(self):
        if self.undermargined_count > 0:
            self.huge_blow = True
            return True  # 如果保证金不足还想开仓 直接结束
        return super(GymEnvDoneIfUndermargined, self)._if_done_when_step()

    def _pick_day_when_reset(self):
        return super(GymEnvDoneIfUndermargined, self)._pick_day_when_reset()

    def _pick_start_index_and_time_when_reset(self):
        return super(GymEnvDoneIfUndermargined, self)._pick_start_index_and_time_when_reset()

    def _calculate_reward(self):
        return super(GymEnvDoneIfUndermargined, self)._calculate_reward()
