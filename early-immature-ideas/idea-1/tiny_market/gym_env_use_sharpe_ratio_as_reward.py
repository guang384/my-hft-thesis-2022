import os
from decimal import Decimal

import numpy as np
import pandas as pd
from gym import logger

from .gym_env_base import GymEnvBase


class GymEnvUseSharpRatioAsReward(GymEnvBase):

    def _pick_day_when_reset(self):
        return super(GymEnvUseSharpRatioAsReward, self)._pick_day_when_reset()

    def _pick_start_index_and_time_when_reset(self):
        return super(GymEnvUseSharpRatioAsReward, self)._pick_start_index_and_time_when_reset()

    def _if_done_when_step(self):
        return super(GymEnvUseSharpRatioAsReward, self)._if_done_when_step()

    def _calculate_reward(self):
        r = pd.Series(self.records['amount']).astype('float').pct_change().fillna(0)

        tick_sr = r.mean() / (r.std()+1e-10)
        # Tick夏普到日夏普转换因子
        return Decimal(tick_sr * np.sqrt(20700 / self.seconds_from_start))  # 按照每天交易时间5小时45分钟(20700秒)
