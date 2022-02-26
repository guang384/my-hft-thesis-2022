import os
from decimal import Decimal

import numpy as np
import pandas as pd
from gym import logger

from .gym_env_base import GymEnvBase


class GymEnvDoneIfOrderTimeout(GymEnvBase):

    def _observation(self):
        ob = super()._observation()

        # 最早的订单超时风险程度
        timeout_risk = 0
        if len(self.order_list) > 0 and self.unclosed_order_index < len(self.order_list):
            open_time, _, _, _, _, _ = self.order_list[self.unclosed_order_index]
            timeout_risk = (self.time - open_time).total_seconds() / self.MAX_HOLD_SECONDS

        return np.append(ob, timeout_risk)

    def _if_done_when_step(self):
        if self.timeout_close_count > 0:
            self.huge_blow = True
            return True  # 如果有超时单 直接结束
        return super(GymEnvDoneIfOrderTimeout, self)._if_done_when_step()

    def _pick_day_when_reset(self):
        return super(GymEnvDoneIfOrderTimeout, self)._pick_day_when_reset()

    def _pick_start_index_and_time_when_reset(self):
        return super(GymEnvDoneIfOrderTimeout, self)._pick_start_index_and_time_when_reset()

    def _calculate_reward(self):
        return super(GymEnvDoneIfOrderTimeout, self)._calculate_reward()
