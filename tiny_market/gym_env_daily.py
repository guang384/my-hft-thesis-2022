import os
import random

import pandas as pd
from gym import logger

from tiny_market.gym_env_base import GymEnvBase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pd.set_option('display.float_format', lambda x: '%.10f' % x)  # 为了直观的显示数字，不采用科学计数法


class GymEnvDaily(GymEnvBase):

    # 按照顺序
    def __init__(self):
        super().__init__()
        self.current_day_index = None

    def current_day(self):
        return self.possible_days[self.current_day_index]

    def set_capital(self, capital):
        self.capital = capital

    # 从头开始逐日进行
    def pick_day_when_reset(self):
        if self.current_day_index is None:
            self.current_day_index = 0
            return self.possible_days[self.current_day_index]
        else:
            if self.current_day_index < len(self.possible_days) - 1:
                self.current_day_index += 1
            else:
                logger.warn("Days end !!!")
                self.done = True
            return self.possible_days[self.current_day_index]

    # 从头开始
    def pick_start_index_and_time_when_reset(self):
        return 600

    # 直到收盘才结束 (不能开仓了且已经清仓
    def if_done_when_step(self):
        if_end = self.time > self.TIME_ONLY_CLOSE and self.current_position == 0
        return if_end or self.current_observation_index >= self.max_observation_index
