import os
from decimal import Decimal

import pandas as pd
from gym import logger

from .gym_env_base import GymEnvBase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pd.set_option('display.float_format', lambda x: '%.10f' % x)  # 为了直观的显示数字，不采用科学计数法


# 按照顺序 执行全天的交易
class GymEnvDaily(GymEnvBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_day_index = None

    def current_day(self):
        return self.possible_days[self.current_day_index]

    def if_all_days_done(self):
        if self.current_day_index is None:
            return self.done
        else:
            return self.current_day_index >= len(self.possible_days) - 1 and self.done

    def set_capital(self, capital):
        self.capital = Decimal(str(capital))

    # 从头开始逐日进行
    def _pick_day_when_reset(self):
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
    def _pick_start_index_and_time_when_reset(self):
        return 600

    # 直到收盘才结束 (不能开仓了且已经清仓
    def _if_done_when_step(self):
        if_no_money = self.current_position == 0 and not self._can_open_new_position()
        if_no_time = self.current_position == 0 and self.time > self.TIME_ONLY_CLOSE
        if_current_day_end = self.current_observation_index >= self.max_observation_index
        return if_no_money or if_no_time or if_current_day_end
