import os
import random

import numpy as np
import pandas as pd

from .gym_env_base import GymEnvBase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pd.set_option('display.float_format', lambda x: '%.10f' % x)  # 为了直观的显示数字，不采用科学计数法


class GymEnvRandom(GymEnvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = 'Random'

    # 随机选择日期
    def _pick_day_when_reset(self):
        return np.random.choice(self.possible_days, 1)[0]

    # 随机时刻开始
    def _pick_start_index_and_time_when_reset(self):
        random_index = random.randint(600, self.max_observation_index)
        # 检查随机的开始时间是否允许开仓
        time = pd.to_datetime(str(int(self.transaction_data.iloc[random_index]['time'])), format='%H%M%S%f')
        while time > self.TIME_ONLY_CLOSE:
            random_index = random.randint(600, self.max_observation_index)
            time = pd.to_datetime(str(int(self.transaction_data.iloc[random_index]['time'])), format='%H%M%S%f')
        return random_index

    # 开过仓(有交易记录），但是当前头寸为0 或者 收盘了
    def _if_done_when_step(self):
        if_return_to_zero_position = len(self.order_list) > 0 and self.current_position_info['position'] == 0
        return if_return_to_zero_position or self.current_observation_index >= self.max_observation_index
