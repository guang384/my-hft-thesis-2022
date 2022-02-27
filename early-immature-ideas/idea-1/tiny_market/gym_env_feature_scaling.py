import numpy as np
import pandas as pd
from .gym_env_base import GymEnvBase


class GymEnvFeatureScaling(GymEnvBase):
    """
    Feature Scaling

    Idea: Make sure features are on a similar scale.


    把价格相关数据缩放到10左右
    """

    def _observation(self):
        # 获取交易数据
        transaction_state = self.transaction_data.iloc[self.current_observation_index].copy()  # 前两位是日期数据不要
        # 缩放量价信息数据到10以内
        transaction_state['last_price'] = transaction_state['last_price'] / 1000
        transaction_state['ask'] = transaction_state['ask'] / 1000
        transaction_state['bid'] = transaction_state['bid'] / 1000
        transaction_state['ask_volume'] = transaction_state['ask_volume'] / 100
        transaction_state['bid_volume'] = transaction_state['bid_volume'] / 100
        # 获取仓位数据
        position_state = pd.Series(self._position_info())
        # 拼接数据并返回结果
        return np.array(tuple(transaction_state)[2:]
                        + (
                            position_state['position'],
                            float(position_state['risk']),
                            float(position_state['floating_pl'] / 10))
                        )

    def _pick_day_when_reset(self):
        return super(GymEnvFeatureScaling, self)._pick_day_when_reset()

    def _pick_start_index_and_time_when_reset(self):
        return super(GymEnvFeatureScaling, self)._pick_start_index_and_time_when_reset()

    def _if_done_when_step(self):
        return super(GymEnvFeatureScaling, self)._if_done_when_step()
