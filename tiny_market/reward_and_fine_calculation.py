import pandas as pd
import numpy as np

# 直接以收益作为奖励
def profits_or_loss_reward(records):
    return records['amount'][-1]



# 日夏普比率
def sharpe_ratio_reward(records):
    r = pd.DataFrame(records['amount']).pct_change()
    seconds_of_watching = np.timedelta64(self.time - self.start_time, 's').astype('int')

    tick_sr =  r.mean() / r.std() * np.sqrt(len(records['amount']))
    # Tick夏普到日夏普转换因子
    return tick_sr * np.sqrt(20700/seconds_of_watching)    # 按照每天交易时间5小时45分钟(20700秒)

# 索提诺比率
def sortino_ratio_reward(records):
    raise NotImplementedError


# 观望罚金 不惩罚
def none_fine(seconds_of_watching):
    return 0


# 观望罚金 随着观望时间增长线性增加
def linear_fine(cost):
    def fine_func(seconds_of_watching):
        return seconds_of_watching * cost
    return fine_func
