# 直接以收益作为奖励
def profits_or_loss_reward(records):
    return records['amount'][-1]


# 夏普比率
def sharpe_ratio_reward(records):
    raise NotImplementedError


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
