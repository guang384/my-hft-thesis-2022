# 直接以收益作为奖励
def profits_or_loss_reward(records):
    return records['amount'][-1]


# 收益考虑罚金
def profits_or_loss_with_fine_reward(records):
    return records['amount'][-1] - records['fine'][-1]


# 夏普比率
def sharpe_ratio_reward(records):
    raise NotImplementedError


# 索提诺比率
def sortino_ratio_reward(records):
    raise NotImplementedError
