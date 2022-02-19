import random
import timeit

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import gym
from gym import logger
from gym import spaces

import matplotlib.pyplot as plt
import os
from .reward_and_fine_calculation import profits_or_loss_reward, none_fine

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pd.set_option('display.float_format', lambda x: '%.10f' % x)  # 为了直观的显示数字，不采用科学计数法


class GymEnvBase(gym.Env):

    def pick_day_when_reset(self):
        raise NotImplementedError

    def pick_start_index_and_time_when_reset(self):
        raise NotImplementedError

    def if_done_when_step(self):
        raise NotImplementedError

    def __init__(self):

        self.fine_func = None
        self.last_reward = None
        self.file = None
        self.file_path = None  # 文件所在位置
        self.possible_days = None  # 可选的交易日

        self.reward_func = None  # 奖励计算函数
        self.capital = 0  # 本金
        self.fine = 0  # 罚金 （空仓观望会有罚金
        self.done = False
        self.closed_pl = 0  # 平仓盈亏 Closed Trade P/L
        self.commission = 0  # 已花费手续费

        # 持仓信息  list Tuple(open_time, open_index, direction, open_price, close_price, close_time)
        self.order_list = None
        self.unclosed_order_index = 0

        self.current_position = None  # 当前头寸 多头>0 空头<0 空仓=0
        self.margin_pre_lot = None  # 当前每手保证金
        self.last_price = None  # 当前最新价
        self.transaction_data = None  # 交易数据
        self.time = None  # 当前时间
        self.start_time = None  # 开始时间

        self.current_observation_index = None  # 数据指针 指向当前数据 （从第5分钟（9：05）开始）
        self.max_observation_index = None  # 最大指针

        self.records = {
            'amount': [],
            'position': [],
            'risk': [],
            'action': []
        }

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(- float('inf'), float('inf'), (32,), dtype=np.float32)

        self.MAX_HOLD_SECONDS = 300  # 单个合约最长持仓时间（秒数） 5分钟
        self.MIN_ORDER_VOLUME = 1  # 想要开仓成功订单量的最小阈值

        self.MARGIN_RATE = 0.15  # 保证金率
        self.COMMISSION_PRE_LOT = 5  # 佣金
        self.CONTRACT_SIZE = 10  # 合约规模

        self.TIME_ONLY_CLOSE = pd.to_datetime('225500000', format='%H%M%S%f')  # 最后6分钟只平仓不开仓
        self.TIME_CLOSE_ALL = pd.to_datetime('225900000', format='%H%M%S%f')  # 最后1分钟强平所有然后结束

    def init(self,
             file_path,  # 数据文件位置
             capital,  # 初始本金
             date_start,  # 日期范围 exp:"20220105"
             date_end,  # 日期范围 exp:"20220106"
             reward_func=profits_or_loss_reward,  # 回报计算函数
             fine_func=none_fine,  # 罚金计算函数 （空仓不动可能会有惩罚
             ):
        self.capital = capital
        self.fine_func = fine_func

        self.reward_func = reward_func
        self.last_reward = 0

        self.file = h5py.File(file_path, 'r')  # 只读打开数据

        self.file_path = file_path

        days = pd.to_datetime(list(self.file.keys()))

        days = days[days.isin(pd.date_range(date_start, date_end))].strftime('%Y%m%d')
        self.possible_days = days

        days_count = len(days.values)
        logger.info("Environment initialization complete!!! \n---\nFind %d possible days: %s"
                    "\nMargin rate %.2f, commission %d, contract size %d. \nFile path: %s\n---"
                    % (days_count,
                       str(days.values) if days_count < 10 else (" ".join(str(x) for x in days.values[0:10]) + ' ...'),
                       self.MARGIN_RATE, self.COMMISSION_PRE_LOT, self.CONTRACT_SIZE, file_path))

    def reset(self):
        random.seed(timeit.default_timer())
        self.done = False
        self.fine = 0
        self.closed_pl = 0
        self.commission = 0

        del self.order_list
        self.order_list = []

        self.current_position = 0
        self.unclosed_order_index = 0

        del self.records
        self.records = {
            'amount': [self.capital],
            'position': [0],
            'risk': [0.],
            'action': [0],
            'fine': [0]
        }

        self.last_reward = self.reward_func(self.records)

        # 选取日期
        day = self.pick_day_when_reset()
        # 加载数据
        self.transaction_data = pd.DataFrame(self.file[day][()])

        logger.info("| --> Load data size : %d" % len(self.transaction_data))

        self.max_observation_index = len(self.transaction_data) - 1

        # 选取交易开始的位置
        start_index = self.pick_start_index_and_time_when_reset()

        self.current_observation_index = start_index
        self.last_price = self.transaction_data.iloc[self.current_observation_index]['last_price']
        self.margin_pre_lot = self.last_price * self.MARGIN_RATE * self.CONTRACT_SIZE
        self.start_time = pd.to_datetime(str(int(self.transaction_data.iloc[start_index]['time'])), format='%H%M%S%f')
        logger.info("| --> Market start at : %s" % self.start_time.strftime('%X'))

        observation = self._observation()

        return observation

    def set_capital(self, capital):
        self.capital = capital

    '''
    Actions:
        Type: Discrete(3)
        Num   Action
        0     保持
        1     开多 或 平空 
        2     开空 或 平多

    '''

    def step(self, action):
        action = int(action)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        assert self.current_observation_index < self.max_observation_index, "Already OVER !!!"
        if self.done:
            logger.warn("Already DONE !!!")

        last_transaction_data = self.transaction_data.iloc[self.current_observation_index]
        self.current_observation_index += 1  # 推动时间前进
        current_transaction_data = self.transaction_data.iloc[self.current_observation_index]

        # 最新价
        self.last_price = current_transaction_data['last_price']
        # 最新每手保证金
        self.margin_pre_lot = self.last_price * self.MARGIN_RATE * self.CONTRACT_SIZE
        # 当前时刻
        self.time = pd.to_datetime(str(int(current_transaction_data['time'])), format='%H%M%S%f')

        # 挂单量
        ask_volume = current_transaction_data['ask_volume']
        bid_volume = current_transaction_data['bid_volume']

        # 市价单 可成交卖价 （当前tick 和下一tick 高价）
        market_ask_price = max(last_transaction_data['ask'], current_transaction_data['ask'])
        # 市价单 可成交买价 （当前tick 和下一tick 低价）
        market_bid_price = min(last_transaction_data['bid'], current_transaction_data['bid'])

        # 根据动作调仓
        if self.time > self.TIME_CLOSE_ALL \
                or self.max_observation_index - self.current_observation_index < 100:  # 距离收盘一分钟 平所有
            for i in range(self.unclosed_order_index, len(self.order_list)):
                self._close_earliest(market_ask_price, market_bid_price, ask_volume, bid_volume)
            self.done = True  # 收盘了
        else:
            if action == 0:  # 观望 检查是否有超时需要平仓的
                for i in range(self.unclosed_order_index, len(self.order_list)):
                    open_time, open_index, direction, open_price, close_price, close_time = self.order_list[i]
                    if (self.time - open_time).total_seconds() > self.MAX_HOLD_SECONDS:  # 持仓超过最大持仓时间
                        self._close_earliest(market_ask_price, market_bid_price, ask_volume, bid_volume)
                    else:
                        break
            else:
                if self.unclosed_order_index < len(self.order_list):  # 有持仓
                    # Position info -> Tuple(open_time, open_index, direction, open_price, close_price, close_time)
                    _, _, direction, _, _, _ = self.order_list[self.unclosed_order_index]  # 持仓方向
                    if action == 1:  # 看多
                        if direction < 0:  # 持仓为空头
                            self._close_earliest(market_ask_price, market_bid_price, ask_volume, bid_volume)  # 减仓
                        else:
                            self._open_long(market_ask_price, ask_volume)  # 加多
                    elif action == 2:  # 看空
                        if direction > 0:  # 持仓为多头
                            self._close_earliest(market_ask_price, market_bid_price, ask_volume, bid_volume)  # 减仓
                        else:
                            self._open_short(market_bid_price, bid_volume)  # 加空
                else:  # 无持仓
                    if action == 1:  # 看多
                        self._open_long(market_ask_price, ask_volume)  # 加多
                    elif action == 2:  # 看空
                        self._open_short(market_bid_price, bid_volume)  # 加空

        # 仓位调整完成，检查持仓情况
        position_info = self._position_info()
        while position_info['risk'] > 0.95:  # 爆仓了 强平到不爆
            self._close_earliest(market_ask_price, market_bid_price, ask_volume, bid_volume)
            position_info = self._position_info()

        # 更新当前头寸
        self.current_position = position_info['position']

        # 记录指标变化
        self.records['amount'].append(position_info['amount'])
        self.records['position'].append(position_info['position'])
        self.records['risk'].append(position_info['risk'])
        self.records['action'].append(action)

        # 准备返回 观测状态，奖励，是否完成，和其他信息（持仓情况）
        observation = self._observation()
        # 奖励
        if len(self.order_list) == 0:
            seconds_of_watching = np.timedelta64(self.time - self.start_time, 's').astype('int')
            self.fine = self.fine_func(seconds_of_watching)
        current_reward = self.reward_func(self.records) - self.fine  # 积累奖励等于奖金减去罚款
        reward = current_reward - self.last_reward  # 单步的激励是两次激励的变化量（delta_reward）
        self.last_reward = current_reward
        # 如果完成了就一直时完成状态
        done = self.if_done_when_step() or self.done
        self.done = done
        # 附加信息
        info = position_info.copy()
        info['commission'] = self.commission
        info['trans_count'] = len(self.order_list)
        info['unclosed_index'] = self.unclosed_order_index
        info['fine'] = self.fine

        return observation, reward, done, info

    def get_order_history(self):
        return self.order_list.copy()

    def render(self, mode="human"):
        logger.info("You render the env")
        ax1 = plt.subplot(411)
        ax2 = plt.subplot(412)
        ax3 = plt.subplot(413)
        ax4 = plt.subplot(414)
        ax1.plot(self.transaction_data['last_price'])
        ax2.plot(self.transaction_data['turnover'])
        ax3.plot(self.records['amount'])
        ax4.plot(self.records['position'])
        plt.show()

    def _open_long(self, ask_price, ask_volume):
        if ask_volume < self.MIN_ORDER_VOLUME:  # 订单量不足
            logger.info('Insufficient order quantity.')

        if self.time > self.TIME_ONLY_CLOSE \
                or self.max_observation_index - self.current_observation_index < 550:  # 到了不能开仓的时间段
            logger.info('The position will not be opened '
                        'when it is less than six minutes from the end of the trading time.')
            return
        if not self._can_open_new_position():  # 检查可用资金是否允许开仓
            logger.info("Undermargined ...")
            return

        direction = 1
        # Position info -> Tuple(open_time, open_index, direction, open_price, close_price, close_time)
        self.order_list.append((self.time,
                                self.current_observation_index,
                                direction,
                                ask_price,
                                None,
                                None))  # 开多
        self.commission += self.COMMISSION_PRE_LOT  # 手续费
        logger.info("[%s] Open long on %d" % (self.time.strftime('%X'), ask_price))

    def _open_short(self, bid_price, bid_volume):
        if bid_volume < self.MIN_ORDER_VOLUME:  # 订单量不足
            logger.info('Insufficient order quantity.')
        if self.time > self.TIME_ONLY_CLOSE \
                or self.max_observation_index - self.current_observation_index < 550:  # 到了不能开仓的时间段
            logger.info('The position will not be opened '
                        'when it is less than six minutes from the end of the trading time.')
            return
        if not self._can_open_new_position():  # 检查可用资金是否允许开仓
            logger.info("Undermargined ...")
            return

        direction = -1
        # Position info -> Tuple(open_time, open_index, direction, open_price, close_price, close_time)
        self.order_list.append((self.time,
                                self.current_observation_index,
                                direction,
                                bid_price,
                                None,
                                None))  # 开空
        self.commission += self.COMMISSION_PRE_LOT  # 手续费
        logger.info("[%s] Open short on %d" % (self.time.strftime('%X'), bid_price))

    # 平最早的一单
    def _close_earliest(self, market_ask_price, market_bid_price, ask_volume, bid_volume):
        if self.unclosed_order_index >= len(self.order_list):  # 没有未平仓,则直接返回
            return
        # Position info -> Tuple(open_time, open_index, direction, open_price, close_price, close_time)
        open_time, index, direction, open_price, _, _ = self.order_list[self.unclosed_order_index]  # 第一个未平仓合约
        if direction > 0:
            close_price = market_bid_price
            if bid_volume < self.MIN_ORDER_VOLUME:  # 如果订单数不足惩罚5个点
                close_price = market_bid_price - 5
            self.closed_pl += (close_price - open_price) * self.CONTRACT_SIZE  # 更新平仓盈亏
            logger.info("[%s] Close long on %d" % (self.time.strftime('%X'), market_bid_price))
            self.order_list[self.unclosed_order_index] = (open_time, index, direction, open_price,
                                                          close_price, self.time)
        elif direction < 0:
            close_price = market_ask_price
            if ask_volume < self.MIN_ORDER_VOLUME:  # 如果订单数不足惩罚5个点
                close_price = market_ask_price + 5
            self.closed_pl += (open_price - market_ask_price) * self.CONTRACT_SIZE  # 更新平仓盈亏
            logger.info("[%s] Close short on %d" % (self.time.strftime('%X'), market_ask_price))
            self.order_list[self.unclosed_order_index] = (open_time, index, direction, open_price,
                                                          close_price, self.time)
        # 指针后移，表示当前指针所指的头寸平仓
        self.unclosed_order_index += 1

    # 观测状态 = 交易状态 + 持仓状态
    def _observation(self):
        # 获取交易数据
        transaction_state = self.transaction_data.iloc[self.current_observation_index]  # 前两位是日期数据不要
        # 获取仓位数据
        position_state = pd.Series(self._position_info())
        # 拼接数据并返回结果
        return np.array(tuple(transaction_state)[2:] + tuple(position_state))  # 前两位是日期数据不要

    def _position_info(self):
        # 持仓保证金
        margin = self.margin_pre_lot * (len(self.order_list) - self.unclosed_order_index)
        # 仓位
        position = 0
        # 持仓盈亏 Floating P/L
        floating_pl = 0  # 利润
        for i in range(self.unclosed_order_index, len(self.order_list)):
            # Position info -> Tuple(open_time, open_index, direction, open_price, close_price, close_time)
            _, _, direction, open_price, _, _ = self.order_list[i]
            delta = self.last_price - open_price
            floating_pl += direction * delta  # 只计算盈利点数
            position += direction
        floating_pl *= self.CONTRACT_SIZE  # 盈利点数乘以合约规模就是利润

        # 当前权益 = 本金（期初权益）+ 持仓盈亏  + 平仓盈亏 - 手续费
        amount = self.capital + floating_pl + self.closed_pl - self.commission
        # 可用资金 = 当前权益 - 持仓保证金
        free_margin = amount - margin
        # 风险度 = 保证金 / 当前权益
        risk = margin / amount

        return {
            'position': position,
            'floating_pl': floating_pl,
            'closed_pl': self.closed_pl,
            'amount': amount,
            'risk': risk,
            'free_margin': free_margin,
            'margin': margin
        }

    def _can_open_new_position(self):
        # 更新持仓情况
        position_info = self._position_info()
        return position_info['free_margin'] * 0.8 > self.margin_pre_lot
