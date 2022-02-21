"""
结果不理想
重新使用指标
MACD KDJ RSI OBV
TSF
"""
import datetime
import sys
import talib
import h5py
import hdf5plugin
import numpy as np
import pandas as pd


def preprocessing_tick_data(
        tick_data_file_path,  # 输入的tick文件 exp 'dominant_tick_data_20100104_20100504.h5'
        to_file_path,  # 输出处理结果文件 exp 'dominant_processed_data_20100104_20100504.h5'
        date_start='20211101', date_end='20211130'
):
    # 读取数据
    file = h5py.File(tick_data_file_path, 'r')  # 只读打开
    # 用作存储
    save = h5py.File(to_file_path, 'w')  # 只写打开

    days = pd.to_datetime(list(file.keys()))
    days = days[days.isin(pd.date_range(date_start, date_end))].strftime('%Y%m%d')

    # 处理每日信息
    for day in days:
        print('| -- >  Processing data of  %s ...' % day)

        # 加载数据
        data = pd.DataFrame(file[day][()])

        # 用于保存处理后的数据
        processed_data = pd.DataFrame()

        '''
        必填信息
        '''
        # 整理时间
        processed_data['date'] = data['date']
        processed_data['time'] = data['time']

        # 最新价格
        last_price = data['last'] / 10000
        processed_data['last_price'] = last_price

        # 订单薄信息
        processed_data['ask'] = data['a1'] / 10000  # 卖
        processed_data['bid'] = data['b1'] / 10000  # 买
        processed_data['ask_volume'] = data['a1_v']
        processed_data['bid_volume'] = data['b1_v']

        '''
        扩展信息
        '''
        # 价差
        ask = data['a1'] / 10000  # 卖
        bid = data['b1'] / 10000  # 买
        processed_data['ask_gap'] = ask - last_price  # 卖价格与当前价格差距
        processed_data['bid_gap'] = last_price - bid  # 买价格与当前价格差距

        # RSI
        processed_data['rsi_2m'] = talib.RSI(last_price.to_numpy(), timeperiod=120) - 50  # 改为0为中间轴
        processed_data['rsi_2m'] = talib.RSI(last_price.to_numpy(), timeperiod=240) - 50  # 改为0为中间轴
        processed_data['rsi_5m'] = talib.RSI(last_price.to_numpy(), timeperiod=600) - 50  # 改为0为中间轴

        # 根据Shen（2015）构建的交易量订单流不平衡（Volume Order Imbalance）指标（下称VOI）
        ask = data['a1'] / 10000  # 卖
        bid = data['b1'] / 10000  # 买
        ask_volume = data['a1_v']
        bid_volume = data['b1_v']
        ask_diff = ask.diff()
        bid_diff = bid.diff()
        ask_volume_diff = ask_volume.diff()
        bid_volume_diff = bid_volume.diff()
        voi_bid = pd.Series(np.zeros(bid_diff.shape))
        voi_bid[bid_diff < 0] = 0
        voi_bid[bid_diff == 0] = bid_volume_diff
        voi_bid[bid_diff > 0] = bid_volume
        voi_ask = pd.Series(np.zeros(ask_diff.shape))
        voi_ask[ask_diff < 0] = ask_volume
        voi_ask[ask_diff == 0] = ask_volume_diff
        voi_ask[ask_diff > 0] = 0
        voi = voi_bid - voi_ask
        processed_data['voi'] = voi

        # 委比=[委买数－委卖数]/[委买数＋委卖数]×100％
        processed_data['weibi'] = ((bid_volume - ask_volume) / (bid_volume + ask_volume))

        # 万倍持仓量变化率
        processed_data['open_interest_d'] = data['open_interest'].diff().fillna(0)*10000/data['open_interest']

        # 每跳成交量的波动率（标准差比均值） Volatility
        turnover_diff = data['total_turnover'].diff().fillna(0)
        turnover = (turnover_diff / last_price).round(0) / 2
        turnover[turnover < 0] = 0  # 成交量是不断增长的如果突然一个负值肯定是因为换日了（夜盘开盘的时候） 所以直接指定为0即可
        processed_data['turnover_v_5m'] = turnover.rolling(window=300).std()/turnover.rolling(window=300).mean()

        # 保存数据 ( 用to_records转的可以带列名 直接to_numpy 不带列名 )
        save.create_dataset(day, data=processed_data.to_records(index=False), **hdf5plugin.Blosc())

        print(processed_data)
        print('| < --  Processed data [sharp = %s] !' % str(processed_data.shape))
        del processed_data
        del data

    file.close()
    save.close()
    del file
    del save


if __name__ == '__main__':
    if len(sys.argv) == 3:
        arg_tick_data_file_path = sys.argv[1]
        arg_to_file_path = sys.argv[2]
        preprocessing_tick_data(
            tick_data_file_path=arg_tick_data_file_path,
            to_file_path=arg_to_file_path)
    else:
        preprocessing_tick_data('data/dominant_tick_data_20170103_20100504.h5',
                                'data/dominant_reprocessed_data_202111_ind.h5')
    print("end")
