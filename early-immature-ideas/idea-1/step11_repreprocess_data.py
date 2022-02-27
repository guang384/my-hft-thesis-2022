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


# 计算变化率 rate-of-change
def roc(series):
    diff = series.diff()
    roc_series = diff / series
    # 处理零值
    roc_series[diff == 0] = 0
    roc_series[series == 0] = np.sign(diff)
    return roc_series


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
        processed_data['ask_volume_roc'] = roc(processed_data['ask_volume'])
        processed_data['bid_volume_roc'] = roc(processed_data['bid_volume'])

        # TSF （用预测指标 预测的价格）
        tsf = talib.TSF(last_price.to_numpy(), timeperiod=280)
        processed_data['tsf'] = tsf

        # 成交量变化
        turnover_diff = data['total_turnover'].diff().fillna(0)
        turnover = (turnover_diff / last_price).round(0) / 2
        turnover[turnover < 0] = 0  # 成交量是不断增长的如果突然一个负值肯定是因为换日了（夜盘开盘的时候） 所以直接指定为0即可
        processed_data['turnover'] = turnover
        processed_data['turnover_roc'] = roc(processed_data['turnover'])
        # 持仓量变化
        processed_data['open_interest_diff'] = data['open_interest'].diff().fillna(0)
        processed_data['open_interest_diff_roc'] = roc(processed_data['open_interest_diff'])

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(last_price.to_numpy(),
                                                  fastperiod=120, slowperiod=260, signalperiod=90)
        processed_data['macd'] = macd
        processed_data['macd_roc'] = roc(processed_data['macd'])
        processed_data['macd_signal'] = macd_signal
        processed_data['macd_signal_roc'] = roc(processed_data['macd_signal'])
        processed_data['macd_hist'] = macd_hist
        processed_data['macd_hist_roc'] = roc(processed_data['macd_hist'])

        # RSI
        rsi = talib.RSI(last_price.to_numpy(), timeperiod=280)
        processed_data['rsi'] = rsi
        processed_data['rsi_roc'] = roc(processed_data['rsi'])

        # OBV
        obv = talib.OBV(last_price.to_numpy(), processed_data['turnover'].to_numpy())
        processed_data['obv'] = obv
        processed_data['obv_roc'] = roc(processed_data['obv'])

        # KDJ
        high = last_price.rolling(window=300).max()
        low = last_price.rolling(window=300).min()
        K, D = talib.STOCH(high.to_numpy(), low.to_numpy(), last_price.to_numpy(),
                           fastk_period=90, slowk_period=50, slowk_matype=1, slowd_period=50,
                           slowd_matype=1)  # 计算kdj的正确配置
        J = 3.0 * K - 2.0 * D
        processed_data['k'] = K
        processed_data['k_roc'] = roc(processed_data['k'])
        processed_data['d'] = D
        processed_data['d_roc'] = roc(processed_data['d'])
        processed_data['j'] = J
        processed_data['j_roc'] = roc(processed_data['j'])

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
                                'data/dominant_reprocessed_data_202111_roc.h5')
    print("end")
