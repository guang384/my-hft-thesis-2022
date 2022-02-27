import os
import pandas as pd
import h5py
import hdf5plugin
import numpy as np
import datetime

'''

RQSDK 获取主力合约信息

>> import pandas as pd
>> import rqdatac
>> rqdatac.init()
>> rqdatac.futures.get_dominant('M')
>> pd.DataFrame(rqdatac.futures.get_dominant('M')).to_csv('dominant_contract.csv')

主力合约信息保存为CSV：
----dominant_contract.csv ---
| date,dominant             |
| 2004-01-05,M0405          |
| 2004-01-06,M0405          |
| 2004-01-07,M0405          |
| 2004-01-08,M0405          |
| ...                       |
-----------------------------

RQSDK 下载合约Tick数据

rqsdk update-data -c 1 --tick M

合约以H5格式保存于数据目录，默认路径  '~/.rqalpha-plus/bundle/ticks'

'''


def export_tick_data_from_rqdata_dir(
        start_date,  # 导出数据开始的日期 exp:'2010-01-01'
        dominant_csv='data_sample/dominant_contract.csv',  # 主力合约信息CSV文件
        to_file_name_prefix='data/dominant_tick_data',  # 输出文件名前缀
        data_dir_path=None,  # 默认为 rqdata 目录下读取数据
):
    dominants = pd.read_csv(dominant_csv)

    # 指定日期之后的主力合约
    contracts = dominants[dominants['date'] >= start_date]['dominant'].unique()  # 结果是 numpy.ndarray 类型
    dominant_last_date = dominants.iloc[-1]['date']  # 记录明确是主力合约的最后一天
    # 去 rqdata 目录下读取数据
    if data_dir_path is None:
        data_dir_path = os.path.expanduser('~/.rqalpha-plus/bundle/ticks')

    # 获取开始日期和结束日期（指定日期开始的第一个合约的第一天和拥有数据的最后一个合约的最后一天
    for contract in contracts:
        first_contract_file_path = os.path.join(data_dir_path, "%s.h5" % contract)
        if os.path.isfile(first_contract_file_path):
            break

    with h5py.File(first_contract_file_path, 'r') as file:
        days = pd.to_datetime(list(file.keys()))
        first_date = days[days >= start_date][0].strftime('%Y%m%d')

    for contract in contracts[::-1]:  # 倒叙
        last_contract_file_path = os.path.join(data_dir_path, "%s.h5" % contract)
        if os.path.isfile(last_contract_file_path):
            last_date = dominants[dominants['dominant'] == 'M1009'].iloc[-1]['date'].replace('-', '')
            break

    # 数据存储为H5
    save = h5py.File('%s_%s_%s.h5' % (to_file_name_prefix, first_date, last_date), 'w')  # 写入文件

    for symbol in contracts:
        file_path = os.path.join(data_dir_path, "%s.h5" % symbol)
        if not os.path.isfile(file_path):
            break
        h5 = h5py.File(file_path, 'r')  # 只读打开
        keys = h5.keys()
        # 当前合约作为主力合约的日期
        days = dominants[dominants['dominant'].isin([symbol])]['date'].values  # 结果是 numpy.ndarray 类型
        for day in days:
            if pd.to_datetime(day) >= pd.to_datetime(first_date):
                print('| -- >  Exporting %s %s ...' % (symbol, day))
                day = day.replace('-', '')
                if day in keys:
                    day_data = pd.DataFrame(h5[day.replace('-', '')][()])
                    # 精简数据
                    day_data.drop(
                        columns=['a2', 'b2', 'a3', 'b3', 'a4', 'b4', 'a5', 'b5',
                                 'a2_v', 'b2_v', 'a3_v', 'b3_v', 'a4_v', 'b4_v', 'a5_v', 'b5_v'], inplace=True)
                    # 保存数据 ( 用to_records转的可以带列名 直接to_numpy 不带列名 )
                    save.create_dataset(day, data=day_data.to_records(index=False), **hdf5plugin.Blosc())
                    print('| < --  Exported dataframe [sharp = %s] !' % str(day_data.shape))
                    del day_data
                else:
                    print('| < --  No corresponding data found !')
        h5.close()
        del h5, days

    del dominants
    # 关闭文件
    save.close()
    del save


def preprocessing_tick_data(
        tick_data_file_path,  # 输入的tick文件 exp 'dominant_tick_data_20100104_20100504.h5'
        to_file_path,  # 输出处理结果文件 exp 'dominant_processed_data_20100104_20100504.h5'
):
    # 读取数据
    file = h5py.File(tick_data_file_path, 'r')  # 只读打开
    # 用作存储
    save = h5py.File(to_file_path, 'w')  # 只写打开

    # 处理每日信息
    for day in file.keys():
        print('| -- >  Processing data of  %s ...' % day)

        # 加载数据
        data = pd.DataFrame(file[day][()])

        # 用于保存处理后的数据
        processed_data = pd.DataFrame()

        # 整理时间
        time_series = data['time']
        date_series = data['date']
        datetime_series = pd.to_datetime(date_series.astype(str) + time_series.astype(str), format='%Y%m%d%H%M%S%f')

        processed_data['date'] = data['date']
        processed_data['time'] = data['time']

        last_time = datetime_series.iloc[-1]
        first_time = datetime_series.iloc[0]
        total_seconds = (last_time - first_time).total_seconds()  # 全天时间差

        trading_duration = (datetime_series - first_time).apply(datetime.timedelta.total_seconds)
        processed_data['time_progress'] = (trading_duration / total_seconds * 100).round(4)  # 0-100之间的数来表示当天时间进度

        # 处理最新价格
        last_price = data['last'] / 10000
        high_price = data['high'] / 10000
        low_price = data['low'] / 10000

        # 当前价格在当天最高价与最低价之间的位置(当最高价与最低价还有最新价等时，price_scale=0)
        price_scale = ((last_price - low_price) / (high_price - low_price + 1e-6)).round(4)
        processed_data['last_price'] = last_price
        processed_data['price_scale'] = price_scale

        # 均值和标准差
        processed_data['price_mean_1m'] = last_price.rolling(window=120).mean().round(4)
        processed_data['price_std_1m'] = last_price.rolling(window=120).std().round(4)
        processed_data['price_mean_3m'] = last_price.rolling(window=360).mean().round(4)
        processed_data['price_std_3m'] = last_price.rolling(window=360).std().round(4)
        processed_data['price_mean_5m'] = last_price.rolling(window=600).mean().round(4)
        processed_data['price_std_5m'] = last_price.rolling(window=600).std().round(4)

        # 处理订单簿
        ask = data['a1'] / 10000  # 卖
        bid = data['b1'] / 10000  # 买
        processed_data['ask'] = ask
        processed_data['bid'] = bid
        processed_data['ask_gap'] = ask - last_price  # 卖价格与当前价格差距
        processed_data['bid_gap'] = last_price - bid  # 买价格与当前价格差距
        ask_volume = data['a1_v']
        bid_volume = data['b1_v']
        processed_data['ask_volume'] = data['a1_v']
        processed_data['bid_volume'] = data['b1_v']
        ask_diff = ask.diff()
        bid_diff = bid.diff()
        ask_volume_diff = ask_volume.diff()
        bid_volume_diff = bid_volume.diff()

        # 根据Shen（2015）构建的交易量订单流不平衡（Volume Order Imbalance）指标（下称VOI）
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
        processed_data['weibi'] = (bid_volume - ask_volume) / (bid_volume + ask_volume).round(4)

        # 持仓量变化
        processed_data['open_interest_diff'] = data['open_interest'].diff().fillna(0)

        # 成交量变化
        turnover_diff = data['total_turnover'].diff().fillna(0)
        turnover = (turnover_diff / last_price).round(0) / 2
        turnover[turnover < 0] = 0  # 成交量是不断增长的如果突然一个负值肯定是因为换日了（夜盘开盘的时候） 所以直接指定为0即可
        processed_data['turnover'] = turnover

        # 均值和标准差
        processed_data['turnover_mean_1m'] = turnover.rolling(window=120).mean().round(4)
        processed_data['turnover_std_1m'] = turnover.rolling(window=120).std().round(4)
        processed_data['turnover_mean_3m'] = turnover.rolling(window=360).mean().round(4)
        processed_data['turnover_std_3m'] = turnover.rolling(window=360).std().round(4)
        processed_data['turnover_mean_5m'] = turnover.rolling(window=600).mean().round(4)
        processed_data['turnover_std_5m'] = turnover.rolling(window=600).std().round(4)

        # 保存数据 ( 用to_records转的可以带列名 直接to_numpy 不带列名 )
        save.create_dataset(day, data=processed_data.to_records(index=False), **hdf5plugin.Blosc())

        # print(processed_data)
        print('| < --  Processed data [sharp = %s] !' % str(processed_data.shape))
        del processed_data
        del data

    file.close()
    save.close()
    del file
    del save
