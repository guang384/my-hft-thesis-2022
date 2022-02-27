"""

将导出的数据转为用于训练的市场数据的格式

字段列表：

time                  ask_gap
date                  bid_gap
time_progress         ask_volume
last_price            bid_volume
price_scale           voi
price_mean_1m         weibi
price_mean_1m         open_interest_diff
price_std_1m          turnover
price_mean_3m         turnover_mean_1m
price_std_3m          turnover_std_1m
price_mean_5m         turnover_mean_3m
price_std_5m          turnover_std_3m
ask                   turnover_mean_5m
bid                   turnover_std_5m

"""
import sys
from tiny_market import preprocessing_tick_data

if __name__ == '__main__':
    if len(sys.argv) == 3:
        tick_data_file_path = sys.argv[1]
        to_file_path = sys.argv[2]
        preprocessing_tick_data(
            tick_data_file_path=tick_data_file_path,
            to_file_path=to_file_path)
    else:
        preprocessing_tick_data('data/dominant_tick_data_20100104_20100504.h5',
                                'data/dominant_processed_data_20100104_20100504.h5')
    print("end")
