"""
导出指定日期开始的主力合约交易数据
（数据源头是 RQDATA）
"""

import sys
from tiny_market import export_tick_data_from_rqdata_dir

if __name__ == '__main__':
    if len(sys.argv) == 2:
        argv_start_date = sys.argv[1]
        export_tick_data_from_rqdata_dir(argv_start_date)
    else:
        export_tick_data_from_rqdata_dir('2010-01-01', data_dir_path='data_sample')

    print("end")
