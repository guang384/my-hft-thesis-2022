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
