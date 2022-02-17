from tiny_market import preprocessing_tick_data

if __name__ == '__main__':
    preprocessing_tick_data(
        tick_data_file_path='dominant_tick_data_20170103_20220215.h5',
        to_file_path='dominant_processed_data_20170103_20220215.h5')
    print("end")
