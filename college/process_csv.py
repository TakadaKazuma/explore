import pandas as pd
import datetime
import os
import argparse

def process_and_save_csv(file_path):
    '''
    ファイルのパスを受け取ると、データを加工し新たなcsvファイルを作成する関数

    '''
    data = pd.read_csv(file_path, skiprows=1, usecols=[2,3,4,5,6,7,8],
                       names=["LMST", "LTST", "UTC", "p", "p-_FREQUENCY", "p-TEMP", "p-_TEMP_FREQUENCY"], 
                       encoding="cp932")
    data["UTC_index"] = pd.to_datetime(data["UTC"], format="%Y-%jT%H:%M:%S.%fZ")
    data.set_index("UTC_index", inplace=True)
    
    # 日付と時間の処理
    split_LTST = data["LTST"].str.split(" ", expand=True)
    split_LTST_time = pd.to_datetime(split_LTST[1], format="%H:%M:%S")
    mars_day_offset = datetime.date(2018, 11, 26) - datetime.date(2018, 11, 25)
    data["Sol_Days"] = split_LTST[0].astype(int) * mars_day_offset
    data["Date"] = data["Sol_Days"] + datetime.datetime(2018, 11, 26, 0, 0, 0)
    data["Time_Formatted"] = split_LTST_time.dt.time.astype(str).values + data["UTC"].str[-5:]
    data["MUTC"] = pd.to_datetime(data["Date"].astype(str) + " " + data["Time_Formatted"], format="%Y-%m-%d %H:%M:%S.%fZ")
    data.set_index("MUTC", inplace=True)
    
    # 新しいCSVファイルのパスを作成して保存
    base = os.path.basename(file_path)[:-7]
    directory = file_path[:-20]
    output_file = os.path.join(directory, base + ".csv")
    data.to_csv(output_file)
    print(f"Save completed : {output_file}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="processes a CSV file and saves the result.")
    parser.add_argument('file_path', type=str, help="Path to the CSV file to be processed")
    args = parser.parse_args()
    output_file = process_and_save_csv(args.file_path)