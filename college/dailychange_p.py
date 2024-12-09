import datetime as datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse as argparse

def get_file_path(sol):
    '''
    指定したsolに対応する、ファイル名を作成する関数
    sol:探査機到着後からの経過日数(火星日)
    '''
    directory = '/home/takada/2025B_takada/work/git/solall/'
    return os.path.join(directory, f'ps_calib_{str(sol).zfill(4)}.csv')

# データを読み込む関数
def load_data(file_pass):
    '''
    指定したcsvファイルを読み込む関数
    file_pass:読み込むファイルのpass
    '''
    try:
        data = pd.read_csv(file_pass, skiprows=1, usecols=[0, 1, 2, 3, 4], 
                            names=["MUTC", "LMST", "LTST", "UTC", "p"], parse_dates=[0])
        data["UTC"] = pd.to_datetime(data["UTC"], format="%Y-%jT%H:%M:%S.%fZ")
        return data
    
    except FileNotFoundError:
        return None

def process_dailydata_p(sol):
    '''
    指定したsolに対応する、気圧変化の時系列データ(dataframe)を返す関数
    sol:探査機到着後からの経過日数(火星日)
    '''
    # sol-1, sol, sol+1に対応するデータ
    dataframes = [load_data(get_file_path(sol_offset)) for sol_offset in [sol - 1, sol, sol + 1]]

    # solでフィルタリング
    filtered_dataframes = [
        data[data['LTST'].str[:5] == str(sol).zfill(5)]
        for data in dataframes if data is not None
    ]

    #結合及び時刻の計算
    if filtered_dataframes:
        dataframe = pd.concat(filtered_dataframes, ignore_index=True)
        dataframe = dataframe.drop_duplicates(subset=['MUTC'], keep='first')
        dataframe['Time'] = dataframe['MUTC'].dt.time
        
        return dataframe
    else:
        return None

def plot_dailychange_p(sol):
    '''
    指定したsolに対応する、気圧変化の時系列データ(dataframe)の描画を画像として保存する関数
    sol:探査機到着後からの経過日数(火星日)
    '''
    try:
        #データの取得
        data = process_dailydata_p(sol)
        if data is None:
            raise ValueError(f"No data:sol={sol}")
         
        #プロットの設定
        plt.tight_layout()
        data.plot(x='Time',y='p')
        plt.title(f'sol={sol}')
        plt.xlabel('Time')
        plt.ylabel('Pressure [Pa]')
        plt.grid(True)
        
        #保存の設定
        output_dir = 'dailychange_p'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)}_dailychange.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)}_dailychange.png")
        
        return data
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot daily pressure change")
    parser.add_argument('sol', type=int, help="sol") #solの指定
    args = parser.parse_args()
    plot_dailychange_p(args.sol) #指定されたsolにおける気圧変化の時系列データを描画