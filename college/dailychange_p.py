import datetime as datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse as argparse

def get_file_path(sol):
    '''
    指定されたsolに対応する、ファイルパスを作成する関数
    
    sol:取り扱う火星日(探査機到着後からの経過日数)(int型)
    '''
    directory = '/home/takada/2025B_takada/work/git/solall/'
    return os.path.join(directory, f'ps_calib_{str(sol).zfill(4)}.csv')

def load_data(file_pass):
    '''
    指定されたCSVファイルを読み込む関数

    file_pass: 読み込むファイルのパス
    '''
    try:
        data = pd.read_csv(file_pass, skiprows=1, usecols=[0, 1, 2, 3, 4], 
                           names=["MUTC", "LMST", "LTST", "UTC", "p"], parse_dates=[0])
        data["UTC"] = pd.to_datetime(data["UTC"], format="%Y-%jT%H:%M:%S.%fZ")
        return data

    except FileNotFoundError:
        print(f"Error: The file '{file_pass}' was not found.")
        return None

def process_dailydata_p(sol):
    '''
    指定されたsolに対応する気圧変化の時系列データ(DataFrame)を取得する関数

    - 指定されたsolの前後1sol（sol-1, sol, sol+1）のデータを取得
    - `LTST`（地方太陽時）をもとに、指定されたsolのデータのみをフィルタリング
    - `MUTC`（火星協定時）で重複データを削除
    - `Local Time`（火星地方時刻）を追加

    sol: 取り扱う火星日 (探査機到着後からの経過日数) (int型)
    '''
    # sol-1, sol, sol+1 に対応するデータを取得
    dataframes = [load_data(get_file_path(sol_offset)) for sol_offset in [sol - 1, sol, sol + 1]]

    # sol に対応するデータのみをフィルタリング
    filtered_dataframes = [
        data[data['LTST'].str[:5] == str(sol).zfill(5)]
        for data in dataframes if data is not None
    ]

    # データを結合し、重複を削除
    if filtered_dataframes:
        dataframe = pd.concat(filtered_dataframes, ignore_index=True)
        dataframe = dataframe.drop_duplicates(subset=['MUTC'], keep='first')
        dataframe['Local Time'] = dataframe['MUTC'].dt.time  # 火星地方時刻を追加
        
        return dataframe
    else:
        return None


def process_surround_dailydata(sol):
    '''
    指定されたsolとその前後1sol（sol-1, sol, sol+1）を含む
    気圧変化の時系列データ (DataFrame) を取得する関数

    - 指定されたsolの前後1solのデータを取得
    - データを結合し、`MUTC`（火星協定時）で重複を削除
    - `Local Time`（火星地方時刻）を追加

    sol: 取り扱う火星日 (探査機到着後からの経過日数) (int型)
    '''
    # sol-1, sol, sol+1 に対応する全データを取得
    dataframes = [load_data(get_file_path(sol_offset)) for sol_offset in [sol - 1, sol, sol + 1]]
    
    # None ではないデータのみをリストに格納
    valid_dataframes = [dataframe for dataframe in dataframes if dataframe is not None]

    # データを結合し、重複を削除
    if valid_dataframes:
        dataframe = pd.concat(valid_dataframes, ignore_index=True)
        dataframe = dataframe.drop_duplicates(subset=['MUTC'], keep='first')
        dataframe['Local Time'] = dataframe['MUTC'].dt.time  # 火星地方時刻を追加
        
        return dataframe
    else:
        return None

def plot_dailychange_p(sol):
    '''
    指定したsolに対応する気圧変化の時系列データをプロットし、画像を保存する関数

    - 指定されたsolの気圧データを取得
    - `Local Time` (火星地方時刻) を横軸、`p` (気圧) を縦軸にプロット
    - プロット画像を `dailychange_p` フォルダに保存

    sol: 取り扱う火星日 (探査機到着後からの経過日数) (int型)
    '''
    try:
        # データの取得
        data = process_dailydata_p(sol)
        if data is None or data.empty:
            raise ValueError(f"No data available for sol={sol}")
        
        # プロットの設定
        plt.figure(figsize=(10, 5))  # プロットサイズを調整
        plt.plot(data['Local Time'], data['p'], label='Pressure [Pa]')
        plt.title(f'sol={sol}', fontsize=15)
        plt.xlabel('Local Time', fontsize=15)
        plt.ylabel('Pressure [Pa]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 画像の保存
        output_dir = 'dailychange_p'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"sol={str(sol).zfill(4)}.png" 
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return data
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sol', type=int, help="sol") #solの指定
    args = parser.parse_args()
    plot_dailychange_p(args.sol) #指定されたsolにおける気圧変化の時系列データを描画