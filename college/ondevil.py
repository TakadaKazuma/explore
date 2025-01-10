import numpy as np
import datetime as datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse as argparse
import DATACATALOG
import dailychange_p
import neardevil

def filter_ondevil(data, MUTC, timerange):
    '''
    与えられた時系列データを「MUTCのtime_range秒前」~ 「MUTC」の区間で切り取り、そのデータを返す関数(dataframe型)
    →与えられた時系列データを「dustdevilの発生直前 ~ 発生」までの区間で切り取り、そのデータを返す関数(dataframe型)

    data:気圧の時系列データ(dataframe)
    MUTC:dustdevil発生時刻※ 火星地方日時 (datetime型)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    '''
    stop = MUTC 
    start = stop - datetime.timedelta(seconds=timerange)
    filtered_data = data.query('@start < MUTC < @stop').copy()
    
    return filtered_data

def process_ondevil(ID, timerange):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生」における気圧の時系列データを返す関数

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り取る時間)(秒)(int型)
    '''
    try:
        #IDに対応するsol及びMUTCを取得
        sol, MUTC = neardevil.get_sol_MUTC(ID)

        #該当sol付近の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("")
        
        #該当時系列データをdustdevil近辺でフィルタリング
        on_devildata = filter_ondevil(data, MUTC, timerange)
        if on_devildata is None:
            raise ValueError("")
        
        #「countdown」カラムの追加
        on_devildata = neardevil.calculate_countdown(on_devildata)
        
        return on_devildata, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_ondevil(ID, timerange):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生」における気圧の時系列データを描画した画像を保存する関数
    ※横軸:countdown(s) 縦軸:気圧(Pa)

    ID:通し番号
    timerange:時間間隔(切り取る時間)(秒)(int型)
    '''
    try:
        #描画する時系列データの取得
        on_devildata, sol = process_ondevil(ID, timerange)
        
        if on_devildata is None:
            raise ValueError(f"No data:sol={sol}")
        
        #描画の設定
        plt.plot(on_devildata['countdown'],on_devildata['p'])
        plt.xlabel('Time until devil [s]')
        plt.ylabel('Pressure [Pa]')
        plt.title(f'ID={ID},sol={sol}')
        plt.grid(True)    
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'ondevil_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_ondevil.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_ondevil.png")
        
        return on_devildata
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pressure changes corresponding to the ID")
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerange(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_ondevil(args.ID, args.timerange)