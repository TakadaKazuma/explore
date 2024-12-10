import numpy as np
import datetime as datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse as argparse
import DATACATALOG
import dailychange_p

def get_sol_MUTC(ID):
    '''
    ID(通し番号)に対応するsolとMUTCを取得する関数
    ID:通し番号(datacatalog上でdustdevilに割り振られたもの)
    '''
    datacatalog = DATACATALOG.process_database()
    sol = datacatalog.sol[ID]
    MUTC = datacatalog.MUTC[ID]
    return sol, MUTC

def filter_neardevildata(data, MUTC, time_range, interval):
    '''
    与えられたdataを「MUTCの(ranges+interval)秒前」~ 「MUTCのinterval秒前」でフィルタリングする関数
    →与えられたdataをdustdevil発生直前~発生寸前でフィルタリングする関数

    data:気圧の時系列データ(dataframe)
    MUTC:dustdevil発生時刻※ 火星地方日時 (datetime型)
    time_range:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切るか)(秒)(int型)
    '''
    stop = MUTC - datetime.timedelta(seconds=interval)
    start = stop - datetime.timedelta(seconds=time_range)
    filtered_data = data.query('@start < MUTC < @stop').copy()
    
    return filtered_data

def caluculate_countdown(data):
    '''
    フィルタリング済みのデータに事象が発生するまでの秒数を表す「countdown」のカラムを追加したものを返す関数
    ※countdown ≦ 0

    data:フィルタリングされた気圧の時系列データ(dataframe)
    '''
    new_data = data.copy()
    
    #経過時間(秒)の計算
    new_data["countdown"] = - (new_data['MUTC'].iloc[-1] - new_data['MUTC']).dt.total_seconds()
    
    return new_data

def process_neardevildata(ID, time_range, interval):
    '''
    IDに対応するdustdevil発生直前~発生寸前における気圧の時系列データ(dataframe)を返す関数

    ID:通し番号
    time_range:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''
    try:
        #IDに対応するsol及びMUTCを取得
        sol, MUTC = get_sol_MUTC(ID)

        #該当sol付近の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("")
        
        #該当時系列データをdustdevil近辺でフィルタリング
        near_devildata = filter_neardevildata(data, MUTC, time_range, interval)
        if near_devildata is None:
            raise ValueError("")
        
        #「countdown」カラムの追加
        near_devildata = caluculate_countdown(near_devildata)
        
        return near_devildata, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#IDに対応するダストデビル発生直前～寸前の気圧変化を描画する関数(横軸:カウントダウン)
def plot_neardevil(ID, time_range, interval):
    '''
    IDに対応するdustdevil発生直前~発生寸前における気圧の時系列データを描画した画像を保存する関数
    ※横軸:countdown(s) 縦軸:気圧(Pa)

    ID:通し番号
    time_range:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''
    try:
        #描画する時系列データの取得
        near_devildata, sol = process_neardevildata(ID, time_range, interval)
        if near_devildata is None:
            raise ValueError(f"No data:sol={sol}")

        #描画の設定
        plt.plot(near_devildata['countdown'],near_devildata['p'],
                 label='true_value')
        plt.xlabel('Time until devil starts [s]')
        plt.ylabel('Pressure [Pa]')
        plt.title(f'ID={ID},sol={sol}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'neardevil_{time_range}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_neardevil.png"))
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_neardevil.png")
        
        return near_devildata
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pressure changes corresponding to the ID")
    parser.add_argument('ID', type=int, help="ID")
    parser.add_argument('time_range', type=int, help='time_range(s)')
    args = parser.parse_args()
    plot_neardevil(args.ID, args.time_range, 20)