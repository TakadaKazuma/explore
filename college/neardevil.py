import numpy as np
import datetime as datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse as argparse
import DATACATALOG
import dailychange_p

#IDに対応するsol,MUTCを返す関数
def get_sol_MUTC(ID):
    '''
    
    '''
    database = DATACATALOG.process_database()
    sol = database.sol[ID]
    MUTC = database.MUTC[ID]
    return sol, MUTC

def filter_neardevildata(data, MUTC, ranges, interval):
    '''

    '''
    stop = MUTC - datetime.timedelta(seconds=interval)
    start = stop - datetime.timedelta(seconds=ranges)
    filtered_data = data.query('@start < MUTC < @stop').copy()
    
    return filtered_data

#データから気圧と時刻の線形回帰を行い、残差等も追加したデータを返す関数
def caluculate_residual(data):
    '''

    '''
    new_data = data.copy()
    
    #時間(秒)の計算
    new_data["countdown"] = - (new_data['MUTC'].iloc[-1] - new_data['MUTC']).dt.total_seconds()
    
    #p及びcountdownカラムにNanのある行を削除
    new_data = new_data.dropna(subset=['p', 'countdown'])
    
   #線形回帰
    t, p = new_data['countdown'].values, new_data['p'].values
    a, b = np.polyfit(t, p, 1)
    
    new_data["p-pred"] = a*t + b
    new_data["residual"] = p - new_data['p-pred']
    
    return new_data

#IDに対応するダストデビル発生直前～寸前のデータを作成する関数
def process_neardevildata(ID, ranges, interval):
    '''

    '''
    try:
        #IDに対応するsol及びMUTCを取得
        sol, MUTC = get_sol_MUTC(ID)
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("")
        
        near_devildata = filter_neardevildata(data, MUTC, ranges, interval)
        if near_devildata is None:
            raise ValueError("")
        
        return near_devildata, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#IDに対応するダストデビル発生直前～寸前の気圧変化を描画する関数(横軸:カウントダウン)
def plot_neardevil(ID, ranges, interval):
    '''

    '''
    try:
        near_devildata, sol = process_neardevildata(ID, ranges, interval)
        if near_devildata is None:
            raise ValueError(f"No data:sol={sol}")

        #線形回帰
        near_devildata = caluculate_residual(near_devildata)

        #プロットの設定
        plt.plot(near_devildata['countdown'],near_devildata['p'],
                 label='true_value')
        plt.plot(near_devildata['countdown'],near_devildata['p-pred'],
                 label='predict-value')
        plt.xlabel('Time until devil starts [s]')
        plt.ylabel('Pressure [Pa]')
        plt.title(f'ID={ID},sol={sol}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'neardevil_{ranges}s'
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