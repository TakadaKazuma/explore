import datetime as datetime
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import argparse as argparse
import dailychange_p
import neardevil
import afterdevil
import afterFFT
import returnafterdevil
from Dispersion_Relation import Params
import pandas as pd

def calculate_returnafterresidual(data):
    '''
    線形回帰を適用し、気圧予測値 "p-pred" と残差 "residual" を追加する。

    data: フィルタリング済みの時系列データ (DataFrame)
    '''
    new_data = data.copy()

    # 経過時間 timecount の計算
    new_data = returnafterdevil.calculate_returntimecount(new_data)

    # 欠損値を含む行を削除
    new_data = new_data.dropna(subset=['p', 'timecount'])
    
    # 線形回帰 (説明変数: timecount, 目的変数: 気圧 p)
    t, p = new_data['timecount'].values, new_data['p'].values
    a, b = np.polyfit(t, p, 1)
    new_data["p-pred"] = a*t + b

    # 残差の計算
    new_data["residual"] = p - new_data['p-pred']

    return new_data

def process_returnafterFFT(ID, timerange, interval):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直後の
    時系列データにおける気圧残差を求め、
    それに対するパワースペクトルを算出する関数。

    ID : ダストデビルの識別番号 (int)
    timerange: 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''
    try:
        # ID に対応する sol および MUTC を取得
        sol, MUTC = neardevil.get_sol_MUTC(ID)

        # 該当 sol 周辺の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("Failed to retrieve time-series data.")
        
        # MUTC 付近のデータを抽出
        after_devildata = afterdevil.filter_afterdevildata(data, MUTC, timerange, interval)
        if after_devildata is None or after_devildata.empty:
            raise ValueError("No data available after filtering.")

        # 残差計算の実施
        after_devildata = calculate_returnafterresidual(after_devildata)
        '''
        追加カラム:
        - timecount: 経過時間 (秒) (timecount ≧ 0)
        - p-pred: 線形回帰による気圧予測値 (Pa)
        - residual: 気圧の残差
        '''

        # FFT によるパワースペクトルの導出
        fft_x, fft_y = afterFFT.after_FFT(after_devildata)

        return fft_x, fft_y, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_returnafterFFT(ID, timerange, interval):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直後の
    時系列データにおける気圧残差を求め、
    それに対するパワースペクトルを算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : スペクトル強度 (Pa^2)

    ID : ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''
    try:
        # パワースペクトルの導出
        fft_x, fft_y, sol = process_returnafterFFT(ID, timerange, interval)
        
        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        # 描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-11, 1e2)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'PS_ID={ID}, sol={sol}, timerange={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel(f'Pressure Power [$Pa^2$]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()

        # 保存の設定
        output_dir = f'returnafterFFT_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"ID={str(ID).zfill(5)}, sol={str(sol).zfill(4)}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return fft_x, fft_y

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") # IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    args = parser.parse_args()
    plot_returnafterFFT(args.ID, args.timerange, 20)