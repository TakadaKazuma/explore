import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fft
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearFFT
import nearratio
import nearmovingratio
import nearmovingAS
from Dispersion_Relation import Params

def process_ASmovingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データにおける気圧残差を求め、
    それに対する振幅スペクトル比の移動平均(修正振幅スペクトル)を算出する関数。

    ID : ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    windowsize_FFT : 振幅スペクトルの移動平均に用いる窓数(int)
    windowsize_ratio : 振幅スペクトル比の移動平均に用いる窓数(int)
    '''
    try:
        # ID に対応する sol および MUTC を取得
        sol, MUTC = neardevil.get_sol_MUTC(ID)

        # 該当 sol 周辺の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("Failed to retrieve time-series data.")

        # MUTC 付近のデータを抽出
        near_devildata = neardevil.filter_neardevildata(data, MUTC, timerange, interval)
        if near_devildata is None or near_devildata.empty:
            raise ValueError("No data available after filtering.")

        # 残差計算を実施
        near_devildata = nearFFT.calculate_residual(near_devildata)
        '''
        追加カラム:
        - countdown: 経過時間 (秒) (countdown ≦ 0)
        - p-pred: 線形回帰による気圧予測値 (Pa)
        - residual: 気圧の残差
        '''
        
        # FFT による振幅スペクトルとその移動平均の導出
        AS_x, AS_y, moving_AS_x, moving_AS_y = nearmovingAS.movingAS(near_devildata, windowsize_FFT)

        # 振幅スペクトル比の算出
        moving_AS_x, ratio = nearratio.calculate_ratio(AS_x, AS_y, moving_AS_y, windowsize_FFT)

        # 修正振幅スペクトルの算出
        moving_AS_x, moving_ratio = nearmovingratio.calculate_movingave(moving_AS_x, ratio, windowsize_ratio)

        return  moving_AS_x, moving_ratio, sol

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_ASmovingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データにおける気圧残差を求め、
    それに対する修正振幅スペクトルを算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : 修正振幅スペクトル (/)

    ID: ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    windowsize_FFT : 振幅スペクトルの移動平均に用いる窓数(int)
    windowsize_ratio : 振幅スペクトル比の移動平均に用いる窓数(int)
    '''
    try:
        # 修正振幅スペクトルの導出
        moving_AS_x, moving_ratio, sol = process_ASmovingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio)

        # 特定の周波数より高周波の情報をnanに変更
        #moving_AS_x, moving_ratio = nearratio.filter_xUlimit(moving_AS_x, moving_ratio, 0.8)

        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()

        # 描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-8,1e2)
        plt.plot(moving_AS_x, moving_ratio, label='ratio')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MASR_ID={ID}, sol={sol}, timerange={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Amplitude Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()

        # 保存の設定
        output_dir = f'nearASmovingratio_{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"ID={str(ID).zfill(5)},sol={str(sol).zfill(4)},windowsize_ratio={windowsize_ratio}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return moving_AS_x, moving_ratio

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") # IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average of FFT") # 振幅スペクトルの移動平均に用いる窓数
    parser.add_argument('windowsize_ratio', type=int, 
                        help="The [windowsize] used to calculate the moving average of ratio") # 振幅スペクトル比の移動平均に用いる窓数
    args = parser.parse_args()
    plot_ASmovingratio(args.ID, args.timerange, 20, args.windowsize_FFT, args.windowsize_ratio)