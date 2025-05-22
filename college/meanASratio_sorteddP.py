import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearFFT
import nearratio
import nearmovingAS
import meanFFT_sorteddP
import meanFFT_sortedseason
import meanmovingFFT_sorteddP
from Dispersion_Relation import Params

def process_ASratiolist_dP(dP_Ulimit, timerange, interval, windowsize_FFT):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データにおける気圧残差を求め、
    各ケースの振幅スペクトル比をまとめたリストを返す関数。

    dP_Ulimit:上限となる気圧降下量(Pa) (int)
    ※dP, dP_Ulimit < 0
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均に用いる窓数(int)
    '''
    # 記録用配列の作成
    moving_AS_xlist, ratiolist = [], []
    
    # dP_Ulimit > dP を満たす ID をリスト化
    IDlist = meanFFT_sorteddP.process_IDlist_dP(dP_Ulimit)
    for ID in tqdm(IDlist, desc="Processing IDs"):
        try:
            # ID に対応する sol および MUTC を取得
            sol, MUTC = neardevil.get_sol_MUTC(ID)

            # 該当sol付近の時系列データを取得
            data = dailychange_p.process_surround_dailydata(sol)
            if data is None:
                raise ValueError("Failed to retrieve time-series data.")
            
            # MUTC 付近の時系列データを取得
            near_devildata = neardevil.filter_neardevildata(data, MUTC, timerange, interval)

            # 0.5秒間隔でresample
            near_devildata = meanFFT_sortedseason.data_resample(near_devildata, 0.5)
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
            AS_x, AS_y ,moving_AS_x, moving_AS_y =  nearmovingAS.movingAS(near_devildata, windowsize_FFT)

            # 振幅スペクトル比の算出
            moving_AS_x, ratio = nearratio.calculate_ratio(AS_x, AS_y, moving_AS_y, windowsize_FFT)

            # 結果を配列に記録
            moving_AS_xlist.append(moving_AS_x)
            ratiolist.append(ratio)
            
        except ValueError as e:
            print(e)
            continue 
            
    return moving_AS_xlist, ratiolist

def plot_meanASratio_dP(dP_Ulimit, timerange, interval, windowsize_FFT):
    '''
    dP_Ulimit > dP を満たす全ての ID に対応する、
    MUTC (ダストデビル発生時刻) 直前の時系列データにおける気圧残差を求め、
    各ケースの振幅スペクトル比の平均を算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : 振幅スペクトル比 (/)

    dP_Ulimit:上限となる気圧降下量(Pa) (int)
    ※dP, dP_Ulimit < 0
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均に用いる窓数(int)
    '''
    try:
        # 各ケースの振幅スペクトル比をまとめたものの導出
        moving_AS_xlist, ratiolist = process_ASratiolist_dP(dP_Ulimit, timerange, interval, windowsize_FFT)
        
        # 振幅スペクトル比のケース平均を算出
        moving_AS_x = meanmovingFFT_sorteddP.process_arrays(moving_AS_xlist, np.nanmean)
        ratio = meanmovingFFT_sorteddP.process_arrays(ratiolist, np.nanmean)
        
        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.plot(moving_AS_x, ratio, label='ASR')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MASR_dP>{-dP_Ulimit}.timerange={timerange}s', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Amplitude Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanASratio_dP_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"dP is More {-dP_Ulimit},windowsize_FFT={windowsize_FFT}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return moving_AS_x, ratio

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") # IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average of FFT") # 振幅スペクトルの移動平均に用いる窓数
    args = parser.parse_args()
    plot_meanASratio_dP(args.dP_Ulimit, args.timerange, 20, args.windowsize_FFT)