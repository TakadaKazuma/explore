import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import os
import argparse as argparse
import dailychange_p
import neardevil
import afterdevil
import afterFFT
import meanFFT_sorteddP
import meanFFT_sortedseason
import meanmovingFFT_sorteddP
from Dispersion_Relation import Params

def process_afterFFTlist_dP(dP_Ulimit, timerange, interval):
    '''
    dP_Ulimit > dP を満たす全ての ID に対応する、
    MUTC (ダストデビル発生時刻) 直後の時系列データにおける気圧残差を求め、
    各ケースのパワースペクトルをまとめたリストを返す関数。

    dP_Ulimit:上限となる気圧降下量(Pa) (int)
    ※dP, dP_Ulimit < 0
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''
    # 記録用配列の作成
    fft_xlist, fft_ylist = [], []

    # dP_Ulimit > dP を満たすIDをリスト化
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
            after_devildata = afterdevil.filter_afterdevildata(data, MUTC, timerange, interval)
            if after_devildata is None or after_devildata.empty:
                raise ValueError("No data available after filtering.")

            # 0.5秒間隔でresample
            after_devildata = meanFFT_sortedseason.data_resample(after_devildata, 0.5)

            # 残差計算を実施
            after_devildata = afterFFT.calculate_afterresidual(after_devildata)
            '''
            追加カラム:
            - timecount: 経過時間 (秒) (timecount ≧ 0)
            - p-pred: 線形回帰による気圧予測値 (Pa)
            - residual: 気圧の残差
            '''
        
            # FFT によるパワースペクトルの導出
            fft_x, fft_y = afterFFT.after_FFT(after_devildata)

            # 結果を配列に記録
            fft_xlist.append(fft_x)
            fft_ylist.append(fft_y)
            
        except ValueError as e:
            print(e)
            continue 
            
    return fft_xlist, fft_ylist

def plot_meanafterFFT_dP(dP_Ulimit, timerange, interval):
    '''
    dP_Ulimit > dP を満たす全ての ID に対応する、
    MUTC (ダストデビル発生時刻) 直後の時系列データにおける気圧残差を求め、
    各ケースのパワースペクトルの平均を算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : スペクトル強度 (Pa^2)

    dP_Ulimit:上限となる気圧降下量(Pa) (int)
    ※dP, dP_Ulimit < 0
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''
    try:
        # 各ケースにおけるパワースペクトルをまとめたリストの導出
        fft_xlist, fft_ylist = process_afterFFTlist_dP(dP_Ulimit, timerange, interval)
        
        # パワースペクトルのケース平均を導出
        fft_x = meanmovingFFT_sorteddP.process_arrays(fft_xlist, np.nanmean)
        fft_y = meanmovingFFT_sorteddP.process_arrays(fft_ylist, np.nanmean)
        
        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-6, 1e2)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MPS_dP>{-dP_Ulimit},timerange={timerange}s', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel(f'Pressure Power [$Pa^2$]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanafterFFT_dP_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"dP is More {-dP_Ulimit}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return fft_x, fft_y

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('dP_Ulimit', type=int, 
                        help="Serves as the standard for the upper limit of dP_ave(Negative int)") # dPの上限の指定(負)
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    args = parser.parse_args()
    plot_meanafterFFT_dP(args.dP_Ulimit, args.timerange, 20)
