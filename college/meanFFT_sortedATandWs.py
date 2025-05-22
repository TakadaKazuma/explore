import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import os
import argparse as argparse
import DATACATALOG
import dailychange_p
import neardevil
import nearFFT
import meanFFT_sortedseason
import meanmovingFFT_sorteddP
from Dispersion_Relation import Params

def process_IDlist_ATandWs(AT_Llimit, Ws_Ulimit):
    '''
    datacatalog から AT-ave>AT_Llimit かつ Ws-ave<Ws_Ulimit を満たすIDのリストを作成する関数

    AT_Llimit : 基準となる大気の温度(K) (int)
    Ws_Ulimit : 基準となる風速(m/s) (int)
    '''
    datacatalog = DATACATALOG.process_datacatalog()
    filtered_list = datacatalog[(datacatalog['AT-ave'] > AT_Llimit)&(datacatalog['Ws-ave'] < Ws_Ulimit)]
    return filtered_list['ID'].tolist()


def process_FFTlist_ATandWs(AT_Llimit, Ws_Ulimit, timerange, interval):
    '''
    AT-ave>AT_Llimit かつ Ws-ave<Ws_Ulimit を満たす全ての ID に対応する、
    MUTC (ダストデビル発生時刻) 直前の時系列データにおける気圧残差を求め、
    各ケースのパワースペクトルをまとめたリストを返す関数。

    AT_Llimit : 基準となる大気の温度(K) (int型)
    Ws_Ulimit : 基準となる風速(m/s) (int型)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''

    # 記録用配列の作成
    fft_xlist, fft_ylist = [], []

    # AT-ave>AT_Llimit かつ Ws-ave<Ws_Ulimit を満たすIDをリスト化
    IDlist = process_IDlist_ATandWs(AT_Llimit, Ws_Ulimit)
    for ID in tqdm(IDlist, desc="Processing IDs"):
        try:
            # ID に対応する sol および MUTC を取得
            sol, MUTC = neardevil.get_sol_MUTC(ID)

            #該当sol付近の時系列データを取得
            data = dailychange_p.process_surround_dailydata(sol)
            if data is None:
                raise ValueError("Failed to retrieve time-series data.")
            
            # MUTC 付近の時系列データを取得
            near_devildata = neardevil.filter_neardevildata(data, MUTC, timerange, interval)
            if near_devildata is None or near_devildata.empty:
                raise ValueError("No data available after filtering.")

            #0.5秒間隔でresample
            near_devildata = meanFFT_sortedseason.data_resample(near_devildata, 0.5)
                
            # 残差計算を実施
            near_devildata = nearFFT.calculate_residual(near_devildata)
            '''
            追加カラム:
            - countdown: 経過時間 (秒) (countdown ≦ 0)
            - p-pred: 線形回帰による気圧予測値 (Pa)
            - residual: 気圧の残差
            '''

            # FFT によるパワースペクトルの導出
            fft_x, fft_y =  nearFFT.FFT(near_devildata)

            # 結果を配列に記録
            fft_xlist.append(fft_x)
            fft_ylist.append(fft_y)
            
        except ValueError as e:
            print(e)
            continue 
            
    return fft_xlist, fft_ylist

def plot_meanFFT_ATandWs(AT_Llimit, Ws_Ulimit, timerange, interval):
    '''
    AT-ave>AT_Llimit かつ Ws-ave<Ws_Ulimit を満たす全ての ID に対応する、
    MUTC (ダストデビル発生時刻) 直前の時系列データにおける気圧残差を求め、
    各ケースのパワースペクトルの平均を算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : スペクトル強度 (Pa^2)

    AT_Llimit : 基準となる大気の温度(K) (int型)
    Ws_Ulimit : 基準となる風速(m/s) (int型)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''
    try:
        # 各ケースにおけるパワースペクトルをまとめたリストの導出
        fft_xlist, fft_ylist = process_FFTlist_ATandWs(AT_Llimit, Ws_Ulimit, timerange, interval)
        
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
        plt.title(f'MPS_AT>{AT_Llimit},Ws<{Ws_Ulimit}, time_range={timerange}(s)')
        plt.xlabel('Vibration Frequency [Hz]',fontsize=15)
        plt.ylabel('Pressure Amplitude [Pa]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanFFT_sortedATandWs_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"AT is More{AT_Llimit},Ws is less{Ws_Ulimit}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return fft_x, fft_y

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('AT_Ulimit', type=int, 
                        help="Serves as the standard for the upper limit of AT_ave(K)") # AT_aveの上限の指定
    parser.add_argument('Ws_Llimit', type=int, 
                        help='Serves as the standard for the upper limit of Ws_ave(m/s)') # Ws_aveの下限の指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # り取る時間範囲(秒)
    args = parser.parse_args()
    plot_meanFFT_ATandWs(args.AT_Ulimit, args.Ws_Llimit, args.timerange, 20)