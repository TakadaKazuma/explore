import numpy as np
import datetime as datetime
from datetime import timedelta
import pandas as pd
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
import meanFFT_sorteddP
import Dispersion_Relation
import nearmovingFFT
import meanmovingFFT_sorteddP


def process_movingratiolist_dP(dP_Ulimit, timerange, interval, windowsize):
    '''
    dP_Ulimit > dP を満たす全て事象の時系列データを加工し、
    FFTを用いて導出したパワースペクトルとその移動平均の比をリスト化したものを返す関数

    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_max < 0
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''
    #記録用配列の作成
    moving_fft_xlist, ratiolist = [], []

    #dP_Ulimit > dP を満たすIDリストの作成
    IDlist = meanFFT_sorteddP.process_IDlist_dP(dP_Ulimit)
    for ID in tqdm(IDlist, desc="Processing IDs"):
        try:
            #IDに対応するsol及びMUTCを取得
            sol, MUTC = neardevil.get_sol_MUTC(ID)

            #該当sol付近の時系列データを取得
            data = dailychange_p.process_surround_dailydata(sol)
            if data is None:
                raise ValueError("")

            #該当範囲の抽出
            near_devildata = neardevil.filter_neardevildata(data, MUTC, timerange, interval)
            
            #加工済みデータを0.5秒でresample
            near_devildata = meanFFT_sortedseason.data_resample(near_devildata, 0.5)
            if near_devildata is None:
                raise ValueError("No data")
            
            near_devildata = nearFFT.calculate_residual(near_devildata)
            '''
            「countdown」、「p-pred」、「residual」カラムの追加
            countdown:経過時間(秒) ※countdown ≦ 0
            p-pred:線形回帰の結果(気圧(Pa))
            residual:残差 (Pa)
            '''

            if near_devildata is None:
                raise ValueError("No data")
            
            #パワースペクトルとその移動平均の導出
            fft_x, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.moving_FFT(near_devildata, windowsize)
            
            #比の算出
            ratio = fft_y/moving_fft_y
            
            #記録用配列に保存
            moving_fft_xlist.append(moving_fft_x)
            ratiolist.append(ratio)
            
        except ValueError as e:
            print(e)
            continue
            
    return moving_fft_xlist, ratiolist

def plot_meanmovingratio_dP(dP_Ulimit, timerange, interval, windowsize):
    '''
    dP_Ulimit > dP を満たす全て事象の時系列データを加工し、
    FFTを用いて導出したパワースペクトルとその移動平均の比の描画した画像を保存する関数

    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_max < 0
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''
    try:
        #対応する全事象のパワースペクトルとその移動平均の比をリスト化したものの導出
        moving_fft_xlist, ratiolist = process_movingratiolist_dP(dP_Ulimit, timerange, interval, windowsize)
        if not moving_fft_xlist or not ratiolist:
            raise ValueError("No data")
        
        # パワースペクトルと移動平均の比のケース平均を導出
        moving_fft_x = meanmovingFFT_sorteddP.process_arrays(moving_fft_xlist, np.nanmean)
        ratio = meanmovingFFT_sorteddP.process_arrays(ratiolist, np.mean)
        
        # 音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.plot(moving_fft_x, ratio, label='ratio')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'dP <{dP_Ulimit},time_range={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel('Pressure Amplitude Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanmovingratio_dP_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"meanmovingratio_dP_~{dP_Ulimit}_windowsize={windowsize}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: meanmovingratio_dP_~{dP_Ulimit}_windowsize={windowsize}.png")
        
        return moving_fft_x, ratio

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the average ratio of the power spectrum to its moving average for different values of dP_Ulimit")
    parser.add_argument('dP_Ulimit', type=int, help="Maximum value of dP(Pa)(Negative)") #dPの上限の指定(負)
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    #パワースペクトルの移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize', type=int, help="The [windowsize] used to calculate the moving average") 
    args = parser.parse_args()
    plot_meanmovingratio_dP(args.dP_Ulimit, args.timerange, 20, args.windowsize)