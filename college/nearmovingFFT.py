import numpy as np
import datetime as datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import argparse as argparse
import DATACATALOG
import dailychange_p
import neardevil
import Dispersion_Relation
import nearFFT

def moving_FFT(data, windowsize):
    '''
    フィルタリング及び線形回帰済みの時系列データから
    気圧変化の残差に対してFFTを用いて、
    パワースペクトル及びパワースペクトルの移動平均を返す関数

    data:フィルタリング済みの時系列データ(dataframe)
    windowsize:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''

    #パワースペクトルの導出
    fft_x, fft_y = nearFFT.FFT(data)
    
    '''
    以下ではパワースペクトルの移動平均を導出
    ※パワースペクトルの各要素及び長さは、それ自体に意味があるため、ケース平均した際にその情報が壊れないようにするため、
    正確な移動平均の値が計算できない要素にはnanを代入し、形状を維持するようにしている。

    また、今回扱うケースにおいては低周波側のパワーが高周波側より強い傾向がある。しかし、本研究では高周波側に注目したい。
    そのため、移動平均を算出する際に低周波側から受ける影響を軽減するために、
    パワーの常用対数の移動平均を算出し、対数を外して元に戻したものを、パワーの移動平均とした。
    '''
    #パワーの常用対数を算出
    log10_fft_y = np.log10(fft_y)

    filter_frame = np.ones(windowsize) / windowsize

    #窓数に対応する、片側において移動平均をできない要素の数
    pad_size = (windowsize - 1) // 2
    pad_size = (windowsize - 1) // 2
    
    #fft_x及びfft_yの要素数に統一したndarray(要素は全てnan)を作成
    moving_fft_x = np.full(fft_x.shape, np.nan)
    log10_moving_fft_y = np.full(fft_y.shape, np.nan)
    
    #移動平均を計算できる範囲のみ、その値に変更
    moving_fft_x[pad_size:-pad_size] = fft_x[pad_size:-pad_size]
    log10_moving_fft_y[pad_size:-pad_size] = np.convolve(log10_fft_y, filter_frame, mode="valid") 

    #パワーの常用対数を元に戻す
    moving_fft_y = 10**(log10_moving_fft_y)  
    
    return fft_x, fft_y ,moving_fft_x, moving_fft_y

def process_movingFFT(ID, timerange, interval, windowsize):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、パワースペクトルとその移動平均を導出し、
    その結果(各々ndarray)及び対応するsol(int型)を返す関数

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''
    try:
        #IDに対応するsol及びMUTCを取得
        sol, MUTC = neardevil.get_sol_MUTC(ID)

        #該当sol付近の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("")
        
        #該当範囲の抽出
        near_devildata = neardevil.filter_neardevildata(data, MUTC, timerange, interval)
        if near_devildata is None:
            raise ValueError("")
        
        near_devildata = nearFFT.calculate_residual(near_devildata)
        '''
        「countdown」、「p-pred」、「residual」カラムの追加
        countdown:経過時間(秒) ※countdown ≦ 0
        p-pred:線形回帰の結果(気圧(Pa))
        residual:残差
        '''
        
        #パワースペクトルとその移動平均の導出
        fft_x, fft_y, moving_fft_x, moving_fft_y = moving_FFT(near_devildata, windowsize)
        fft_x, fft_y, moving_fft_x, moving_fft_y = moving_FFT(near_devildata, windowsize)

        return fft_x, fft_y, moving_fft_x, moving_fft_y, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_movingFFT(ID, timerange, interval, windowsize):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、パワースペクトルとその移動平均を計算し、それらを描画した画像を保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度(Pa^2)

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize:パワースペクトルの移動平均を計算する際の窓数(int型)

    '''
    try:
        #パワースペクトルとその移動平均の導出
        fft_x, fft_y, moving_fft_x, moving_fft_y, sol = process_movingFFT(ID, timerange, interval, windowsize)
        
        #音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(fft_x, fft_y, label='FFT')
        plt.plot(moving_fft_x, moving_fft_y,label='FFT_Moving_mean')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'FFT_ID={ID}, sol={sol}, time_range={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel(f'Pressure Power [$Pa^2$]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'nearmovingFFT_{timerange}s_windowsize={windowsize}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_movingFFT.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_movingFFT.png")
        
        return fft_x, fft_y, moving_fft_x, moving_fft_y
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the moving average of the power spectrum corresponding to the ID")
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    #パワースペクトルの移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize', type=int, help="The [windowsize] used to calculate the moving average")
    args = parser.parse_args()
    plot_movingFFT(args.ID, args.timerange, 20, args.windowsize)