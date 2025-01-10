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
import nearFFT
import Dispersion_Relation
import nearmovingFFT

def calculate_movingave(x, y, windowsize):
    '''
    x及びyの移動平均を算出する関数
    ※用途は主にmovingratio移動平均を算出

    x:移動平均を算出したいndarray
    y:xと対をなす移動平均を算出したいndarray
    windowsize:移動平均を計算する際の窓数(int型)
    '''

    '''
    以下では形状を維持しつつ、x及びyの移動平均を導出
    ※主にx,yはパワースペクトルをその移動平均で割ったものが対象となる。
    パワースペクトルの各要素及び長さは、それ自体に意味があるため、
    パワースペクトルから算出される計算の対象も同様である。
    移動平均を算出した際にその情報が壊れないようにするため、
    正確な移動平均の値が計算できない要素にnanを代入し、
    形状を維持するようにしている。
    '''
    filter_frame = np.ones(windowsize) / windowsize
    pad_size = (windowsize - 1) // 2
    moving_x = np.ones(x.shape)*np.nan
    moving_y = np.ones(y.shape)*np.nan
    
    moving_x[pad_size:-pad_size] = np.convolve(x, filter_frame, mode="valid")
    moving_y[pad_size:-pad_size] = np.convolve(y, filter_frame, mode='valid')
    
    return moving_x, moving_y

def process_twicemovingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、パワースペクトルとその移動平均を導出し、
    それらの比に対して再度移動平均を算出したもの(ndarray型)及び対応するsol(int型)を返す関数

    ID:ダストデビルに割り振られた通し番号
    time_range:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
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
        fft_x, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.moving_FFT(near_devildata, windowsize_FFT)

        #比の算出
        ratio = fft_y/moving_fft_y

        #比の移動平均を算出
        twice_moving_fft_x, moving_ratio = calculate_movingave(moving_fft_x, ratio, windowsize_ratio)

        return twice_moving_fft_x, moving_ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_twicemovingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、パワースペクトルとその移動平均を導出し、
    それらの比に対して再度移動平均を算出したものを描画した画像を保存する関数

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
    '''
    try:
        #パワースペクトルとその移動平均の比を導出し、更にその移動平均を算出する
        twice_moving_fft_x, moving_ratio, sol = process_twicemovingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio)

        #音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.plot(twice_moving_fft_x, moving_ratio, label='moving_ratio')        
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'FFT_ID={ID}, sol={sol}, time_range={timerange}(s)')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel('Pressure Amplitude Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'neartwicemovingratio_{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)},windowsize={windowsize_ratio},twicemovingratio.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},ID={str(ID).zfill(5)},windowsize={windowsize_ratio},twicemovingratio.png")
        
        return twice_moving_fft_x, moving_ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the moving average of the ratio of the power spectrum to its moving average for the resampled data corresponding to the given ID.")
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    #パワースペクトルの移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize_FFT', type=int, help="The [windowsize] used to calculate the moving average of FFT")
    #パワースペクトルとその移動平均の比の移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize_ratio', type=int, help="The [windowsize] used to calculate the moving average of ratio")
    args = parser.parse_args()
    plot_twicemovingratio(args.ID, args.timerange, 20, args.windowsize_FFT, args.windowsize_ratio)
