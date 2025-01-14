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
import Dispersion_Relation
import nearmovingFFT
import nearmovingratio
import meanFFT_sortedseason
import meanFFT_sorteddP
import meanmovingFFT_sorteddP

def process_movingratiolist_dP(dP_Ulimit, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    dP_Ulimit > dP を満たす全て事象の時系列データを加工し、
    FFTを用いて導出したパワースペクトルとその移動平均の比の移動平均をリスト化したものを返す関数

    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_max < 0
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
    '''
    #記録用配列の作成
    twice_moving_fft_xlist, moving_ratiolist = [], []

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

            #パワースペクトルとその移動平均の導出
            _, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.moving_FFT(near_devildata, windowsize_FFT)
            
            #比の算出
            ratio = fft_y/moving_fft_y
            
            #比の移動平均を算出
            twice_moving_fft_x, moving_ratio = nearmovingratio.calculate_movingave(moving_fft_x, ratio, windowsize_ratio)
            
            #記録用配列に保存
            twice_moving_fft_xlist.append(twice_moving_fft_x) 
            moving_ratiolist.append(moving_ratio)
            
            
        except ValueError as e:
            print(e)
            continue 
            
    return twice_moving_fft_xlist, moving_ratiolist

def plot_meanmovingratio_dP(dP_Ulimit, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    dP_Ulimit > dP を満たす全て事象の時系列データを加工し、
    FFTを用いて導出したパワースペクトルとその移動平均の比の移動平均を平均したもの描画した画像を保存する関数

    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_max < 0
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
    '''

    try:
        twice_moving_fft_xlist, moving_ratiolist = process_movingratiolist_dP(dP_Ulimit, timerange, interval, windowsize_FFT, windowsize_ratio)
        if not twice_moving_fft_xlist or not moving_ratiolist:
            raise ValueError("No data")
        
        # 平均の導出
        twice_moving_fft_x = meanmovingFFT_sorteddP.process_arrays(twice_moving_fft_xlist, np.nanmean)
        moving_ratio =  meanmovingFFT_sorteddP.process_arrays(moving_ratiolist, np.nanmean)
        
        # 音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.plot(twice_moving_fft_x, moving_ratio, label='moving ratio')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'dP <{dP_Ulimit},time_range={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel('Pressure Amplitude Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanmovingratio_dP_{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"meanmovingratio_dP_~{dP_Ulimit}, windowsize_ratio={windowsize_ratio}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: meanmovingratio_dP_~{dP_Ulimit}, windowsize_ratio={windowsize_ratio}.png")
        
        return twice_moving_fft_x, moving_ratio

    except ValueError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the average of the moving average of the ratio of the power spectrum to its moving average for different values of dP_Ulimit")
    parser.add_argument('dP_Ulimit', type=int, help="Maximum value of dP(Pa)(Negative)") #dPの上限の指定(負)
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    #パワースペクトルの移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize_FFT', type=int, help="The [windowsize] used to calculate the moving average of FFT")
    #パワースペクトルとその移動平均の比の移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize_ratio', type=int, help="The [windowsize] used to calculate the moving average of ratio")
    args = parser.parse_args()
    plot_meanmovingratio_dP(args.dP_Ulimit, args.timerange, 20, args.windowsize_FFT, args.windowsize_ratio)
