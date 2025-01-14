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
import meanFFT_sortedseason
import meanFFT_sorteddP
import Dispersion_Relation
import nearmovingFFT

def process_arrays(arrays, operation):
    '''
    空の行を除外した後、各行の長さ要素数が同じであることを確認し、
    列(axis=0)に対して指定された操作する関数。

    arrays:多次元データ(各行の長さが一致している必要あり)
    operation:施す操作 (例) np.sum, np.max…など
    '''
    # 空配列を除外
    if not arrays or any(len(arr) == 0 for arr in arrays):
        return []

    # 要素数の一致を確認
    length = len(arrays[0])
    if not all(len(arr) == length for arr in arrays):
        raise ValueError("All arrays must have the same length")
    
    #配列に変換し、指定された操作を施す
    result = operation(np.array(arrays), axis=0)

    return result

def process_movingFFTlist_dP(dP_Ulimit, timerange, interval, windowsize):
    '''
    dP_Ulimit > dP を満たす全て事象の時系列データを加工し、
    FFTを用いて導出したパワースペクトルとその移動平均をリスト化したものを返す関数

    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_Ulimit < 0
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    window_size:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''
    #記録用配列の作成
    fft_xlist, fft_ylist = [], []
    moving_fft_xlist, moving_fft_ylist = [], []

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
            
            #記録用配列に保存
            fft_xlist.append(fft_x)
            fft_ylist.append(fft_y)
            moving_fft_xlist.append(moving_fft_x)
            moving_fft_ylist.append(moving_fft_y)
            
        except ValueError as e:
            print(e)
            continue 
            
    return fft_xlist, fft_ylist, moving_fft_xlist, moving_fft_ylist

def plot_meanmovingFFT_dP(dP_Ulimit, timerange, interval, windowsize):
    '''
    dP_max > dP かつ Ws-ave<Ws_min を満たす全て事象の時系列データを加工し、
    FFTを用いて導出したパワースペクトルとその移動平均のケース平均を算出し、それの描画した画像を保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度(Pa^2)

    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_max < 0
    time_range:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''
    try:
        #対応する全事象のパワースペクトルとその移動平均をリスト化したものの導出
        fft_xlist, fft_ylist, moving_fft_xlist, moving_fft_ylist = process_movingFFTlist_dP(dP_Ulimit, timerange, interval, windowsize)
        if not fft_xlist or not fft_ylist or not moving_fft_xlist or not moving_fft_ylist:
            raise ValueError("No data")
        
        # パワースペクトルとその移動平均のケース平均の導出
        fft_x = process_arrays(fft_xlist, np.nanmean)
        fft_y = process_arrays(fft_ylist, np.nanmean) 
        moving_fft_x = process_arrays(moving_fft_xlist, np.nanmean)
        moving_fft_y = process_arrays(moving_fft_ylist, np.nanmean)
        
        # 音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-4,1e1)
        plt.plot(fft_x, fft_y, label='FFT')    
        plt.plot(moving_fft_x, moving_fft_y, label='FFT_Movingmean')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'dP <{dP_Ulimit},time_range={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel(f'Pressure Power [$Pa^2$]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanmovingFFT_dP_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"meanmovingFFT_dP_~{dP_Ulimit}_windowsize={windowsize}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: meanmovingFFT_dP_~{dP_Ulimit}_windowsize={windowsize}.png")
        
        return moving_fft_x, moving_fft_y

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the average moving power spectrum for each dP_Ulimit")
    parser.add_argument('dP_Ulimit', type=int, help="Maximum value of dP(Pa)(Negative)") #dPの上限の指定(負)
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    #パワースペクトルの移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize', type=int, help="The [windowsize] used to calculate the moving average")
    args = parser.parse_args()
    plot_meanmovingFFT_dP(args.dP_Ulimit, args.timerange, 20, args.windowsize)