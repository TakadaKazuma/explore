import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearFFT
import nearmovingFFT
import nearratio
from Dispersion_Relation import Params

def calculate_movingave(x, y, windowsize_ratio):
    '''
    元データ y の移動平均を算出し、
    描画用に x と長さをそろえて返す関数

    ※パワースペクトル各点は物理的意味とデータ長を保持するため、
    この関数では移動平均が計算できない領域にはNaNを付与して形状を維持している。

    x : x軸データ(array)
    y : 移動平均の対象となる元のyデータ(array)
    windowsize_ratio : yの移動平均に用いる窓数
    '''

    # 移動平均用のフィルターを作成
    filter_frame = np.ones(windowsize_ratio) / windowsize_ratio

    # 欠損が発生する要素数の計算
    pad_size = (windowsize_ratio - 1) // 2

    # 形状を維持するため、配列を全てNanに変更(初期化)
    moving_x = np.full(x.shape, np.nan)
    moving_y = np.full(y.shape, np.nan)

    # 有効な範囲の抽出及び計算
    moving_x[pad_size:-pad_size] = x[pad_size:-pad_size]
    moving_y[pad_size:-pad_size] = np.convolve(y, filter_frame, mode='valid')
    
    return moving_x, moving_y


def process_movingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データにおける気圧残差を求め、
    それに対するパワースペクトル比の移動平均(修正パワースペクトル)を算出する関数。

    ID : ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均に用いる窓数(int)
    windowsize_ratio : パワースペクトル比の移動平均に用いる窓数(int)
    '''
    try:
        # IDに対応するsol及びMUTCを取得
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
        
        # FFTによるパワースペクトルとその移動平均の導出
        fft_x, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.movingFFT(near_devildata, windowsize_FFT)

        # パワースペクトル比の算出
        moving_fft_x, ratio = nearratio.calculate_ratio(fft_x, fft_y, moving_fft_y, windowsize_FFT)
        
        # 修正パワースペクトルを算出
        moving_fft_x, moving_ratio = calculate_movingave(moving_fft_x, ratio, windowsize_ratio)

        return moving_fft_x, moving_ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_movingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データにおける気圧残差を求め、
    それに対する修正パワースペクトルを算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : 修正パワースペクトル (/)

    ID: ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均に用いる窓数(int)
    windowsize_ratio : パワースペクトル比の移動平均に用いる窓数(int)
    '''
    try:
        # 修正パワースペクトルの算出
        moving_fft_x, moving_ratio, sol = process_movingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio)

        # 特定の周波数より高周波の情報をNaNに変更
        #moving_fft_x, ratio = nearratio.filter_xUlimit(moving_fft_x, ratio, 0.8)

        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        # 描画の設定
        plt.xscale('log')
        plt.plot(moving_fft_x, moving_ratio, label='Moving_ratio')        
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MPS_ID={ID}, sol={sol}, timerange={timerange}(s)', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Amplitude Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'nearmovingratio_{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"ID={str(ID).zfill(5)},sol={str(sol).zfill(4)},windowsize_ratio={windowsize_ratio}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return moving_fft_x, moving_ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") # IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average of FFT") # パワースペクトルの移動平均に用いる窓数
    parser.add_argument('windowsize_ratio', type=int, 
                        help="The [windowsize] used to calculate the moving average of ratio") # パワースペクトル比のの移動平均に用いる窓数
    args = parser.parse_args()
    plot_movingratio(args.ID, args.timerange, 20, args.windowsize_FFT, args.windowsize_ratio)