import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearratio
import nearmovingratio
import afterdevil
import aftermovingFFT
import returnafterFFT
import meanFFT_sortedseason
from Dispersion_Relation import Params

def process_returnaftermovingratio_resample(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直後の
    時系列データ(0.5秒間隔でリサンプリング済)における気圧残差を求め、
    それに対するパワースペクトル比の移動平均(修正パワースペクトル)を算出する関数。

    ID : ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均に用いる窓数(int)
    windowsize_ratio : パワースペクトル比の移動平均に用いる窓数(int)
    '''
    try:
        # ID に対応する sol および MUTC を取得
        sol, MUTC = neardevil.get_sol_MUTC(ID)

        # 該当 sol 周辺の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("No data available after filtering.")
        
        # MUTC 付近のデータを抽出
        after_devildata = afterdevil.filter_afterdevildata(data, MUTC, timerange, interval)
        if after_devildata is None:
            raise ValueError("No data available after filtering.")

        #0.5秒間隔でresample
        after_devildata = meanFFT_sortedseason.data_resample(after_devildata, 0.5)

        # 残差計算の実施
        after_devildata = returnafterFFT.calculate_returnafterresidual(after_devildata)
        '''
        追加カラム:
        - timecount: 経過時間 (秒) (timecount ≧ 0)
        - p-pred: 線形回帰による気圧予測値 (Pa)
        - residual: 気圧の残差
        '''

        # FFT によるパワースペクトルとその移動平均の導出
        fft_x, fft_y, _, moving_fft_y = aftermovingFFT.after_movingFFT(after_devildata, windowsize_FFT)
        
        # パワースペクトル比の算出
        moving_fft_x, ratio = nearratio.calculate_ratio(fft_x, fft_y, moving_fft_y, windowsize_FFT)

        # 修正パワースペクトルを算出
        moving_fft_x, moving_ratio = nearmovingratio.calculate_movingave(moving_fft_x, ratio, windowsize_ratio)

        return moving_fft_x, moving_ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_returnaftermovingratio_resample(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直後の
    時系列データ(0.5秒間隔でリサンプリング済)における気圧残差を求め、
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
        moving_fft_x, moving_ratio, sol = process_returnaftermovingratio_resample(ID, timerange, interval, windowsize_FFT, windowsize_ratio)

        # 特定の周波数より高周波の情報をnanに変更
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
        output_dir = f'returnaftermovingratioresample_{timerange}s_windowsize_FFT={windowsize_FFT}'
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
    plot_returnaftermovingratio_resample(args.ID, args.timerange, 20, args.windowsize_FFT, args.windowsize_ratio)