import numpy as np
import matplotlib.pyplot as plt
import os
import argparse as argparse
import dailychange_p
import focuschange_p
import nodevil
from tqdm import tqdm
import nearFFT
import nearmovingFFT
import nearratio
import meanFFT_sortedseason
import meanmovingFFT_sorteddP
from Dispersion_Relation import Params

def process_focusratiolist(MUTC_h, timerange, windowsize_FFT):
    '''
    ダストデビルの発生がない sol (火星日)におけるMUTC_h (地方時) 直後の
    時系列データにおける気圧残差を求め、
    各ケースのパワースペクトルとその移動平均の比(パワースペクトル比)をまとめたリストを返す関数。

    MUTC_h : 火星地方時(0 ≦ MUTC_h ≦ 23)(int)
    timerange : 切り取る時間範囲 (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均に用いる窓数(int)
    '''
    # 記録用配列の作成 
    moving_fft_xlist, ratio_list = [], []

    # ダストデビルの発生がないsolのリストを作成
    nodevilsollist = nodevil.process_nodevilsollist()

    for sol in tqdm(nodevilsollist, desc="Processing nodevil sol"):
        # 該当 sol に対応する時系列データの取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None or data.empty:
            continue
        
        # 該当 sol 及び MUTC_h に対応する時系列データの取得
        focus_data = focuschange_p.filter_focusdata(data, sol, MUTC_h, timerange)
        if focus_data is None or focus_data.empty:
            continue

        # 0.5秒間隔でresample     
        focus_data = meanFFT_sortedseason.data_resample(focus_data, 0.5)

        # 残差計算を実施
        focus_data = nearFFT.calculate_residual(focus_data)
        '''
        追加カラム:
        - countdown: 経過時間 (秒) (countdown ≦ 0)
        - p-pred: 線形回帰による気圧予測値 (Pa)
        - residual: 気圧の残差
        '''
            
        # FFT によるパワースペクトルとその移動平均の導出
        fft_x, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.movingFFT(focus_data, windowsize_FFT)

        # パワースペクトル比の導出
        moving_fft_x, ratio = nearratio.calculate_ratio(fft_x, fft_y, moving_fft_y, windowsize_FFT)
            
        # 結果を配列に記録
        moving_fft_xlist.append(moving_fft_x)
        ratio_list.append(ratio)
            
    return moving_fft_xlist, ratio_list

def plot_meanfocusratio(MUTC_h, timerange, windowsize_FFT):
    '''
    ダストデビルの発生がない sol (火星日)におけるMUTC_h (地方時) 直後の
    時系列データにおける気圧残差を求め、
    各ケースのパワースペクトル比の平均を算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : パワースペクトル比 (/)

    MUTC_h : 火星地方時(0 ≦ MUTC_h ≦ 23)(int)
    timerange : 切り取る時間範囲 (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均に用いる窓数(int)
    '''
    try:
        # 各ケースにおけるパワースペクトル比をまとめたリストを導出
        moving_fft_xlist, ratio_list = process_focusratiolist(MUTC_h, timerange, windowsize_FFT)
        
        # パワースペクトル比のケース平均の導出
        moving_fft_x =  meanmovingFFT_sorteddP.process_arrays(moving_fft_xlist, np.nanmean)
        ratio = meanmovingFFT_sorteddP.process_arrays(ratio_list, np.nanmean)

        # 特定の周波数より高周波の情報をnanに変更
        #moving_fft_x, moving_ratio = nearratio.filter_xUlimit(moving_fft_x, moving_ratio, 0.8)
        
        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.plot(moving_fft_x, ratio, label='ratio')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MPSR_MUTC={MUTC_h}:00~{timerange}s')
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel(f'Pressure Amplitude Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanfocusratio,MUTC={MUTC_h}:00~'
        filename = f"{timerange}s_windowsize_FFT={windowsize_FFT}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return moving_fft_x, ratio

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('MUTC_h', type=int, help='Base start time') # MUTC_h (火星地方時)の指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average of FFT") # パワースペクトルの移動平均に用いる窓数
    args = parser.parse_args()
    plot_meanfocusratio(args.MUTC_h, args.timerange, args.windowsize_FFT)
