import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse as argparse
import focuschange_p
import nodevil
from tqdm import tqdm
import nearFFT
import nearmovingFFT
import Dispersion_Relation

def process_focusratio(sol, MUTC_h, timerange, windowsize_FFT):
    '''
    sol,時刻MUTC_hからtimerange秒間に対応する気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、
    FFTを用いてパワースペクトルとその移動平均の比を導出(各々ndarray)する関数

    sol:取り扱う火星日(探査機到着後からの経過日数)(int型)
    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    windosize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''
    try:
        #該当する時系列データの取得
        focus_data = focuschange_p.process_focusdata_p(sol, MUTC_h, timerange)
        if focus_data is None:
            raise ValueError("")
        
        focus_data = nearFFT.calculate_residual(focus_data)
        '''
        「countdown」、「p-pred」、「residual」カラムの追加
        countdown:経過時間(秒) ※countdown ≦ 0
        p-pred:線形回帰の結果(気圧(Pa))
        residual:残差
        '''
        
        #パワースペクトルとその移動平均の導出
        _, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.moving_FFT(focus_data, windowsize_FFT)
        
        #比の算出
        ratio = fft_y/moving_fft_y

        return moving_fft_x, ratio
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_focusratio(sol, MUTC_h, timerange, windowsize_FFT):
    '''
    sol,時刻MUTC_hからtimerange秒間に対応する気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、
    FFTを用いてパワースペクトルとその移動平均の比を導出(各々ndarray)を描画した画像を保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度の比

    sol:取り扱う火星日(探査機到着後からの経過日数)(int型)
    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    windosize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''
    try:
        #パワースペクトルとその移動平均の導出
        moving_fft_x, ratio = process_focusratio(sol, MUTC_h, timerange, windowsize_FFT)
        
        #音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.plot(moving_fft_x, ratio,label='ratio', fontsize=15)
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'ratio_sol={sol},({MUTC_h}~_{timerange}s)', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Power Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'focusmovingratio({MUTC_h}~_{timerange}s)_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"ratio_sol={sol},({MUTC_h}~_{timerange}s)_focusmovingratio.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},({MUTC_h}~_{timerange}s)_focusmovingratio.png")
        
        return moving_fft_x, ratio
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot focus pressure changes corresponding to the sol, LTST_h and timerange")
    parser.add_argument('MUTC_h', type=int, help='Base start time') #基準となる開始の時間
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    parser.add_argument('windowsize_FFT', type=int, help="The [windowsize] used to calculate the moving average of FFT")
    #パワースペクトルとその移動平均の比の移動平均を計算する際の窓数の指定
    args = parser.parse_args()
    
    #ダストデビルのないsolを描画
    nodevilsollist = nodevil.process_nodevilsollist()
    for sol in tqdm(nodevilsollist, desc="Processing nodevil sols"):
        plot_focusratio(sol, args.LTST_h, args.timerange, args.windowsize_FFT) 