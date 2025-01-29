import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse as argparse
import dailychange_p
import focuschange_p
import nodevil
from tqdm import tqdm
import nearFFT
import Dispersion_Relation
import meanFFT_sortedseason

def process_focusFFT(sol, MUTC_h, timerange):
    '''
    sol,時刻MUTC_hからtimerange秒間に対応する気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、
    FFTを用いてパワースペクトルを導出(各々ndarray)及び対応するsol(int型)を返す関数

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
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
        
        #パワースペクトルの導出
        fft_x, fft_y = nearFFT.FFT(focus_data)

        return fft_x, fft_y
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_focusFFT(sol, LTST_h, timerange):
    '''
    sol,時刻MUTC_hからtimerange秒間に対応する気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、
    FFTを用いてパワースペクトルを導出(各々ndarray)及び対応するsol(int型)を導出し、
    それを描画した画像を保存する関数。
    横軸:周波数(Hz) 縦軸:スペクトル強度(Pa^2)

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''  
    try:
        #パワースペクトルの導出
        fft_x, fft_y = process_focusFFT(sol, LTST_h, timerange)
        
        #音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-8,1e2)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'FFT_sol={sol},({LTST_h}~_{timerange}s) ')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel(f'Pressure Power [$Pa^2&]')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        
        #保存の設定
        output_dir = f'focusFFT({LTST_h}~_{timerange}s)'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"FFT_sol={sol},({LTST_h}~_{timerange}s)_focusFFT.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},({LTST_h}~_{timerange}s)_focusFFT.png")
        
        return fft_x, fft_y
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

