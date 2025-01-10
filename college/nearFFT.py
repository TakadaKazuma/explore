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

def calculate_residual(data):
    '''
    フィルタリング済みの時系列データに線形回帰の結果「p-pred」と残差「residual」のカラムを追加する関数
    ※説明変数:経過時間、目的変数:気圧

    data:フィルタリング済みの時系列データ(dataframe)
    '''
    new_data = data.copy()

    #経過時間(秒)の計算及びcountdownカラムの追加
    new_data = neardevil.calculate_countdown(new_data)

    #p及びcountdownカラムにNanのある行を削除
    new_data = new_data.dropna(subset=['p', 'countdown'])
    
   #説明変数:経過時間(秒) 目的変数:気圧(Pa)で線形回帰
    t, p = new_data['countdown'].values, new_data['p'].values
    a, b = np.polyfit(t, p, 1)
    new_data["p-pred"] = a*t + b

    #残差(Pa)の計算
    new_data["residual"] = p - new_data['p-pred']

    return new_data

def FFT(data):
    '''
    フィルタリング及び線形回帰済みの時系列データから
    気圧変化の残差に対してFFTを用いて、パワースペクトルを導出する関数(各々ndarray)

    data:フィルタリング済みの時系列データ(dataframe)
    '''
    #サンプリング周波数の計算
    sampling_freq = 1 / np.mean(np.diff(data['countdown']))
    
    #パワースペクトルの導出
    fft_x, fft_y = signal.periodogram(data['residual'].values, fs=sampling_freq)
    
    return fft_x, fft_y

def process_nearFFT(ID, timerange, interval):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、
    FFTを用いてパワースペクトルを導出(各々ndarray)及び対応するsol(int型)を返す関数

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
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
        
        near_devildata = calculate_residual(near_devildata)
        '''
        「countdown」、「p-pred」、「residual」カラムの追加
        countdown:経過時間(秒) ※countdown ≦ 0
        p-pred:線形回帰の結果(気圧(Pa))
        residual:残差
        '''
        
        #パワースペクトルの導出
        fft_x, fft_y = FFT(near_devildata)

        return fft_x, fft_y, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def plot_nearFFT(ID, timerange, interval):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、FFTを用いてパワースペクトルを導出し、それを描画した画像を保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度(Pa^2)

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''
    try:
        #パワースペクトルの導出
        fft_x, fft_y, sol = process_nearFFT(ID, timerange, interval)
        
        #音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-8,1e8)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'FFT_ID={ID}, sol={sol}, time_range={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel(f'Pressure Power [$Pa^2&]')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        
        #保存の設定
        output_dir = f'nearFFT_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_nearFFT.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_nearFFT.png")
        
        return fft_x, fft_y
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Power Spectrum corresponding to the ID")
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_nearFFT(args.ID, args.timerange, 20)