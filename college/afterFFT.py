import numpy as np
import datetime as datetime
from scipy import signal
import matplotlib.pyplot as plt
import os
import argparse as argparse
import dailychange_p
import neardevil
import afterdevil
from Dispersion_Relation import Params

def calculate_afterresidual(data):
    '''
    フィルタリング済みの時系列データに線形回帰の結果「p-pred」と残差「residual」のカラムを追加する関数
    ※説明変数:経過時間、目的変数:気圧

    data:フィルタリング済みの時系列データ(dataframe)
    '''
    new_data = data.copy()

    #経過時間(秒)の計算及びtimecountカラムの追加
    new_data = afterdevil.calculate_timecount(new_data)

    #p及びtimecountカラムにNanのある行を削除
    new_data = new_data.dropna(subset=['p', 'timecount'])
    
   #説明変数:経過時間(秒) 目的変数:気圧(Pa)で線形回帰
    t, p = new_data['timecount'].values, new_data['p'].values
    a, b = np.polyfit(t, p, 1)
    new_data["p-pred"] = a*t + b

    #残差(Pa)の計算
    new_data["residual"] = p - new_data['p-pred']

    return new_data

def afterFFT(data):
    '''
    フィルタリング及び線形回帰済みの時系列データから
    気圧変化の残差に対してFFTを用いて、パワースペクトルを導出する関数(各々ndarray)

    data:フィルタリング済みの時系列データ(dataframe)
    '''
    #サンプリング周波数の計算
    sampling_freq = 1 / np.mean(np.diff(data['timecount']))
    
    #パワースペクトルの導出
    fft_x, fft_y = signal.periodogram(data['residual'].values, fs=sampling_freq)
    
    return fft_x, fft_y

def process_afterFFT(ID, timerange, interval):
    '''
    IDに対応する「dustdevilの発生直後からtimarange秒後まで」における気圧の時系列データを対象とし、
    気圧変化の線形回帰から導かれる残差に対して、
    パワースペクトルと対応するsolを返す関数

    ID:ダストデビルに割り振られた通し番号(int型)
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
        after_devildata = afterdevil.filter_afterdevildata(data, MUTC, timerange, interval)
        if after_devildata is None:
            raise ValueError("")
        
        after_devildata = calculate_afterresidual(after_devildata)
        '''
        「timecount」、「p-pred」、「residual」カラムの追加
        timecount:経過時間(秒) ※timecount ≦ 0
        p-pred:線形回帰の結果(気圧(Pa))
        residual:残差
        '''
        
        #パワースペクトルの導出
        fft_x, fft_y = afterFFT(after_devildata)

        return fft_x, fft_y, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def plot_afterFFT(ID, timerange, interval):
    '''
    IDに対応する「dustdevilの発生直後からtimarange秒後まで」における気圧の時系列データを対象とし、
    気圧変化の線形回帰から導かれる残差に対して、
    パワースペクトルを描画した画像を保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度(Pa^2)

    ID:ダストデビルに割り振られた通し番号(int型)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''
    try:
        #パワースペクトルの導出
        fft_x, fft_y, sol = process_afterFFT(ID, timerange, interval)
        
        #音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-8,1e2)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'PS_ID={ID}, sol={sol}, time_range={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel(f'Pressure Power [$Pa^2&]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'afterFFT_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed:sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}.png")
        
        return fft_x, fft_y

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_afterFFT(args.ID, args.timerange, 20)