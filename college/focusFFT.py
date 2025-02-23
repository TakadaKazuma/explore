import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse as argparse
import focuschange_p
import nodevil
from tqdm import tqdm
import nearFFT
import Dispersion_Relation

def process_focusFFT(sol, MUTC_h, timerange):
    '''
    sol,MUTC_h(時刻)からtimerange秒間に対応する気圧の時系列データを対象とし、
    気圧変化の線形回帰から導かれる残差に対して、
    パワースペクトルを返す関数。

    sol:取り扱う火星日(探査機到着後からの経過日数)(int型)
    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
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

def plot_focusFFT(sol, MUTC_h, timerange):
    '''
    sol,MUTC_h(時刻)からtimerange秒間に対応する気圧の時系列データを対象とし、
    気圧変化の線形回帰から導かれる残差に対して、
    パワースペクトルを描画した画像を保存する関数。
    横軸:周波数(Hz) 縦軸:スペクトル強度(Pa^2)

    sol:取り扱う火星日(探査機到着後からの経過日数)(int型)
    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    '''  
    try:
        #パワースペクトルの導出
        fft_x, fft_y = process_focusFFT(sol, MUTC_h, timerange)
        
        #音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-8,1e2)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'PS_sol={sol}, MUTC={MUTC_h}~{timerange}s', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel(f'Pressure Power [$Pa^2$]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'focusFFT,MUTC={MUTC_h}~{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={sol},MUTC={MUTC_h}~{timerange}s,focusFFT.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},MUTC={MUTC_h}~{timerange}s,focusFFT.png")
        
        return fft_x, fft_y
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot focus pressure changes corresponding to the sol, LTST_h and timerange")
    parser.add_argument('MUTC_h', type=int, help='Base start time') #基準となる開始の時間
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    
    #ダストデビルのないsolを描画
    nodevilsollist = nodevil.process_nodevilsollist()
    for sol in tqdm(nodevilsollist, desc="Processing nodevil sols"):
        plot_focusFFT(sol, args.MUTC_h, args.timerange)