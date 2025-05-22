import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse as argparse
import focuschange_p
import nodevil
from tqdm import tqdm
import nearFFT
import nearmovingFFT
from Dispersion_Relation import Params

def process_focusmovingFFT(sol, MUTC_h, timerange, windowsize_FFT):
    '''
    指定された sol (火星日) における MUTC_h (地方時) 直後の
    時系列データにおける気圧残差を求め、
    それに対するパワースペクトルとその移動平均を算出する関数。

    sol : 火星日(探査機到着後からの経過日数)(int)
    MUTC_h : 火星地方時(0 ≦ MUTC_h ≦ 23)(int)
    timerange : 切り取る時間範囲 (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均を計算する際に用いる窓数(int)
    '''
    try:
        # 該当 sol 及び MUTC_h に対応する時系列データの取得
        focus_data = focuschange_p.process_focusdata_p(sol, MUTC_h, timerange)
        if focus_data is None:
            raise ValueError("Failed to retrieve time-series data.")
        
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

        return fft_x, fft_y, moving_fft_x, moving_fft_y
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def plot_focusmovingFFT(sol, MUTC_h, timerange, windowsize_FFT):
    '''
    指定された sol (火星日) における MUTC_h (地方時) 直後の
    時系列データにおける気圧残差を求め、
    それに対するパワースペクトルとその移動平均算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : スペクトル強度 (Pa^2)

    sol : 火星日(探査機到着後からの経過日数)(int)
    MUTC_h : 火星地方時(0 ≦ MUTC_h ≦ 23)(int)
    timerange : 切り取る時間範囲 (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均を計算する際に用いる窓数(int)
    '''  
    try:
        # パワースペクトルとその移動平均の導出
        fft_x, fft_y, moving_fft_x, moving_fft_y = process_focusmovingFFT(sol, MUTC_h, timerange, windowsize_FFT)
        
        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        # 描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-11, 1e2)
        plt.plot(fft_x, fft_y, label='FFT')
        #plt.ylim(1e-6, 1e-1)
        plt.plot(moving_fft_x, moving_fft_y,label='FFT_Moving_mean')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'PS_sol={sol}, MUTC={MUTC_h}:00~{timerange}s', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel(f'Pressure Power [$Pa^2$]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'focusmovingFFT,MUTC={MUTC_h}:00~_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"sol={sol}_{timerange}s.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return fft_x, fft_y, moving_fft_x, moving_fft_y
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('MUTC_h', type=int, help='Base start time') # MUTC_h (火星地方時)の指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average of FFT") # パワースペクトルの移動平均に用いる窓数
    args = parser.parse_args()
    
    # ダストデビルのない sol のパワースペクトルを描画
    nodevilsollist = nodevil.process_nodevilsollist()
    for sol in tqdm(nodevilsollist, desc="Processing nodevil sols"):
        plot_focusmovingFFT(sol, args.MUTC_h, args.timerange, args.windowsize_FFT) 
