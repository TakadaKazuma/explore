import matplotlib.pyplot as plt
import os
import argparse as argparse
import focuschange_p
import nodevil
from tqdm import tqdm
import nearFFT
import nearmovingFFT
import nearratio
import nearmovingratio
from Dispersion_Relation import Params

def process_focusmovingratio_resample(sol, MUTC_h, timerange, windowsize_FFT, windowsize_ratio):
    '''
    指定された sol (火星日) における MUTC_h (地方時) 直後の
    時系列データ(0.5秒間隔でリサンプリング済)における気圧残差を求め、
    それに対するパワースペクトル比の移動平均(修正パワースペクトル)を算出する関数。

    sol : 火星日(探査機到着後からの経過日数)(int)
    MUTC_h : 火星地方時(0 ≦ MUTC_h ≦ 23)(int)
    timerange : 切り取る時間範囲 (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均を計算する際に用いる窓数(int)
    windowsize_ratio : パワースペクトル比の移動平均に用いる窓数(int)
    '''
    try:
        # 該当 sol 及び MUTC_h に対応する時系列データの取得
        focus_data = focuschange_p.process_focusdata_p(sol, MUTC_h, timerange)
        if focus_data is None:
            raise ValueError("Failed to retrieve time-series data.")

        # 0.5秒間隔でresample
        focus_devildata = meanFFT_sortedseason.data_resample(focus_devildata, 0.5)
        
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
        
        # パワースペクトル比の算出
        moving_fft_x, ratio = nearratio.calculate_ratio(fft_x, fft_y, moving_fft_y, windowsize_FFT)

        # 修正パワースペクトルを算出
        moving_fft_x, moving_ratio = nearmovingratio.calculate_movingave(moving_fft_x, ratio, windowsize_ratio)
        

        return moving_fft_x, moving_ratio
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_focusmovingratio_resample(sol, MUTC_h, timerange, windowsize_FFT, windowsize_ratio):
    '''
    指定された sol (火星日) における MUTC_h (地方時) 直後の
    時系列データ(0.5秒間隔でリサンプリング済)における気圧残差を求め、
    それに対する修正パワースペクトルを算出し、プロットを保存する関数。
    
    - X軸 : 振動数 (Hz)
    - Y軸 : 修正パワースペクトル (/)

    sol : 火星日(探査機到着後からの経過日数)(int)
    MUTC_h : 火星地方時(0 ≦ MUTC_h ≦ 23)(int)
    timerange : 切り取る時間範囲 (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均を計算する際に用いる窓数(int)
    '''
    try:
        # 修正パワースペクトルの算出
        moving_fft_x, moving_ratio = process_focusmovingratio_resample(sol, MUTC_h, timerange, windowsize_FFT, windowsize_ratio)

        # 特定の周波数より高周波の情報をNaNに変更
        #moving_fft_x, ratio = nearratio.filter_xUlimit(moving_fft_x, ratio, 0.8)
        
        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        # 描画の設定
        plt.xscale('log')
        plt.ylim(1e-8, 1e2)
        plt.plot(moving_fft_x, moving_ratio,label='ratio')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'PSR_sol={sol}, MUTC={MUTC_h}:00~{timerange}s')
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Power Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'focusmovingratioresample,MUTC={MUTC_h}:00~_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"sol={sol}_{timerange}s_windowsize_ratio={windowsize_ratio}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return moving_fft_x, moving_ratio
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('MUTC_h', type=int, help='Base start time') # MUTC_h (火星地方時)の指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average of FFT") # パワースペクトルの移動平均に用いる窓数
    parser.add_argument('windowsize_ratio', type=int, 
                        help="The [windowsize] used to calculate the moving average of ratio") # パワースペクトル比のの移動平均に用いる窓数
    args = parser.parse_args()
    
    #ダストデビルの発生がない時間帯の修正パワースペクトルを描画
    nodevilsollist = nodevil.process_nodevilsollist()
    for sol in tqdm(nodevilsollist, desc="Processing nodevil sols"):
        plot_focusmovingratio_resample(sol, args.MUTC_h, args.timerange, args.windowsize_FFT, args.windowsize_ratio) 