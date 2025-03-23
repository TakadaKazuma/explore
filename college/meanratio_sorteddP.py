import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearFFT
import nearmovingFFT
import nearratio
import meanFFT_sortedseason
import meanFFT_sorteddP
import meanmovingFFT_sorteddP
from Dispersion_Relation import Params

def process_ratiolist_dP(dP_Ulimit, timerange, interval, windowsize_FFT):
    '''
    dP_Ulimit > dP を満たす全て事象の時系列データを加工し、
    全ての「パワースペクトルとその移動平均の比(パワースペクトル比)」
    を列挙したリストを返す関数

    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_max < 0
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトル比を計算する際の窓数(int型)
    windowsize_ratio:修正パワースペクトルを計算する際の窓数窓数(int型)
    '''
    #記録用配列の作成
    moving_fft_xlist, ratiolist = [], []

    #dP_Ulimit > dP を満たすIDリストの作成
    IDlist = meanFFT_sorteddP.process_IDlist_dP(dP_Ulimit)
    for ID in tqdm(IDlist, desc="Processing IDs"):
        try:
            #IDに対応するsol及びMUTCを取得
            sol, MUTC = neardevil.get_sol_MUTC(ID)

            #該当sol付近の時系列データを取得
            data = dailychange_p.process_surround_dailydata(sol)
            if data is None:
                raise ValueError("")
            
            #該当範囲の抽出
            near_devildata = neardevil.filter_neardevildata(data, MUTC, timerange, interval)

            #加工済みデータを0.5秒でresample
            near_devildata = meanFFT_sortedseason.data_resample(near_devildata, 0.5)
            if near_devildata is None:
                raise ValueError("No data")
             
            near_devildata = nearFFT.calculate_residual(near_devildata)
            '''
            「countdown」、「p-pred」、「residual」カラムの追加
            countdown:経過時間(秒) ※countdown ≦ 0
            p-pred:線形回帰の結果(気圧(Pa))
            residual:残差 (Pa)
            '''

            #パワースペクトルとその移動平均の導出
            fft_x, fft_y, _, moving_fft_y = nearmovingFFT.moving_FFT(near_devildata, windowsize_FFT)
            
            
            #比の算出
            moving_fft_x, ratio = nearratio.calculate_ratio(fft_x, fft_y, moving_fft_y, windowsize_FFT)
            
            
            #記録用配列に保存
            moving_fft_xlist.append(moving_fft_x) 
            ratiolist.append(ratio)
            
            
        except ValueError as e:
            print(e)
            continue 
            
    return moving_fft_xlist, ratiolist

def plot_meanratio_dP(dP_Ulimit, timerange, interval, windowsize_FFT):
    '''
    dP_Ulimit > dP  を満たす全て事象の時系列データを加工し、
    全ての「パワースペクトルとその移動平均の比(パワースペクトル比)」の平均を描画及び保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度の比
    
    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_max < 0
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
    '''

    try:
        #対応する全事象のパワースペクトルとその移動平均の比を（修正)リスト化したものを導出
        moving_fft_xlist, ratiolist = process_ratiolist_dP(dP_Ulimit, timerange, interval, windowsize_FFT)
        if not moving_fft_xlist or not ratiolist:
            raise ValueError("No data")
        
        #修正パワースペクトルのケース平均の導出
        moving_fft_x = meanmovingFFT_sorteddP.process_arrays(moving_fft_xlist, np.nanmean)
        ratio =  meanmovingFFT_sorteddP.process_arrays(ratiolist, np.nanmean)
        
        #特定の周波数より高周波の情報をnanに変更
        #moving_fft_x, moving_ratio = nearratio.filter_xUlimit(moving_fft_x, moving_ratio, 0.8)
        
        #音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        #プロットの設定
        plt.xscale('log')
        plt.plot(moving_fft_x, ratio, label='moving ratio')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MPSR_dP >{-dP_Ulimit},time_range={timerange}s', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Amplitude Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'meanratio_dP_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"dP is More{-dP_Ulimit},windowsize_FFT={windowsize_FFT}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed:dP is More{-dP_Ulimit},windowsize_FFT={windowsize_FFT}.png")
        
        return moving_fft_x, ratio

    except ValueError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the average of the moving average of the ratio of the power spectrum to its moving average for different values of dP_Ulimit")
    parser.add_argument('dP_Ulimit', type=int, help="Maximum value of dP(Pa)(Negative)") #dPの上限の指定(負)
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average of FFT") #パワースペクトルの移動平均を計算する際の窓数の指定
    args = parser.parse_args()
    plot_meanratio_dP(args.dP_Ulimit, args.timerange, 20, args.windowsize_FFT)