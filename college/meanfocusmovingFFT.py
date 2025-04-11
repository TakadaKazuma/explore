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
import meanFFT_sortedseason
import meanmovingFFT_sorteddP
from Dispersion_Relation import Params

def process_focusmovingFFTlist(MUTC_h, timerange, windowsize_FFT):
    '''
    ダストデビルが1つも発生していない火星日のうち、
    MUTC_h(時刻)からtimerange秒間に対応する気圧の時系列データを全て加工し、
    全てのパワースペクトルとその移動平均を列挙したリストを返す関数

    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り出す時間)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''
    #記録用配列の作成    
    fft_xlist, fft_ylist = [], []
    moving_fft_xlist, moving_fft_ylist = [], []

    #ダストデビルの発生がないsolのリストを作成
    nodevilsollist = nodevil.process_nodevilsollist()

    for sol in tqdm(nodevilsollist, desc="Processing nodevil sol"):
        #該当sol付近の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None or data.empty:
            continue
        
        #該当範囲の抽出
        focus_data = focuschange_p.filter_focusdata(data, sol, MUTC_h, timerange)
        if focus_data is None or focus_data.empty:
            continue

        #加工済みデータを0.5秒でresample      
        focus_data = meanFFT_sortedseason.data_resample(focus_data, 0.5)

        focus_data = nearFFT.calculate_residual(focus_data)
        '''
        「countdown」、「p-pred」、「residual」カラムの追加
        countdown:経過時間(秒) ※countdown ≦ 0
        p-pred:線形回帰の結果(気圧(Pa))
        residual:残差 (Pa)
        '''

        #パワースペクトルとその移動平均の導出
        fft_x, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.moving_FFT(focus_data, windowsize_FFT)

        #記録用配列に追加
        fft_xlist.append(fft_x)
        fft_ylist.append(fft_y)
        moving_fft_xlist.append(moving_fft_x)
        moving_fft_ylist.append(moving_fft_y)
        
    return fft_xlist, fft_ylist, moving_fft_xlist, moving_fft_ylist
            

def plot_focusmeanmovingFFT(MUTC_h, timerange, windowsize_FFT):
    '''
    ダストデビルが1つも発生していない火星日のうち、
    MUTC_h(時刻)からtimerange秒間に対応する気圧の時系列データを全て加工し、
    パワースペクトルとその移動平均の両者を平均を描画した画像を保存する関数。
    横軸:周波数(Hz) 縦軸:スペクトル強度(Pa^2)

    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り出す時間)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    '''
    try:
        #対応する全事象のパワースペクトルとその移動平均をリスト化したものの導出
        fft_xlist, fft_ylist, moving_fft_xlist, moving_fft_ylist = process_focusmovingFFTlist(MUTC_h, timerange, windowsize_FFT)
        if not fft_xlist or not fft_ylist or not moving_fft_xlist or not moving_fft_ylist:
            raise ValueError("No data")
        
        #パワースペクトルとその移動平均両者のケース平均の導出
        fft_x = meanmovingFFT_sorteddP.process_arrays(fft_xlist, np.nanmean)
        fft_y = meanmovingFFT_sorteddP.process_arrays(fft_ylist, np.mean) 
        moving_fft_x = meanmovingFFT_sorteddP.process_arrays(moving_fft_xlist, np.nanmean)
        moving_fft_y = meanmovingFFT_sorteddP.process_arrays(moving_fft_ylist, np.nanmean)
        
        #音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        #プロットの設定
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(fft_x, fft_y, label='FFT')    
        plt.plot(moving_fft_x, moving_fft_y, label='FFT_Movingmean')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MPS_MUTC={MUTC_h}:00~{timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel(f'Pressure Power [$Pa^2$]')
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'meanfocusmovingFFT,MUTC={MUTC_h}:00~'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{timerange}s,windowsize_FFT={windowsize_FFT}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed:{timerange}s.png")
        
        return moving_fft_x, moving_fft_y

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the average moving power spectrum for each LTST_h")
    parser.add_argument('MUTC_h', type=int, help='Base start time') #基準となる開始の時間
    parser.add_argument('timerange', type=int, help='timerange(s)') #時間間隔(切り出す時間)の指定(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average") #パワースペクトルの移動平均を計算する際の窓数の指定
    args = parser.parse_args()
    plot_focusmeanmovingFFT(args.MUTC_h, args.timerange, args.windowsize_FFT)