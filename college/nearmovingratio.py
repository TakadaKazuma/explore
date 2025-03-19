import numpy as np
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
import meanFFT_sortedseason
import meanmovingFFT_sorteddP
from Dispersion_Relation import Params

def process_focusmovingratiolist(MUTC_h, timerange, windowsize_FFT, windowsize_ratio):
    '''
    ダストデビルが1つも発生していない火星日のうち、
    MUTC_h(時刻)からtimerange秒間に対応する気圧の時系列データを全て加工し、
    「パワースペクトルとその移動平均の比(パワースペクトル比)」が算出できる。
    更にパワースペクトル比に対して、
    再度移動平均をとった修正パワースペクトルを列挙したリストを返す関数

    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り出す時間)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
    '''

    #記録用配列の作成 
    moving_fft_xlist, moving_ratiolist = [], []

    #ダストデビルの発生がないsolのリストを作成
    nodevilsollist = nodevil.process_nodevilsollist()

    for sol in tqdm(nodevilsollist, desc="Processing nodevil sol"):
        #該当する時系列データの取得
        focus_data = focuschange_p.process_focusdata_p(sol, MUTC_h, timerange)
        if focus_data is None or focus_data.empty:
            continue

        #加工済みデータを0.5秒でresample      
        focus_data = meanFFT_sortedseason.data_resample(focus_data, 0.5)

        focus_data = nearFFT.calculate_residual(focus_data)
        '''
        「countdown」、「p-pred」、「residual」カラムの追加
        countdown:経過時間(秒) ※countdown ≦ 0
        p-pred:線形回帰の結果(気圧(Pa))
        residual:残差
        '''
        
        #パワースペクトルの導出
        fft_x, fft_y = nearFFT.FFT(focus_data)
            
        #パワースペクトルとその移動平均の導出
        fft_x, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.moving_FFT(focus_data, windowsize_FFT)

        #比の算出
        moving_fft_x, ratio = nearratio.calculate_ratio(fft_x, fft_y, moving_fft_y, windowsize_FFT)
        
        #比の移動平均を算出
        moving_fft_x, moving_ratio = nearmovingratio.calculate_movingave(moving_fft_x, ratio, windowsize_ratio)
            
        #記録用配列に保存
        moving_fft_xlist.append(moving_fft_x) 
        moving_ratiolist.append(moving_ratio)
        
        count += 1 
            
        if count >= 15:
            break
            
    return moving_fft_xlist, moving_ratiolist

def plot_focusmeanmovingratio(MUTC_h, timerange, windowsize_FFT, windowsize_ratio):
    '''
    ダストデビルが1つも発生していない火星日のうち、
    MUTC_h(時刻)からtimerange秒間に対応する気圧の時系列データを全て加工し、
    パワースペクトルとその移動平均の比を平均を描画した画像を保存する関数。
    横軸:周波数(Hz) 縦軸:スペクトル強度の比

    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り出す時間)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
    '''
    try:
        #対応する全事象のパワースペクトルとその移動平均の比をリスト化したもの(修正パワースペクトル)を導出
        moving_fft_xlist, moving_ratiolist = process_focusmovingratiolist(MUTC_h, timerange, windowsize_FFT, windowsize_ratio)
        if not moving_fft_xlist or not moving_ratiolist:
            raise ValueError("No data")
        
        #修正パワースペクトルのケース平均の導出
        moving_fft_x =  meanmovingFFT_sorteddP.process_arrays(moving_fft_xlist, np.nanmean)
        moving_ratio = meanmovingFFT_sorteddP.process_arrays(moving_ratiolist, np.nanmean)
        
        #特定の周波数より高周波の情報をnanに変更
        #moving_fft_x, moving_ratio = nearratio.filter_xUlimit(moving_fft_x, moving_ratio, 0.8)
        
        #音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        #プロットの設定
        plt.xscale('log')
        plt.plot(moving_fft_x, moving_ratio, label='moving ratio')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MMPS_MUTC={MUTC_h}:00~{timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel(f'Pressure Amplitude Ratio')
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'meanfocustwicemovingratio,MUTC={MUTC_h}:00~{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{timerange}s_windowsize_ratio={windowsize_ratio}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed:{timerange}s_windowsize_ratio={windowsize_ratio}.png")
        
        return moving_fft_x, moving_ratio

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the average moving power spectrum for each LTST_h")
    parser.add_argument('MUTC_h', type=int, help='Base start time') #基準となる開始の時間
    parser.add_argument('timerange', type=int, help='timerange(s)') #時間間隔(切り出す時間)の指定(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average") #パワースペクトルの移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize_ratio', type=int, 
                        help="The [windowsize] used to calculate the moving average of ratio") #パワースペクトルとその移動平均の比の移動平均を計算する際の窓数の指定
    args = parser.parse_args()
    plot_focusmeanmovingratio(args.MUTC_h, args.timerange, args.windowsize_FFT, args.windowsize_ratio)
