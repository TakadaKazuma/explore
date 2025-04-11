import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearFFT
import nearmovingFFT
from Dispersion_Relation import Params

def calculate_ratio(x, y, y_movmean, windowsize_FFT):
    '''
    配列yとその移動平均y_movmeanの比: x/y を形状を維持しつつ導出し、
    yと対をなすxを連動させる関数

    x:連動させる配列(ndarray型)
    y:分母(ndarray)
    '''
    #失われている要素数(片側分)の算出
    pad_size = (windowsize_FFT - 1) // 2
    
    #計算可能な範囲のみを抽出
    x_new = x[pad_size:-pad_size]
    y_new = y[pad_size:-pad_size]
    y_movmean_new = y_movmean[pad_size:-pad_size]
    
    #比を計算
    ratio = y_new/y_movmean_new
    
    return x_new, ratio

def filter_xUlimit(x, y, xUlimit):
    '''
    指定された上限以上の値をとるxの要素に全てnanに変更し、
    そのxの要素に対応したyの要素もnanに変更する関数。

    x:フィルタリングを行いたい配列(ndarray型)
    y:xに連動する配列(ndarray型)
    x_Ulimit:xの基準となる上限(int型)
    '''
    # 配列の型をNan取り扱うことが可能なものに変換
    x = x.astype(float)
    y = y.astype(float)

    # 条件を満たす要素を
    x[x >= xUlimit] = np.nan
    y[x >= xUlimit] = np.nan

    return x, y

def process_ratio(ID, timerange, interval, windowsize_FFT):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データを対象とし、
    気圧変化の線形回帰から導かれる残差に対して、
    「パワースペクトルとその移動平均の比」及びそれに対応するsolを返す関数

    ID:ダストデビルに割り振られた通し番号(int型)
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
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
        
        near_devildata = nearFFT.calculate_residual(near_devildata)
        '''
        「countdown」、「p-pred」、「residual」カラムの追加
        countdown:経過時間(秒) ※countdown ≦ 0
        p-pred:線形回帰の結果(気圧(Pa))
        residual:残差
        '''
        
        #パワースペクトルとその移動平均の導出
        fft_x, fft_y, _, moving_fft_y = nearmovingFFT.moving_FFT(near_devildata, windowsize_FFT)

        #比の算出
        moving_fft_x, ratio = calculate_ratio(fft_x, fft_y, moving_fft_y, windowsize_FFT)
        
        #特定の周波数より高周波の情報をnanに変更
        moving_fft_x, ratio = filter_xUlimit(moving_fft_x, ratio, 0.8)

        return moving_fft_x, ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def plot_ratio(ID, timerange, interval, windowsize_FFT):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データを対象とし、
    気圧変化の線形回帰から導かれる残差に対して、
    「パワースペクトルとその移動平均の比」を描画した画像を保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度の比

    ID:ダストデビルに割り振られた通し番号(int型)
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize:パワースペクトルの移動平均を計算する際の窓数(int型)

    '''
    try:
        #パワースペクトルとその移動平均の比を導出
        moving_fft_x, ratio, sol = process_ratio(ID, timerange, interval, windowsize_FFT)
        
        #音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.plot(moving_fft_x, ratio, label='Ratio')
        plt.axvline(x=w, color='r', label='Border')
        plt.title(f'PSR_ID={ID}, sol={sol}, timerange={timerange}s', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Power Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'nearratio_{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}.png")
        
        return moving_fft_x, ratio
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the ratio of the power pectrum to its moving average for the given ID")
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    parser.add_argument('windowsize_FFT', type=int,
                         help="The [windowsize] used to calculate the moving average")  #パワースペクトルの移動平均を計算する際の窓数の指定
    args = parser.parse_args()
    plot_ratio(args.ID, args.timerange, 20, args.windowsize_FFT)