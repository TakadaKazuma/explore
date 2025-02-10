import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearFFT
import Dispersion_Relation
import nearmovingFFT

def calculate_movingave(x, y, windowsize):
    '''
    x及びyの移動平均を算出する関数
    ※用途は主にratioの移動平均を算出

    x:移動平均を算出したいndarray
    y:xと対をなす移動平均を算出したいndarray
    windowsize:移動平均を計算する際の窓数(int型)
    '''

    '''
    以下では形状を維持しつつ、x及びyの移動平均を導出
    ※主にx,yはパワースペクトルをその移動平均で割ったものが対象となる。
    パワースペクトルの各要素及び長さは、それ自体に意味があるため、
    パワースペクトルから算出される計算の対象も同様である。
    移動平均を算出した際にその情報が壊れないようにするため、
    正確な移動平均の値が計算できない要素にnanを代入し、
    形状を維持するようにしている。
    '''
    filter_frame = np.ones(windowsize) / windowsize
    pad_size = (windowsize - 1) // 2
    moving_x = np.full(x.shape, np.nan)
    moving_y = np.ones(y.shape, np.nan)
    
    moving_x[pad_size:-pad_size] = x[pad_size:-pad_size]
    moving_y[pad_size:-pad_size] = np.convolve(y, filter_frame, mode='valid')
    
    return moving_x, moving_y


def filter_xUlimit(x, y, xUlimit):
    '''
    指定された上限以上の値をとるxの要素に全てnanに変更し、
    そのxの要素に対応したyの要素もnanに変更する関数。

    x:フィルタリングを行いたいndarray
    y:ndarray
    x_Ulimit:xの基準となる上限(int型)
    '''
    # 配列の型をNan取り扱うことが可能なものに変換
    x = x.astype(float)
    y = y.astype(float)

    # 条件を満たす要素を
    x[x >= xUlimit] = np.nan
    y[x >= xUlimit] = np.nan

    return x, y


def process_movingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、パワースペクトルとその移動平均を導出し、
    それらの比に対して再度移動平均を算出したもの(ndarray型)及び対応するsol(int型)を返す関数

    ID:ダストデビルに割り振られた通し番号
    time_range:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
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
        fft_x, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.moving_FFT(near_devildata, windowsize_FFT)

    
        #特定の周波数より高周波の情報をnanに変更
        fft_x, fft_y = filter_xUlimit(fft_x, fft_y, 4.5)
        moving_fft_x, moving_fft_y = filter_xUlimit(moving_fft_x, moving_fft_y, 4.5)
    

        #比の算出
        ratio = fft_y/moving_fft_y

        #比の移動平均を算出
        twice_moving_fft_x, moving_ratio = calculate_movingave(moving_fft_x, ratio, windowsize_ratio)

        return twice_moving_fft_x, moving_ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_movingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、パワースペクトルとその移動平均を導出し、
    それらの比に対して再度移動平均を算出したものを描画した画像を保存する関数

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
    '''
    try:
        #パワースペクトルとその移動平均の比を導出し、更にその移動平均を算出する
        twice_moving_fft_x, moving_ratio, sol = process_movingratio(ID, timerange, interval, windowsize_FFT, windowsize_ratio)

        #音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.plot(twice_moving_fft_x, moving_ratio, label='moving_ratio')        
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'FFT_ID={ID}, sol={sol}, time_range={timerange}(s)', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Amplitude Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'nearmovingratio_{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)},windowsize_ratio={windowsize_ratio},movingratio.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},ID={str(ID).zfill(5)},windowsize_ratio={windowsize_ratio},movingratio.png")
        
        return twice_moving_fft_x, moving_ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the moving average of the ratio of the power spectrum to its moving average for the resampled data corresponding to the given ID.")
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    #パワースペクトルの移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize_FFT', type=int, help="The [windowsize] used to calculate the moving average of FFT")
    #パワースペクトルとその移動平均の比の移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize_ratio', type=int, help="The [windowsize] used to calculate the moving average of ratio")
    args = parser.parse_args()
    plot_movingratio(args.ID, args.timerange, 20, args.windowsize_FFT, args.windowsize_ratio)
