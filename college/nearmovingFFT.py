import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearFFT
from Dispersion_Relation import Params

def movingFFT(data, windowsize_FFT):
    """
    フィルタリング済みデータにFFTを適用し、
    パワースペクトルとその移動平均を算出する関数。
    
    ※パワースペクトル各点は物理的意味とデータ長を保持するため、
    この関数では移動平均が計算できない領域にはNaNを付与して形状を維持している。
    また、低周波側の影響を抑制し高周波成分を正確に評価するため、
    パワーの常用対数を平滑化後、逆変換して移動平均を定義する。
    この手法により、全体構造を損なわず高周波側の解析精度を高める。
    
    data : フィルタリング済みの時系列データ(DattaFrame
    windowsize_FFT : パワースペクトルの移動平均に用いる窓数(int)
    """

    # パワースペクトルを計算
    fft_x, fft_y = nearFFT.FFT(data)

    # パワーの常用対数を計算
    log10_fft_y = np.log10(fft_y)

    # 移動平均用のフィルターを作成
    filter_frame = np.ones(windowsize_FFT) / windowsize_FFT

    # 欠損が発生する要素数の計算
    pad_size = (windowsize_FFT - 1) // 2

    # 形状を維持するため、全ての要素をNanに変更(初期化)
    moving_fft_x = np.full(fft_x.shape, np.nan)
    log10_moving_fft_y = np.full(fft_y.shape, np.nan)

    # 有効な範囲の抽出及び計算
    moving_fft_x[pad_size:-pad_size] = fft_x[pad_size:-pad_size]
    log10_moving_fft_y[pad_size:-pad_size] = np.convolve(log10_fft_y, filter_frame, mode="valid")

    # 対数を戻してパワーに変換
    moving_fft_y = 10**(log10_moving_fft_y)

    return fft_x, fft_y, moving_fft_x, moving_fft_y


def process_movingFFT(ID, timerange, interval, windowsize_FFT):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データにおける気圧残差を求め、
    それに対するパワースペクトルとその移動平均を算出する関数。

    ID: ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    windowsize_FFT :パワースペクトルの移動平均を計算する際に用いる窓数(int)
    '''
    try:
        # ID に対応する sol および MUTC を取得
        sol, MUTC = neardevil.get_sol_MUTC(ID)

        # 該当 sol 周辺の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("Failed to retrieve time-series data.")

        # MUTC 付近のデータを抽出
        near_devildata = neardevil.filter_neardevildata(data, MUTC, timerange, interval)
        if near_devildata is None or near_devildata.empty:
            raise ValueError("No data available after filtering.")

        # 残差計算を実施
        near_devildata = nearFFT.calculate_residual(near_devildata)
        '''
        追加カラム:
        - countdown: 経過時間 (秒) (countdown ≦ 0)
        - p-pred: 線形回帰による気圧予測値 (Pa)
        - residual: 気圧の残差
        '''
        
        # FFT によるパワースペクトルとその移動平均の導出
        fft_x, fft_y, moving_fft_x, moving_fft_y = movingFFT(near_devildata, windowsize_FFT)

        return  fft_x, fft_y, moving_fft_x, moving_fft_y, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def plot_movingFFT(ID, timerange, interval, windowsize_FFT):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データにおける気圧残差を求め、
    それに対するパワースペクトルとその移動平均を算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : スペクトル強度 (Pa^2)

    ID : ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    windowsize_FFT : パワースペクトルの移動平均に用いる窓数(int)
    '''
    try:
        # パワースペクトルとその移動平均の導出
        fft_x, fft_y, moving_fft_x, moving_fft_y, sol = process_movingFFT(ID, timerange, interval, windowsize_FFT)
        
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
        plt.title(f'PS_ID={ID}, sol={sol}, timerange={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel(f'Pressure Power [$Pa^2$]')
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 画像の保存
        output_dir = f'nearmovingFFT_{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"ID={str(ID).zfill(5)}, sol={str(sol).zfill(4)}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return moving_fft_x, moving_fft_y
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") # IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average of FFT") # パワースペクトルの移動平均に用いる窓数
    args = parser.parse_args()
    plot_movingFFT(args.ID, args.timerange, 20, args.windowsize_FFT)