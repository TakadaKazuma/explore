import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
import os
import argparse as argparse
import dailychange_p
import neardevil
from Dispersion_Relation import Params

def calculate_residual(data):
    '''
    線形回帰を適用し、気圧予測値 "p-pred" と残差 "residual" を追加する。

    data: フィルタリング済みの時系列データ (DataFrame)
    '''
    new_data = data.copy()

    # 経過時間 (countdown) の計算
    new_data = neardevil.calculate_countdown(new_data)

    # 欠損値を含む行を削除
    new_data = new_data.dropna(subset=['p', 'countdown'])

    # 線形回帰 (説明変数: countdown, 目的変数: 気圧 p)
    t, p = new_data['countdown'].values, new_data['p'].values
    a, b = np.polyfit(t, p, 1)
    new_data["p-pred"] = a * t + b

    # 残差の計算
    new_data["residual"] = p - new_data['p-pred']

    return new_data

def FFT(data):
    '''
    残差 "residual" に対して FFT を適用し、パワースペクトルを算出する。

    data: フィルタリング済みの時系列データ (DataFrame)
    '''
    # サンプリング周波数の計算
    sampling_freq = 1 / np.mean(np.diff(data['countdown']))

    # FFT によるパワースペクトルの導出
    fft_x, fft_y = signal.periodogram(data['residual'].values, fs=sampling_freq)

    return fft_x, fft_y

def process_nearFFT(ID, timerange, interval):
    '''
    ID に対応する Dust Devil 発生直前の気圧時系列データから
    残差を求め、FFT によるパワースペクトルを算出する。

    ID: ダストデビルの識別番号 (int)
    timerange: 切り取る時間範囲 (秒) (int)
    interval: 開始オフセット (秒) (int)
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
        near_devildata = calculate_residual(near_devildata)
        '''
        追加カラム:
        - countdown: 経過時間 (秒) (countdown ≦ 0)
        - p-pred: 線形回帰による気圧予測値 (Pa)
        - residual: 気圧の残差
        '''

        # FFT によるパワースペクトルの導出
        fft_x, fft_y = FFT(near_devildata)

        return fft_x, fft_y, sol

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
def plot_nearFFT(ID, timerange, interval):
    '''
    ID に対応する Dust Devil 発生直前の気圧時系列データから
    残差のパワースペクトルを算出し、プロットを保存する。

    - X軸: 振動数 (Hz)
    - Y軸: スペクトル強度 (Pa^2)

    ID: ダストデビルの識別番号 (int)
    timerange: 切り取る時間範囲 (秒) (int)
    interval: 開始オフセット (秒) (int)
    '''
    try:
        # パワースペクトルの導出
        fft_x, fft_y, sol = process_nearFFT(ID, timerange, interval)

        # 音波と重力波の境界周波数を取得
        params = Params()
        w = params.border_Hz()

        # 描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-8, 1e2)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'PS_ID={ID}, sol={sol}, time_range={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Power [$Pa^2$]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()

        # 画像の保存
        output_dir = f'nearFFT_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")

        return fft_x, fft_y

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_nearFFT(args.ID, args.timerange, 20)