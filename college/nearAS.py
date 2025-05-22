import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fft
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearFFT
from Dispersion_Relation import Params

def AS(data):
    '''
    残差 "residual" に対して FFT を適用し、振幅スペクトルを算出する関数。

    data : フィルタリング済みの時系列データ (DataFrame)
    '''
    # サンプリング周波数の定義
    sampling_freq = 1 / np.mean(np.diff(data['countdown']))

    # フーリエ変換
    Y = scipy.fft.fft(data['residual'].to_numpy(), norm='forward')

    # データ数の取得
    N = len(data['residual'])
    
    # 周波数軸の作成
    AS_x = scipy.fft.fftfreq(N, d=sampling_freq)

    # 対称性を利用した調整
    Y = Y[:N//2]
    AS_x = AS_x[:N//2]
    Y = 2*Y

    # 過大評価を防ぐための操作(データ長が奇数である場合のみ)
    if N % 2 == 1:
        Y[0] = Y[0]/2

    # 振幅スペクトルの算出
    AS_y = np.abs(Y)

    return AS_x, AS_y


def process_nearAS(ID, timerange, interval):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データにおける気圧残差を求め、
    それに対する振幅スペクトルを算出する関数。

    ID : ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
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

        # FFT による振幅スペクトルの導出
        AS_x, AS_y = AS(near_devildata)

        return AS_x, AS_y, sol

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_nearAS(ID, timerange, interval):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データにおける気圧残差を求め、
    それに対する振幅スペクトルを算出し、プロットを保存する関数。

    - X軸 : 振動数 (Hz)
    - Y軸 : スペクトル強度 (Pa)

    ID : ダストデビルの識別番号 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''
    try:
        # 振幅スペクトルの導出
        AS_x, AS_y, sol = process_nearAS(ID, timerange, interval)

        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()

        # 描画の設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-3,1e4)
        plt.plot(AS_x, AS_y, label='AS')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'AS_ID={ID}, sol={sol}, timerange={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Amplitude [Pa]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()

        # 保存の設定
        output_dir = f'nearAS_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"ID={str(ID).zfill(5)}, sol={str(sol).zfill(4)}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")

        return AS_x, AS_y

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") # IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    args = parser.parse_args()
    plot_nearAS(args.ID, args.timerange, 20)