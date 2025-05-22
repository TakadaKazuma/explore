import numpy as np
import datetime as datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import os
import argparse as argparse
import DATACATALOG
import dailychange_p
import neardevil
import nearFFT
import meanmovingFFT_sorteddP
from Dispersion_Relation import Params

def process_IDlist_ls(ls):
    '''
    ls に対応する疑似的な ls (DATACATALOG.py 参照)を計算し、
    datacatalog から疑似的な ls が一致しているIDのリストを作成する関数

    ls : 季節を表す指標(int)
    '''
    datacatalog = DATACATALOG.process_datacatalog()

    #疑似的なlsの導出
    pseudo_ls = (ls // 30) * 30 % 360

    #疑似的なlsを用いて、datacatalogをフィルタリング
    filtered_list = datacatalog[datacatalog['ls'] == pseudo_ls]

    return filtered_list['ID'].tolist(), pseudo_ls

def data_resample(data, s):
    '''
    与えられた時系列データをs秒間隔でresampleする関数

    data : フィルタリング済みの時系列データ(DataFrame)
    s : 秒(int)
    '''

    new_data = data.copy()

    #「MUTC」カラムをdatetime型に変換
    new_data["MUTC"]=pd.to_datetime(new_data["MUTC"],format="%Y-%m-%d %H:%M:%S.%f")

    # indexを「MUTC」に変更し、これを基準にs秒でresample
    new_data = new_data.set_index("MUTC").resample(f"{s}S").mean()

    # indexのリセット
    new_data = new_data.reset_index()

    return new_data

'''
def process_arrays(arrays, operation):
    
    Nanを含むリスト・ndarray等を除外し、全てを最短の長さにそろえ、
    指定された操作を各列に操作する関数

    arrays:多次元データ
    operation:施す操作 (例) median, np.max…など
    

    #Nanを含むリスト・ndarray等の除外
    trimmed_arrays = [arr for arr in arrays if len(arr) > 0 and not np.any(np.isnan(arr))]

    #全て除外された場合は、空のリストを返す
    if not trimmed_arrays:
        return []

    #最短の長さの取得
    min_length = min(len(arr) for arr in trimmed_arrays)

    #全てのデータを最短の長さにそろえる
    trimmed_arrays = [arr[:min_length] for arr in trimmed_arrays]
    
    return [operation(values) for values in zip(*trimmed_arrays)]
'''

def process_FFTlist_season(ls, timerange, interval):
    '''
    疑似的な ls が一致している全ての ID に対応する、
    MUTC (ダストデビル発生時刻) 直前の時系列データにおける気圧残差を求め、
    各ケースのパワースペクトルをまとめたリストを返す関数。

    ls : 季節を表す指標 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''
    # 記録用配列の作成
    fft_xlist, fft_ylist = [], []

    # 疑似的なlsが一致するIDをリスト化
    IDlist, LS = process_IDlist_ls(ls)
    for ID in tqdm(IDlist, desc="Processing IDs"):
        try:
            # ID に対応する sol および MUTC を取得
            sol, MUTC = neardevil.get_sol_MUTC(ID)

            # 該当sol付近の時系列データを取得
            data = dailychange_p.process_surround_dailydata(sol)
            if data is None:
                raise ValueError("Failed to retrieve time-series data.")
            
            # MUTC 付近の時系列データを取得
            near_devildata = neardevil.filter_neardevildata(data, MUTC, timerange, interval)
            if near_devildata is None or near_devildata.empty:
                raise ValueError("No data available after filtering.")

            # 0.5秒間隔でresample
            near_devildata = data_resample(near_devildata, 0.5)
                
            # 残差計算を実施
            near_devildata = nearFFT.calculate_residual(near_devildata)
            '''
            追加カラム:
            - countdown: 経過時間 (秒) (countdown ≦ 0)
            - p-pred: 線形回帰による気圧予測値 (Pa)
            - residual: 気圧の残差
            '''

            # FFT によるパワースペクトルの導出
            fft_x, fft_y =  nearFFT.FFT(near_devildata)

            # 結果を配列に記録
            fft_xlist.append(fft_x)
            fft_ylist.append(fft_y)
            
        except ValueError as e:
            print(e)
            continue 
            
    return fft_xlist, fft_ylist, LS

def plot_meanFFT_season(ls, timerange, interval):
    '''
    疑似的な ls が一致している全ての ID に対応する、
    MUTC (ダストデビル発生時刻) 直前の時系列データにおける気圧残差を求め、
    各ケースのパワースペクトルの平均を算出し、プロットを保存する関数。
    
    - X軸 : 振動数 (Hz)
    - Y軸 : スペクトル強度 (Pa^2)

    ls : 季節を表す指標 (int)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''
    try:
        # 各ケースにおけるパワースペクトルをまとめたリストの導出
        fft_xlist, fft_ylist, LS = process_FFTlist_season(ls, timerange, interval)
        
        # パワースペクトルのケース平均を算出
        fft_x = meanmovingFFT_sorteddP.process_arrays(fft_xlist, np.nanmean)
        fft_y = meanmovingFFT_sorteddP.process_arrays(fft_ylist, np.nanmean)
        
        # 音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-6, 1e2)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MPS_{LS}≦ ls <{LS+30},time_range={timerange}s', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel(f'Pressure Power [$Pa^2$]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanFFT_sortedseason_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"ls is More{str(LS).zfill(3)} and less{str(LS+30).zfill(3)}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return fft_x, fft_y

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ls', type=int, help="ls(season)") # 疑似的なlsの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') # 切り取る時間範囲(秒)
    args = parser.parse_args()
    plot_meanFFT_season(args.ls, args.timerange, 20)