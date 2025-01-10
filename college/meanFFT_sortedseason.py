import numpy as np
import datetime as datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import os
import argparse as argparse
import DATACATALOG
import dailychange_p
import neardevil
import Dispersion_Relation
import nearFFT
import meanmovingFFT_sorteddP

def process_IDlist_ls(ls):
    '''
    lsに対応する疑似的なls(DATACATALOG.py参照)を計算し、
    datacatalogから疑似的なlsが一致しているIDのリストを作成する関数

    ls:季節を表す指標(int型)
    '''
    datacatalog = DATACATALOG.process_datacatalog()

    #疑似的なlsの導出
    pseudo_ls = (ls // 30) * 30 % 360

    #疑似的なlsを用いて、datacatalogをフィルタリング
    filtered_list = datacatalog[datacatalog['ls'] == pseudo_ls]

    return filtered_list['ID'].tolist(), pseudo_ls

def data_resample(data, s):
    '''
    与えられた時系列データをs秒でresampleする関数

    data:フィルタリング済みの時系列データ(dataframe)
    s:秒(int型)
    '''

    new_data = data.copy()

    #「MUTC」カラムをdatetime型に変換
    new_data["MUTC"]=pd.to_datetime(new_data["MUTC"],format="%Y-%m-%d %H:%M:%S.%f")

    #indexを先ほどの「MUTC」に変更し、これをもとにs秒でresample
    new_data = new_data.set_index("MUTC").resample(f"{s}S").mean()

    #indexのリセット
    new_data = new_data.reset_index()

    return new_data

def process_arrays(arrays, operation):
    '''
    Nanを含むリスト・ndarray等を除外し、全てを最短の長さにそろえ、
    指定された操作を各列に操作する関数

    arrays:多次元データ
    operation:施す操作 (例) median, np.max…など
    '''

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

def process_FFTlist_season(ls, timerange, interval):
    '''
    lsに対応する、疑似的なlsが一致している全て事象の時系列データを加工し、
    FFTを用いて導出したパワースペクトルをリスト化したものを返す関数

    ls:季節を表す指標 (int型)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''
    #記録用配列の作成
    fft_xlist, fft_ylist = [], []

    #lsに対応するIDリストの作成
    IDlist, LS = process_IDlist_ls(ls)

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
            near_devildata = data_resample(near_devildata, 0.5)
            if near_devildata is None:
                raise ValueError(f"No data:sol={sol}")
            
            near_devildata = nearFFT.calculate_residual(near_devildata)
            '''
            「countdown」、「p-pred」、「residual」カラムの追加
            countdown:経過時間(秒) ※countdown ≦ 0
            p-pred:線形回帰の結果(気圧(Pa))
            residual:残差
            '''

            #パワースペクトルの導出
            fft_x, fft_y =  nearFFT.FFT(near_devildata)

            #記録用配列に追加
            fft_xlist.append(fft_x)
            fft_ylist.append(fft_y)
            
        except ValueError as e:
            print(e)
            continue 
            
    return fft_xlist, fft_ylist, LS

def plot_meanFFT_season(ls, timerange, interval):
    '''
    lsに対応する、疑似的なlsが一致している全て事象の時系列データを加工し、
    FFTを用いて導出したパワースペクトルをケース平均し、それの描画した画像を保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度(Pa^2)

    ls:季節を表す指標 (int型)
    time_range:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''
    try:
        #対応する全事象のパワースペクトルをリスト化したものの導出
        fft_xlist, fft_ylist, LS = process_FFTlist_season(ls, timerange, interval)
        if not fft_xlist or not fft_ylist:
            raise ValueError(f"No data: ls={ls}")
        
        # パワースペクトルのケース平均を導出
        fft_x = meanmovingFFT_sorteddP.process_arrays(fft_xlist, np.nanmean)
        fft_y = meanmovingFFT_sorteddP.process_arrays(fft_ylist, np.nanmean)
        
        # 音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-8, 1e8)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'meanFFT_{LS}≦ ls <{LS+30},time_range={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel(f'Pressure Power [$Pa^2$]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanFFT_sortedseason_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"meanFFT_ls_{str(LS).zfill(3)}~{str(LS+30).zfill(3)}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: meanFFT_ls_{str(LS).zfill(3)}~{str(LS+30).zfill(3)}.png")
        
        return fft_x, fft_y

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the case average of the power spectrum corresponding to the ls")
    parser.add_argument('ls', type=int, help="ls(season)") #疑似的なlsの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_meanFFT_season(args.ls, args.timerange, 20)