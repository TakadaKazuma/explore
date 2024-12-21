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
import meanFFT_sortedseason

def process_IDlist_ATandWs(AT_Llimit, Ws_Ulimit):
    '''
    datacatalogから AT-ave>AT_Llimit かつ Ws-ave<Ws_Ulimit を満たすIDのリストを作成する関数

    AT_Llimit:下限の基準となる大気の温度(K) (int型)
    Ws_Ulimit:上限の基準となる風速(m/s) (int型)
    '''
    datacatalog = DATACATALOG.process_datacatalog()
    filtered_list = datacatalog[(datacatalog['AT-ave'] > AT_Llimit)&(datacatalog['Ws-ave'] < Ws_Ulimit)]
    return filtered_list['ID'].tolist()


def process_FFTlist_ATandWs(AT_Llimit, Ws_Ulimit, time_range, interval):
    '''
    AT-ave>AT_Llimit かつ Ws-ave<Ws_Ulimit を満たす全て事象の時系列データを加工し、
    求めたFFTをリスト化したものを返す関数

    AT_Llimit:下限の基準となる大気の温度(K) (int型)
    Ws_Ulimit:上限の基準となる風速(m/s) (int型)
    time_range:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''

    #記録用配列の作成
    fft_xlist, fft_ylist = [], []

    #AT-ave>AT_Llimit かつ Ws-ave<Ws_Ulimit を満たすIDリストの作成
    IDlist= process_IDlist_ATandWs(AT_Llimit, Ws_Ulimit)
    for ID in tqdm(IDlist, desc="Processing IDs"):
        try:
            #IDに対応するsol及びMUTCを取得
            sol, MUTC = neardevil.get_sol_MUTC(ID)

            #該当sol付近の時系列データを取得
            data = dailychange_p.process_surround_dailydata(sol)
            if data is None:
                raise ValueError("")
            
            #該当範囲の抽出
            near_devildata = neardevil.filter_neardevildata(data, MUTC, time_range, interval)

            #加工済みデータを1秒でresample
            near_devildata = meanFFT_sortedseason.data_resample(near_devildata, 1)
            if near_devildata is None:
                raise ValueError("No data")

              
            near_devildata = neardevil.caluculate_residual(near_devildata)
            '''
            「countdown」、「p-pred」、「residual」カラムの追加
            countdown:経過時間(秒) ※countdown ≦ 0
            p-pred:線形回帰の結果(気圧(Pa))
            residual:残差 (Pa)
            '''

            #FFTの導出
            fft_x, fft_y =  nearFFT.FFT(near_devildata)

            #記録用配列に追加
            fft_xlist.append(fft_x)
            fft_ylist.append(fft_y)
            
        except ValueError as e:
            print(e)
            continue 
            
    return fft_xlist, fft_ylist

def plot_meanFFT_ATandWs(AT_Llimit, Ws_Ulimit, time_range, interval):
    '''
    AT-ave>AT_Llimit かつ Ws-ave<Ws_Ulimit を満たす全て事象の時系列データを加工し、
    求められるFFTをケース平均したものを描画し、保存する関数

    AT_Llimit:下限の基準となる大気の温度(K) (int型)
    Ws_Ulimit:上限の基準となる風速(m/s) (int型)
    time_range:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''
    try:
        #対応する全事象のFFTをリスト化したものの導出
        fft_xlist, fft_ylist = process_FFTlist_ATandWs(AT_Llimit, Ws_Ulimit, time_range, interval)
        if not fft_xlist or not fft_ylist:
            raise ValueError("No data")
        
        # FFTのケース平均を導出
        fft_x = meanFFT_sortedseason.process_arrays(fft_xlist, np.mean)
        fft_y = meanFFT_sortedseason.process_arrays(fft_ylist, np.mean)
        
        # 音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-8, 1e8)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'AT >{AT_Llimit},Ws<{Ws_Ulimit}(time_range={time_range}(s))')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel('Pressure Amplitude [Pa]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanFFT_ATandWs(time_range={time_range}s)'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"meanFFT_AT_{AT_Llimit}~,Ws_~{Ws_Ulimit}~.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: meanFFT_AT_{AT_Llimit}~,Ws_~{Ws_Ulimit}~.png")
        
        return fft_x, fft_y

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the case average of the power spectrum corresponding to the AT_Ulimit and Ws_Llimit")
    parser.add_argument('AT_Ulimit', type=int, help="Serves as the standard for the upper limit of AT_ave(K)")
    parser.add_argument('Ws_Llimit', type=int, help='Serves as the standard for the upper limit of Ws_ave(m/s)')
    parser.add_argument('time_range', type=int, help='time_rang(s)')
    args = parser.parse_args()
    plot_meanFFT_ATandWs(args.AT_Ulimit, args.Ws_Llimit, args.time_range, 20)