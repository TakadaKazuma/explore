import numpy as np
import datetime as datetime
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
import meanmovingFFT_sorteddP

def process_IDlist_dP(dP_Ulimit):
    '''
    datacatalogから dP_Ulimit> dP を満たすIDのリストを作成する関数

    dP_Ulimit:上限となる気圧効果量(Pa) (int型)
    ※dP, dP_max < 0
    '''
    datacatalog = DATACATALOG.process_datacatalog()
    filtered_list = datacatalog[datacatalog['dP'] < dP_Ulimit ]
    return filtered_list['ID'].tolist()

def process_FFTlist_dP(dP_Ulimit, timerange, interval):
    '''
    dP_Ulimit > dP を満たすダストデビル全ての時系列データを加工し、
    全てのパワースペクトルを列挙したリストを返す関数

    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_Ulimit < 0
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    '''
     #記録用配列の作成
    fft_xlist, fft_ylist = [], []

    #dP_Ulimit > dP を満たすIDリストの作成
    IDlist = process_IDlist_dP(dP_Ulimit)
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
            near_devildata = meanFFT_sortedseason.data_resample(near_devildata, 0.5)
            if near_devildata is None:
                raise ValueError("No data")
                
            near_devildata = nearFFT.calculate_residual(near_devildata)
            '''
            「countdown」、「p-pred」、「residual」カラムの追加
            countdown:経過時間(秒) ※countdown ≦ 0
            p-pred:線形回帰の結果(気圧(Pa))
            residual:残差 (Pa)
            '''

            #パワースペクトルの導出
            fft_x, fft_y =  nearFFT.FFT(near_devildata)

            #記録用配列に追加
            fft_xlist.append(fft_x)
            fft_ylist.append(fft_y)
            
        except ValueError as e:
            print(e)
            continue 
            
    return fft_xlist, fft_ylist

def plot_meanFFT_dP(dP_Ulimit, timerange, interval):
    '''
    dP_Ulimit > dP を満たすダストデビル全ての時系列データを加工し、
    パワースペクトルの平均を描画及び保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度(Pa^2)

    dP_Ulimit:上限となる気圧降下量(Pa) (int型)
    ※dP, dP_Ulimit < 0
    time_range:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    '''
    try:
        #対応する全事象のパワースペクトルをリスト化したものの導出
        fft_xlist, fft_ylist = process_FFTlist_dP(dP_Ulimit, timerange, interval)
        if not fft_xlist or not fft_ylist:
            raise ValueError("No data")
        
        #パワースペクトルのケース平均を導出
        fft_x = meanmovingFFT_sorteddP.process_arrays(fft_xlist, np.nanmean)
        fft_y = meanmovingFFT_sorteddP.process_arrays(fft_ylist, np.nanmean)
        
        # 音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        # プロットの設定
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-6, 1e2)
        plt.plot(fft_x, fft_y, label='FFT')
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'dP <{dP_Ulimit}.timerange={timerange}s', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel(f'Pressure Power [$Pa^2$]', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'meanFFT_dP_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"meanFFT_dP_~{dP_Ulimit}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: meanFFT_dP_~{dP_Ulimit}.png")
        
        return fft_x, fft_y

    except ValueError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the case average of the power spectrum corresponding to the dP_Ulimit") 
    parser.add_argument('dP_Ulimit', type=int, help="Serves as the standard for the upper limit of dP_ave(Negative int)") #dPの上限の指定(負)
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_meanFFT_dP(args.dP_Ulimit, args.timerange, 20)