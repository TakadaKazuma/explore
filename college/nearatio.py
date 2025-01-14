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


def process_ratio(ID, timerange, interval, windowsize_FFT):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、パワースペクトルとその移動平均を導出し、
    それらの比(ndarray型)及び対応するsol(int型)を返す関数

    ID:ダストデビルに割り振られた通し番号
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
        _, fft_y, moving_fft_x, moving_fft_y = nearmovingFFT.moving_FFT(near_devildata, windowsize_FFT)

        #比の算出
        ratio = fft_y/moving_fft_y

        return moving_fft_x, ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def plot_ratio(ID, timerange, interval, windowsize_FFT):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データに線形回帰を実行。
    これに伴い、導出できる残差に対して、パワースペクトルとその移動平均を計算し、それらの比を描画した画像を保存する関数
    横軸:周波数(Hz) 縦軸:スペクトル強度の比

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize:パワースペクトルの移動平均を計算する際の窓数(int型)

    '''
    try:
        #パワースペクトルとその移動平均の比を導出
        moving_fft_x, ratio, sol = process_ratio(ID, timerange, interval, windowsize_FFT)
        
        #音波と重力波の境界に該当する周波数
        w = Dispersion_Relation.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.plot(moving_fft_x, ratio, label='Ratio')
        plt.axvline(x=w, color='r', label='Border')
        plt.title(f'FFT_ID={ID}, sol={sol}, timerange={timerange}s')
        plt.xlabel('Vibration Frequency [Hz]')
        plt.ylabel(f'Pressure Power Ratio')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'nearratio_{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_ratio.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}_ratio.png")
        
        return moving_fft_x, ratio
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the ratio of the power pectrum to its moving average for the given ID")
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    #パワースペクトルの移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize_FFT', type=int, help="The [windowsize] used to calculate the moving average")
    args = parser.parse_args()
    plot_ratio(args.ID, args.timerange, 20, args.windowsize_FFT)