import datetime as datetime
import matplotlib.pyplot as plt
from scipy import signal
import os
import argparse as argparse
import dailychange_p
import neardevil
import nearFFT
import nearmovingFFT
import nearratio
import nearmovingratio
import meanFFT_sortedseason
from Dispersion_Relation import Params

def process_movingratio_resample(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データを対象とし、
    0.5秒間隔でのリサンプリングを行った後、気圧変化の線形回帰から導かれる残差に対して、
    「パワースペクトルとその移動平均の比(パワースペクトル比)」が算出できる。
    更にパワースペクトル比に対して、
    再度移動平均をとった修正パワースペクトル及び対応するsolを返す関数

    ID:ダストデビルに割り振られた通し番号(int型)
    timerange:時間間隔(切り出す時間)(秒)(int型)
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
        
        #加工済みデータを0.5秒でresample      
        near_devildata = meanFFT_sortedseason.data_resample(near_devildata, 0.5)
        
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
        moving_fft_x, ratio = nearratio.calculate_ratio(fft_x, fft_y, moving_fft_y, windowsize_FFT)

        #比の移動平均を算出
        moving_fft_x, moving_ratio = nearmovingratio.calculate_movingave(moving_fft_x, ratio, windowsize_ratio)
        
        #特定の周波数より高周波の情報をnanに変更
        #moving_fft_x, ratio = nearratio.filter_xUlimit(moving_fft_x, ratio, 0.8)

        return moving_fft_x, moving_ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_movingratio_resample(ID, timerange, interval, windowsize_FFT, windowsize_ratio):
    '''
    IDに対応する「dustdevilの発生直前 ~ 発生寸前」における気圧の時系列データを対象とし、
    0.5秒間隔でのリサンプリングを行った後、気圧変化の線形回帰から導かれる残差に対して、
    「パワースペクトルとその移動平均の比(パワースペクトル比)」が算出できる。
    更にパワースペクトル比に対して、
    再度移動平均をとった修正パワースペクトルを描画した画像を保存する関数。
    横軸:周波数(Hz) 縦軸:スペクトル強度の比

    ID:ダストデビルに割り振られた通し番号(int型)
    timerange:時間間隔(切り出す時間)(秒)(int型)
    interval:ラグ(何秒前から切り出すか)(秒)(int型)
    windowsize_FFT:パワースペクトルの移動平均を計算する際の窓数(int型)
    windowsize_ratio:パワースペクトルとその移動平均の比の移動平均を計算するときの窓数(int型)
    '''
    try:
        #パワースペクトルとその移動平均の比を導出し、更にその移動平均を算出する
        moving_fft_x, moving_ratio, sol = process_movingratio_resample(ID, timerange, interval, windowsize_FFT, windowsize_ratio)

        #音波と重力波の境界に該当する周波数
        params = Params()
        w = params.border_Hz()
        
        #描画の設定
        plt.xscale('log')
        plt.plot(moving_fft_x, moving_ratio, label='moving_ratio')        
        plt.axvline(x=w, color='r', label='border')
        plt.title(f'MPS_ID={ID}, sol={sol}, time_range={timerange}(s)', fontsize=15)
        plt.xlabel('Vibration Frequency [Hz]', fontsize=15)
        plt.ylabel('Pressure Amplitude Ratio', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'nearmovingratioresample_{timerange}s_windowsize_FFT={windowsize_FFT}'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)},windowsize_ratio={windowsize_ratio}.png"))
        plt.clf()
        plt.close()
        print(f"Save completed:sol={str(sol).zfill(4)},ID={str(ID).zfill(5)},windowsize_ratio={windowsize_ratio}.png")
        
        return moving_fft_x, moving_ratio, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the moving average of the ratio of the power spectrum to its moving average for the given ID")
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    parser.add_argument('windowsize_FFT', type=int, 
                        help="The [windowsize] used to calculate the moving average of FFT") #パワースペクトルの移動平均を計算する際の窓数の指定
    parser.add_argument('windowsize_ratio', type=int,
                         help="The [windowsize] used to calculate the moving average of ratio") #パワースペクトルとその移動平均の比の移動平均を計算する際の窓数の指定
    args = parser.parse_args()
    plot_movingratio_resample(args.ID, args.timerange, 20, args.windowsize_FFT, args.windowsize_ratio)