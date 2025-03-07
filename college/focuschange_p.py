import datetime
import matplotlib.pyplot as plt
import os
import argparse as argparse
import dailychange_p
import nodevil
import nearFFT
from tqdm import tqdm

def filter_focusdata(data, sol, MUTC_h, timerange):
    '''
    与えられた時系列データを指定されたMUTC_hからtimerange秒間のデータを返す関数(dataframe型)

    data:気圧の時系列データ(dataframe)
    sol:取り扱う火星日(探査機到着後からの経過日数)(int型)
    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    '''
    #基準となる時刻の設定
    start_time = datetime.time(MUTC_h,0,0)

    #solに対応するMUTCを計算した後に、それに応じたデータのフィルタリング
    date = datetime.date(2018, 11, 26)+datetime.timedelta(days=sol)
    start = datetime.datetime.combine(date, start_time)
    stop = start + datetime.timedelta(seconds=timerange)
    filtered_data = data.query('@start < MUTC < @stop').copy()
    
    return filtered_data

def process_focusdata_p(sol, MUTC_h, timerange):
    '''
    sol,時刻MUTC_hからtimerange秒間に対応する
    気圧の時系列データ及びその線形回帰の結果を返す関数

    sol:取り扱う火星日(探査機到着後からの経過日数)(int型)
    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    '''
    #該当sol付近の時系列データを取得
    focus_data = dailychange_p.process_surround_dailydata(sol)
    
    #該当データを時刻MUTC_hからtimerange秒間でフィルタリング
    focus_data = filter_focusdata(focus_data, sol, MUTC_h, timerange)

    focus_data = nearFFT.calculate_residual(focus_data)
    '''
    「countdown」、「p-pred」、「residual」カラムの追加
    countdown:経過時間(秒) ※countdown ≦ 0
    p-pred:線形回帰の結果(気圧(Pa))
    residual:残差
    '''
    return focus_data

def plot_focuschange_p(sol, MUTC_h, timerange):
    '''
    sol,時刻MUTC_hからtimerange秒間に対応する
    気圧の時系列データ及びその線形回帰の結果を描画した画像を保存する関数

    sol:取り扱う火星日(探査機到着後からの経過日数)(int型)
    MUTC_h:基準となる開始時刻(int型)(0 ≦ MUTC_h ≦ 23)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    '''
    try:
        #描画する時系列データの取得
        focus_data = process_focusdata_p(sol, MUTC_h, timerange)
        if focus_data is None:
            raise ValueError(f"No data:sol={sol}")
         
        #プロットの設定
        plt.tight_layout()
        plt.plot(focus_data["countdown"],focus_data["p"],
            label='true_value')
        plt.plot(focus_data["countdown"],focus_data["p-pred"],
            label='Linearize_value')
        plt.title(f'sol={sol},MUTC={MUTC_h}:00~{timerange}s', fontsize=15)
        plt.xlabel('Local Time', fontsize=15)
        plt.ylabel('Pressure [Pa]', fontsize=15)
        plt.grid(True)
        
        #保存の設定
        output_dir = f'focuschange_p,MUTC={MUTC_h}:00~'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)}_{timerange}s.png"))
        plt.clf()
        plt.close()
        print(f"Save completed: sol={str(sol).zfill(4)}_{timerange}s.png")

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
   
    return focus_data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot focus pressure changes corresponding to the sol, LTST_h and timerange")
    parser.add_argument('MUTC_h', type=int, help='Base start time') #基準となる開始の時間
    parser.add_argument('timerange', type=int, help='timerang(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    
    #ダストデビルが発生しなかったが、時系列データが存在するの気圧変化の描画
    nodevilsollist = nodevil.process_nodevilsollist()
    for sol in tqdm(nodevilsollist, desc="Processing nosdevil sols"):
        plot_focuschange_p(sol, args.MUTC_h, args.timerange)