import datetime as datetime
import matplotlib.pyplot as plt
import os
import argparse as argparse
import DATACATALOG
import dailychange_p
import nearFFT
import neardevil

def filter_afterdevildata(data, MUTC, timerange, interval):
    '''
    与えられた時系列データを「MUTCの(interval)秒後」～ 「MUTCの(interval+timerange)秒後」の区間で切り取り、そのデータを返す関数(dataframe型)

    data:気圧の時系列データ(dataframe)
    MUTC:dustdevil発生時刻※ 火星地方日時 (datetime型)
    timerange:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切るか)(秒)(int型)
    '''
    start = MUTC + datetime.timedelta(seconds=interval)
    stop = start + datetime.timedelta(seconds=timerange)
    filtered_data = data.query('@start < MUTC < @stop').copy()
    
    return filtered_data

def calculate_timecount(data):
    '''
    フィルタリング済みの時系列データに最後までの秒数を示す「timecount」のカラムを追加する関数
    →dustdevil発生寸前までの時間を示す「timecount」のカラムを追加する関数
    ※timecount ≦ 0

    data:フィルタリングされた気圧の時系列データ(dataframe)
    '''
    new_data = data.copy()
    
    #経過時間(秒)の計算
    new_data["timecount"] = (new_data['MUTC'] - new_data['MUTC'].iloc[0]).dt.total_seconds()

    
    return new_data

def process_afterdevildata(ID, timerange, interval):
    '''
    IDに対応する「dustdevilの発生直後からtimarange秒後まで」の気圧の時系列データを返す関数

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒後から切り取るか)(秒)(int型)
    '''
    try:
        #IDに対応するsol及びMUTCを取得
        sol, MUTC = neardevil.get_sol_MUTC(ID)

        #該当sol付近の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("")
        
        #該当時系列データをdustdevil近辺でフィルタリング
        after_devildata = filter_afterdevildata(data, MUTC, timerange, interval)
        if after_devildata is None:
            raise ValueError("")
        
        after_devildata = calculate_timecount(after_devildata)
        '''
        「timecount」、「p-pred」、「residual」カラムの追加
        timecount:経過時間(秒) ※timecount ≦ 0
        p-pred:線形回帰の結果(気圧(Pa))
        residual:残差
        '''
        
        return after_devildata, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_afterdevil(ID, timerange, interval):
    '''
    IDに対応する「dustdevilの発生直後からtimarange秒後まで」における、
    気圧の時系列データ及びその線形回帰の結果を描画した画像を保存する関数
    ※横軸:timecount(s) 縦軸:気圧(Pa)

    ID:ダストデビルに割り振られた通し番号
    timerange:時間間隔(切り取る時間)(秒)(int型)
    interval:ラグ(何秒前から切り取るか)(秒)(int型)
    '''
    try:
        #描画する時系列データの取得
        after_devildata, sol = process_afterdevildata(ID, timerange, interval)
        if after_devildata is None:
            raise ValueError(f"No data:sol={sol}")

        #描画の設定
        plt.plot(after_devildata['timecount'],after_devildata['p'],
                 label='true_value')
        #plt.plot(after_devildata['timecount'],after_devildata['p-pred'],
        #         label='Liafterize_value')
        plt.xlabel('Elapsed time after dust devil  [s]', fontsize=15)
        plt.ylabel('Pressure [Pa]', fontsize=15)
        plt.title(f'ID={ID},sol={sol}', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        #保存の設定
        output_dir = f'afterdevil_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}.png"))
        plt.close()
        print(f"Save completed:sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}.png")
        
        return after_devildata
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerange(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_afterdevil(args.ID, args.timerange, 20)
