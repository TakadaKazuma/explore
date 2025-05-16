import datetime as datetime
import matplotlib.pyplot as plt
import os
import argparse as argparse
import dailychange_p
import neardevil
import afterdevil

def calculate_returntimecount(data):
    '''
    データに Dust Devil 発生からの秒数を示す "timecount" カラムを追加。

    data : フィルタリングされた気圧の時系列データ (DataFrame)
    '''
    new_data = data.copy()
    
    new_data["timecount"] =  - (new_data['MUTC'].iloc[-1] - new_data['MUTC']).dt.total_seconds()

    
    return new_data

def process_returnafterdevildata(ID, timerange, interval):
    '''
    指定 ID の Dust Devil 発生直後データを取得・処理し、時系列データを返す。

    ID : ダストデビルの識別番号
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
        after_devildata = afterdevil.filter_afterdevildata(data, MUTC, timerange, interval)
        if after_devildata is None  or after_devildata.empty:
            raise ValueError("No data available after filtering.")
        
        after_devildata = calculate_returntimecount(after_devildata)
        '''
        追加カラム:
        - timecount: 経過時間 (秒) (timecount ≧ 0)
        - p-pred: 線形回帰による気圧予測値 (Pa)
        - residual: 気圧の残差
        '''
        
        return after_devildata, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_returnafterdevil(ID, timerange, interval):
    '''
    ID に対応する Dust Devil 発生直前の気圧時系列データと
    線形回帰の結果をプロットし、画像を保存する。

    - X軸 : timecount (s) [発生からの経過時間]
    - Y軸 : 気圧 (Pa)

    ID : ダストデビルの識別番号
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''   
    try:
        # 描画する時系列データの取得
        after_devildata, sol = process_returnafterdevildata(ID, timerange, interval)

        # 描画の設定
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
        
        # 保存の設定
        output_dir = f'returnafterdevil_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"ID={str(ID).zfill(5)}, sol={str(sol).zfill(4)}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return after_devildata
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerange(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_returnafterdevil(args.ID, args.timerange, 20)