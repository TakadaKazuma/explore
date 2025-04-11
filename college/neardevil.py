import datetime as datetime
import matplotlib.pyplot as plt
import os
import argparse as argparse
import DATACATALOG
import dailychange_p
import nearFFT

def get_sol_MUTC(ID):
    '''
    ID(通し番号)に対応するsolとMUTCを取得する関数
    ID:通し番号(datacatalog上でdustdevilに割り振られたもの)
    '''
    datacatalog = DATACATALOG.process_datacatalog()
    sol = datacatalog.sol[ID]
    MUTC = datacatalog.MUTC[ID]
    return sol, MUTC

def filter_neardevildata(data, MUTC, timerange, interval):
    '''
    指定時刻 MUTC の (timerange + interval) 秒前 〜 interval 秒前の範囲で 
    データを抽出し、DataFrameとして返す関数。

    data: 気圧の時系列データ (DataFrame)
    MUTC: Dust Devil 発生時刻 (datetime)
    timerange: 切り取る時間範囲 (秒) (int)
    interval: 開始オフセット (秒) (int)
    '''
    stop = MUTC - datetime.timedelta(seconds=interval)
    start = stop - datetime.timedelta(seconds=timerange)
    filtered_data = data.query('@start < MUTC < @stop').copy()
    
    return filtered_data

def calculate_countdown(data):
    '''
    データに Dust Devil 発生までの秒数を示す "countdown" カラムを追加。

    data: フィルタリングされた気圧の時系列データ (DataFrame)
    '''
    new_data = data.copy()
    
    new_data["countdown"] = - (new_data['MUTC'].iloc[-1] - new_data['MUTC']).dt.total_seconds()
    
    return new_data

def process_neardevildata(ID, timerange, interval):
    '''
    指定 ID の Dust Devil 発生直前データを取得・処理し、時系列データを返す。

    ID: ダストデビルの識別番号
    timerange: 切り取る時間範囲 (秒) (int)
    interval: 開始オフセット (秒) (int)
    '''
    try:
        # ID に対応する sol および MUTC を取得
        sol, MUTC = get_sol_MUTC(ID)

        # 該当 sol 周辺の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("Failed to retrieve time-series data.")
        
        # MUTC 付近のデータを抽出
        near_devildata = filter_neardevildata(data, MUTC, timerange, interval)
        if near_devildata is None or near_devildata.empty:
            raise ValueError("No data available after filtering.")

        # 残差計算を実施
        near_devildata = nearFFT.calculate_residual(near_devildata)
        '''
        追加カラム:
        - countdown: Dust Devil 発生までの秒数 (countdown ≦ 0)
        - p-pred: 線形回帰による気圧予測値 (Pa)
        - residual: 気圧の残差
        '''
        
        return near_devildata, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_neardevil(ID, timerange, interval):
    '''
    ID に対応する Dust Devil 発生直前の気圧時系列データと
    線形回帰の結果をプロットし、画像を保存する。

    - X軸: countdown (s) [発生までの時間]
    - Y軸: 気圧 (Pa)

    ID: ダストデビルの識別番号
    timerange: 切り取る時間範囲 (秒) (int)
    interval: 開始オフセット (秒) (int)
    '''   
    try:
        # 描画する時系列データの取得
        near_devildata, sol = process_neardevildata(ID, timerange, interval)
        if near_devildata is None or near_devildata.empty:
            raise ValueError("No data available.")

        # 描画の設定
        plt.plot(near_devildata['countdown'],near_devildata['p'],
                 label='True_Value')
        plt.plot(near_devildata['countdown'],near_devildata['p-pred'],
                 label='Linearize_Value')
        plt.xlabel('Time until devil starts [s]', fontsize=15)
        plt.ylabel('Pressure [Pa]', fontsize=15)
        plt.title(f'ID={ID},sol={sol}', fontsize=15)
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.tight_layout()
        
        # 保存の設定
        output_dir = f'neardevil_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"sol={str(sol).zfill(4)},ID={str(ID).zfill(5)}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")
        
        return near_devildata
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerange(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_neardevil(args.ID, args.timerange, 20)