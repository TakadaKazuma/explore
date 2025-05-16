import datetime as datetime
import matplotlib.pyplot as plt
import os
import argparse as argparse
import dailychange_p
import neardevil

def filter_ondevil(data, MUTC, timerange):
    '''
    指定された時刻 MUTC (地方時) を基準に
    timerange 秒前 〜 MUTC
    の範囲でデータを抽出する関数。
    
    data : 気圧の時系列データ (DataFrame)
    MUTC : ダストデビル発生時刻 (datetime)
    timerange : 切り取る時間範囲 (秒) (int)
    interval : 開始オフセット (秒) (int)
    '''
    stop = MUTC 
    start = stop - datetime.timedelta(seconds=timerange)
    filtered_data = data.query('@start < MUTC < @stop').copy()
    
    return filtered_data

def process_ondevil(ID, timerange):
    '''
    指定された ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データを取得・処理し、所定の形式で返す関数。

    ID : ダストデビルの識別番号
    timerange : 切り取る時間範囲 (秒) (int)
    '''
    try:
        # ID に対応する sol および MUTC を取得
        sol, MUTC = neardevil.get_sol_MUTC(ID)

        # 該当 sol 周辺の時系列データを取得
        data = dailychange_p.process_surround_dailydata(sol)
        if data is None:
            raise ValueError("Failed to retrieve time-series data.")
        
        # MUTC 付近のデータを抽出
        on_devildata = filter_ondevil(data, MUTC, timerange)
        if on_devildata is None or on_devildata.empty:
            raise ValueError("No data available after filtering.")
        
        # 残差計算を実施
        on_devildata = neardevil.calculate_countdown(on_devildata)
        '''
        追加カラム:
        - countdown: Dust Devil 発生までの秒数 (countdown ≦ 0)
        - p-pred: 線形回帰による気圧予測値 (Pa)
        - residual: 気圧の残差
        '''
        
        return on_devildata, sol
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_ondevil(ID, timerange):
    '''
    ID に対応する MUTC (ダストデビル発生時刻)直前の
    時系列データの気圧変化のプロットを保存する関数。

    - X軸 : countdown (s) [発生までの時間]
    - Y軸 : 気圧 (Pa)

    ID: ダストデビルの識別番号
    timerange: 切り取る時間範囲 (秒) (int)
    '''  
    try:
        # 描画する時系列データの取得
        on_devildata, sol = process_ondevil(ID, timerange)

        if on_devildata is None or on_devildata.empty:
            raise ValueError(f"No data available for sol={sol}")

        # 描画の設定
        plt.plot(on_devildata['countdown'], on_devildata['p'])
        plt.xlabel('Time until devil [s]', fontsize=15)
        plt.ylabel('Pressure [Pa]', fontsize=15)
        plt.title(f'ID={ID}, sol={sol}')
        plt.grid(True)
        plt.tight_layout()

        # 保存の設定
        output_dir = f'ondevil_{timerange}s'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"ID={str(ID).zfill(5)}, sol={str(sol).zfill(4)}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.clf()
        plt.close()
        print(f"Save completed: {filename}")

        return on_devildata

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=int, help="ID") #IDの指定
    parser.add_argument('timerange', type=int, help='timerange(s)') #時間間隔(切り出す時間)の指定(秒)
    args = parser.parse_args()
    plot_ondevil(args.ID, args.timerange)