import DATACATALOG
def ondevil_sols():
    '''
    ダストデビルが発生したsolをリストとして返す関数
    '''
    datacatalog = DATACATALOG.process_datacatalog()
    # `sol` カラムの重複を削除し、昇順に並べてリスト化
    return sorted(datacatalog['sol'].unique())

def nodevil_sols():
    '''
    ダストデビルが発生しなかったsolをリストとして返す関数
    '''
    # 全solが含まれる整数リストの作成
    all_sols = set(range(1219 + 1))

    #ダストデビルが発生したsolのリストを作成
    ondevil_sollist = set(ondevil_sols())

    #ダストデビルが発生しなかったsolのリストを作成
    nodevil_sollist = all_sols - ondevil_sollist

    return nodevil_sollist