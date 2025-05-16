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
    ダストデビルが発生しなかったsolのリストを返す関数
    '''
    # 全solが含まれる整数リストの作成
    all_sols = set(range(1219 + 1))

    #ダストデビルが発生したsolのリストを作成
    ondevil_sollist = set(ondevil_sols())

    #ダストデビルが発生しなかったsolのリストを作成
    nodevil_sollist = all_sols - ondevil_sollist

    return nodevil_sollist

def process_nodevilsollist():
    '''
    ダストデビルが発生しなかったsolのうち、
    時系列データが存在するsolのリストを返す関数
    '''
    nodevil_sollist= [
    5, 10, 30, 120, 189, 232, 260, 266, 284, 370, 384, 422, 473, 477, 482, 
    488, 502, 503, 510, 550, 551, 552, 553, 554, 555, 556, 557, 567, 612, 
    637, 650, 666, 678, 680, 683, 702, 720, 721, 722, 723, 724, 725, 726, 
    727, 728, 729, 730, 735, 749, 754, 774, 780, 793, 794, 802, 807, 816, 
    822, 823, 825, 826, 831, 837, 839, 843, 845, 847, 853, 859, 860, 861, 
    862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 875, 876, 877, 878, 
    880, 882, 884, 885, 886, 887, 888, 889
    ]
    
    return nodevil_sollist