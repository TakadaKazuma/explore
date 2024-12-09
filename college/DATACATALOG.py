import datetime as datetime
import pandas as pd

def UTC_to_ls(UTC):
    '''
    UTC(地球地方時)からls(季節を表す指標)を求める関数
    
    UTC:地球地方時(datetime型)
    '''
    ranges = [
        ((datetime.datetime(2019, 3, 23), datetime.datetime(2019, 5, 24)), 0),
        ((datetime.datetime(2021, 2, 7), datetime.datetime(2021, 4, 10)), 0),
        ((datetime.datetime(2019, 5, 25), datetime.datetime(2019, 7, 30)), 30),
        ((datetime.datetime(2021, 4, 11), datetime.datetime(2021, 6, 16)), 30),
        ((datetime.datetime(2019, 7, 31), datetime.datetime(2019, 10, 7)), 60),
        ((datetime.datetime(2021, 6, 17), datetime.datetime(2021, 8, 24)), 60),
        ((datetime.datetime(2019, 10, 8), datetime.datetime(2019, 12, 12)), 90),
        ((datetime.datetime(2021, 8, 25), datetime.datetime(2021, 10, 29)), 90),
        ((datetime.datetime(2019, 12, 13), datetime.datetime(2020, 2, 11)), 120),
        ((datetime.datetime(2021, 10, 30), datetime.datetime(2021, 12, 29)), 120),
        ((datetime.datetime(2020, 2, 12), datetime.datetime(2020, 4, 7)), 150),
        ((datetime.datetime(2021, 12, 30), datetime.datetime(2022, 2, 23)), 150),
        ((datetime.datetime(2020, 4, 8), datetime.datetime(2020, 5, 28)), 180),
        ((datetime.datetime(2022, 2, 24), datetime.datetime(2022, 4, 15)), 180),
        ((datetime.datetime(2020, 5, 29), datetime.datetime(2020, 7, 15)), 210),
        ((datetime.datetime(2022, 4, 16), datetime.datetime(2022, 6, 2)), 210),
        ((datetime.datetime(2020, 7, 16), datetime.datetime(2020, 9, 1)), 240),
        ((datetime.datetime(2018, 10, 16), datetime.datetime(2018, 12, 3)), 270),
        ((datetime.datetime(2020, 9, 2), datetime.datetime(2020, 10, 20)), 270),
        ((datetime.datetime(2018, 12, 4), datetime.datetime(2019, 1, 24)), 300),
        ((datetime.datetime(2020, 10, 21), datetime.datetime(2020, 12, 21)), 300),
        ((datetime.datetime(2019, 1, 25), datetime.datetime(2019, 3, 22)), 330),
        ((datetime.datetime(2020, 12, 22), datetime.datetime(2021, 2, 6)), 330)
    ]
    
    for (start, end), value in ranges:
        if start <= UTC <= end:
            return value
    return None

def LTSTh_to_MUTC(row):
    '''
    LTST_h(火星の旧地方時:実数)からMUTC(火星の地方時:datetime型)を作成する関数
    '''
    BASE_DATE = datetime.datetime(2018, 11, 26)
    ltst_hours = int(row['LTST_h'])
    ltst_minutes = int((row['LTST_h'] - ltst_hours) * 60)
    ltst_seconds = int((((row['LTST_h'] - ltst_hours) * 60) - ltst_minutes) * 60)
    utc_milliseconds = row['UTC'].microsecond

    date = BASE_DATE + datetime.timedelta(days=row['sol'])

    MUTC = date.replace(hour=ltst_hours, minute=ltst_minutes, second=ltst_seconds, microsecond=utc_milliseconds)

    return MUTC


def process_datacatalog():
    '''
    datacatalogを作成する関数
    '''
    datacatalog = pd.read_csv( "~/2025B_takada/work/InSight_CV_Catalog_v3.csv", skiprows=1, 
    usecols=[0, 2, 3, 4, 5, 153, 154, 155, 156, 157, 158],
    names=["ID", "sol", "LTST_h", "UTC", "dP", "Ws-ave", "Ws-std", "Wd-ave", "Wd-std", "AT-ave", "AT-std"],
    encoding="cp932"
    )
    datacatalog["UTC"] = pd.to_datetime(datacatalog["UTC"], format="%Y-%jT%H:%M:%S.%fZ")
    datacatalog['MUTC'] = datacatalog.apply(LTSTh_to_MUTC, axis=1)
    datacatalog['ls'] = datacatalog['UTC'].apply(UTC_to_ls)
    return datacatalog

datacatalog = process_datacatalog()