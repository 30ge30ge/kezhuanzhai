import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import tushare as ts
import akshare as ak
from  tqdm import tqdm


class Getdata():
    def __init__(self):
        self.code = None

    def kzzdata(self):
        data=pd.read_csv('可转债数据.csv')
        print(data.columns)
        data = data[['code','short_name','company_code','list_date','delist_Date']]
        columns = ['代码', '简称', '证券代码', '上市日期', '退市日期']
        data.columns = columns
        # 将退市日期转换为日期类型
        data['退市日期'] = pd.to_datetime(data['退市日期'])
        data_nan = data[data['退市日期'].isna()]
        # 筛选出退市日期在2023年1月1日之后的数据
        data_2023 = data[data['退市日期'] > pd.to_datetime('2023-01-01')]
        datamerge =pd.concat([data_nan,data_2023],axis =0)
        print(datamerge)
        datamerge['代码'] = datamerge['代码'].astype(str)
        datamerge['证券代码'] = datamerge['证券代码'].astype(str)
        datamerge['代码'] = datamerge['代码'].apply(lambda X: 'sh' + X[:6] if X[:2] == "11" else 'sz' + X[:6])
        datamerge =datamerge.reset_index(drop =True)




        # data.index = pd.to_datetime(data['时间'])
        # data['openinterest'] = 0
        # data = data[['开盘', '最高', '最低', '收盘', '成交量','openinterest','symbol']]
        # columns = ['open', 'high', 'low', 'close', 'volume', 'openinterest','symbol']
        # data.columns = columns

        return datamerge

    def getdata(self):
        self._data=self.kzzdata()
        banks=[]

        for i in tqdm (self._data['代码']):
            k = self._data.loc[self._data['代码'] == i, '证券代码'].values[0]
            j =k[:6]
            list_date =self._data.loc[self._data['代码'] == i, '上市日期'].values[0]
            delist_Date =self._data.loc[self._data['代码'] == i, '退市日期'].values[0]
            try:
                df = ak.bond_zh_hs_cov_daily(symbol=i)
                df['上市日期'] =list_date
                df['退市日期'] = delist_Date
                df['代码'] =i
                df['证券代码'] = k

                stock_a_indicator_df = ak.stock_a_indicator_lg(symbol=j)
                stock_a_indicator_df['total_mv'] =round(stock_a_indicator_df['total_mv'] / 10000, 2)
                stock_a_indicator_df =stock_a_indicator_df[['trade_date','total_mv']]
                stock_a_indicator_df.columns = ['date', 'total_mv']
                bond_merge =pd.merge(df,stock_a_indicator_df,on='date')

                #收集数据
                banks.append(bond_merge)

            except:
                continue
        newdata = pd.concat(banks)
        return newdata



if __name__ == '__main__':

    # from_idx = datetime(2023, 4, 1)  # 记录行情数据的开始时间和结束时间
    # to_idx = datetime(2023, 4, 15)
    # print(from_idx, to_idx)
    data = Getdata()
    df = data.getdata()
    df.to_csv('可转债带正股市值数据.csv',index=False)

    print(df)
