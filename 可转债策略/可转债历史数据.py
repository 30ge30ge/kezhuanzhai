import akshare as ak
from tqdm import tqdm
import pandas as pd
from scipy.stats import linregress
import numpy as np



class Getdata():
    def __init__(self):
        self._data = None
        self.page     = None
        self.pagesize = None


    def basicdata(self):
        bond_zh_hs_cov_spot_df = ak.bond_zh_hs_cov_spot()
        return bond_zh_hs_cov_spot_df

    def getdata(self):
        self._data=self.basicdata()
        banks=[]

        for i in tqdm (self._data['symbol']):
            try:
                df = ak.bond_zh_hs_cov_daily(symbol=i)
                df['symbol']=i

                # 动量策略
                data_m5 = round(df['close'].rolling(5).apply(self.momentum_func).to_frame('momentum').reset_index(), 2)
                df['momentum_5'] = data_m5['momentum']

                # Rbreak突破策略
                df['pivot'] = round((df['high'].shift() + df['low'].shift() + df['close'].shift()) / 3, 2)  # '中枢点'
                df['bBreak'] = round(df['high'].shift() + 2 * (df['pivot'] - df['low'].shift()), 2)  # 突破买入价
                df['bEnter'] = round(2 * df['pivot'] - df['high'].shift(), 2)  # 反转买入价

                #收集数据
                banks.append(df)

            except:
                continue
        newdata = pd.concat(banks)
        return newdata

    def momentum_func(self, the_array):
        r = np.log(the_array)
        slope, _, rvalue, _, _ = linregress(np.arange(len(r)), r)
        annualized = (1 + slope) ** 252
        return annualized * (rvalue ** 2)





    def getmindata(self):
        self._data = self.basicdata()
        minbanks=[]

        for i in tqdm (self._data['symbol']):
            try:
                min_df=ak.bond_zh_hs_cov_min(symbol=i, adjust='', period='15',start_date="1979-09-01 09:32:00", end_date="2222-01-01 09:32:00")
                min_df['symbol'] = i
                # 收集数据
                minbanks.append(min_df)
            except:
                continue
        newdata = pd.concat(minbanks)
        return newdata


    def savedata(self):
        data=self.getmindata()
        data.to_csv('20230824可转债分时.txt')


if __name__ == "__main__":
    data=Getdata()
    data.savedata()






