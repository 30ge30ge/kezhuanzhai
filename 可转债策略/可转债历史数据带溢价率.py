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
        self.savedata()


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


                # 可转债溢价率
                j=i[2:]
                bond_zh_cov_value_analysis_df = ak.bond_zh_cov_value_analysis(symbol=j)


                #溢价率合并
                df=pd.merge(df,bond_zh_cov_value_analysis_df,left_on='date',right_on='日期')


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




    def savedata(self):
        data=self.getdata()
        data.to_csv('可转债日线数据带溢价率.txt')







if __name__ == "__main__":
    # data=Getdata()
    bond_cb_jsl_df = ak.bond_cb_jsl('kbzw__Session=p2sc91vg0s1b4e1g3seafhm8d2;Hm_lvt_164fe01b1433a19b507595a43bf58262=1710728547;kbz_newcookie=1;kbzw__user_login=7Obd08_P1ebax9aXwZKpmKqoprGaoIKvpuXK7N_u0ejF1dSe2ZnSxqTapaKpn6yVqZSsqdWpmqCUrd-toNrQqpjck6mYrqXW2cXS1qCbqZ-tl6iXmLKgubXOvp-qrJqpo6marpeomK6ltrG_0aTC2PPV487XkKylo5iJvcLX4uPd6N_fnZaq5evY5IG9wteZxLyZxJeTpsCorNKvipCi5OnhztDR2a3f1aaspq-Po5eUocCxzbnDjpbN4OLYmKjVxN_onom81OnR48amqKWqj6CPpKeliczN3cPoyqaspq-Po5c.;Hm_lpvt_164fe01b1433a19b507595a43bf58262=1710728882; SERVERID=0e73f01634e37a9af0b56dfcd9143ef3|1710729034|1710728548')
    print(bond_cb_jsl_df)







