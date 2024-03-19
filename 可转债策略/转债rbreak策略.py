import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

#数据降采样时要耦合


#获取数据
class Getdata():
    def __init__(self):
        self.code = None


    def mindata(self):
        data=pd.read_csv('可转债分时.txt')
        data.index = pd.to_datetime(data['时间'])
        data['openinterest'] = 0
        data = data[['开盘', '最高', '最低', '收盘', '成交量','openinterest','symbol']]
        columns = ['open', 'high', 'low', 'close', 'volume', 'openinterest','symbol']
        data.columns = columns

        return data

    def dailydata(self):
        data=pd.read_csv('可转债日线行情.txt')
        data.index = pd.to_datetime(data['date'])
        data['openinterest'] = 0
        data = data[['open', 'high', 'low', 'close', 'volume', 'openinterest', 'symbol','momentum_5', 'pivot', 'bBreak', 'bEnter']]
        return data


#拓展数据
class Dailydataextend(bt.feeds.PandasData):
    # 增加线
    lines = ('momentum_5','bBreak','bEnter', )
    params = (('momentum_5', -1),('bBreak', -1),('bEnter', -1),
              ('dtformat', '%Y-%m-%d'),)



#主策略逻辑
#init里写指标
#next里写买卖逻辑




class R_BreakStrategy(bt.Strategy):
    params = (
        ('lowestperiod', 5),
        ('trailamount', 0.0),
        ('trailpercent', 0.05),
    )
    # 日志函数
    def log(self, txt, dt=None):
        # 以第一个数据data0，即指数作为时间基准
        dt = dt or self.data0.datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.order = {}
        self.banks = []
        self.inds = dict()
        #小周期数据用（）进行耦合
        for i, d in enumerate(self.datas):
            self.inds[d]=dict()
            if 0 == i % 2:
                self.datas[i].bBreak = self.datas[i + 1].bBreak()
                self.datas[i].bEnter = self.datas[i + 1].bEnter()
                self.datas[i].momentum_5 = self.datas[i + 1].momentum_5()
                print(f'数据耦合 for {d._name}, DONE.')
                #读取selfdata里的列columns
                # print(d,self.datas[i+1],self.datas[i+1].lines.getlinealiases())





    def next(self):
        for i, d in enumerate(self.datas):
            if 0 == i % 2:
                pos = self.getposition(d)
                cash = self.broker.cash/10

                print(d.close[0],self.datas[i+1].bEnter[0],d.lines.datetime.date(0),d._name)

                if d.close[0]>self.datas[i + 1].bEnter[0] and self.datas[i + 1].momentum_5[0]>1:
                    print(d.close[0],self.datas[i + 1].bEnter[0],self.datas[i + 1].momentum_5[0],d._name,d.lines.datetime.date(0))




















if __name__ == '__main__':

    from_idx = datetime(2023, 4, 1)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2023, 4, 15)
    print(from_idx, to_idx)
    #启动回测

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1000000.0)
    #获取数据源
    data = Getdata()
    data_0 = data.mindata()
    data_1 = data.dailydata()

    banks = data_1['symbol'].unique().tolist()
    for stk_code in banks[:5]:
        data_m = data_0[data_0['symbol'] == stk_code]
        #加载15分钟数据
        data_m = bt.feeds.PandasData(dataname=data_m, fromdate=from_idx, todate=to_idx, timeframe=bt.TimeFrame.Minutes,
                                     name='15m'+stk_code)
        cerebro.adddata(data_m)
        #加载1日数据(用自己算好的指标数据)
        data_dayly = data_1[data_1['symbol'] == stk_code]
        data_d = Dailydataextend(dataname=data_dayly, fromdate=from_idx, todate=to_idx, timeframe=bt.TimeFrame.Days,name='1d'+stk_code)
        cerebro.adddata(data_d)
        # cerebro.resampledata(data_m, name='1d', timeframe=bt.TimeFrame.Days)

    # 载入策略
    cerebro.addstrategy(R_BreakStrategy)
    print('add strategy DONE.')

    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # Plot the result绘制结果
    cerebro.plot(volume=False, style='candle', barup='red', bardown='green')