import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
from  tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



class Getdata():
    def __init__(self):
        self.code = None

    def kzzdata(self):
        data=pd.read_csv('可转债带正股市值数据.csv')
        data['date'] = pd.to_datetime(data['date'])
        data['退市日期'] = pd.to_datetime(data['退市日期'])
        data['上市日期'] = pd.to_datetime(data['上市日期'])
        data = data.sort_values(by=['代码', 'date'])
        return data



# 要拓展的数据
class Dailydataextend(bt.feeds.PandasData):
    # 增加资金费率线
    lines = ('total_mv',)
    params = (('total_mv', -1),
              ('dtformat', '%Y-%m-%d'),)


class bondStrategy(bt.Strategy):
    params = (
        ('num_volume', 50),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        log_message = '%s, %s' % (dt.isoformat(), txt)
        print(log_message)
        self.log_file.write(log_message + '\n')

    def __init__(self):
        # 删除多余字符 's'
        self.log_file = open('strategy_fund.txt', 'a')
        self.log_file_closed = False
        self.lastRanks = []  # 上次交易标的列表
        self.stocks = self.datas
        self.savedf = pd.DataFrame(columns=['rank','股票名称', '总市值', '日期'])

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单状态 submitted/accepted，无动作
            return

        # 订单完成
        if order.status in [order.Completed]:
            with open("orders.txt", "a") as file:
                # 记录手续费

                execution_time = self.datas[0].datetime.datetime()
                if order.isbuy():
                    file.write('买单执行, %s, %.2f, %i, %s' % (
                        order.data._name, order.executed.price, order.executed.size, execution_time) + "\n")

                elif order.issell():
                    # 平仓时扣除手续费

                    file.write('卖单执行, %s, %.2f, %i, %s, 资金费用: %.2f' % (
                        order.data._name,
                        order.executed.price,
                        order.executed.size,
                        execution_time,
                        order.executed.comm
                    ) + "\n")
    def next(self):
        print('调仓日')
        print('next 账户总值', self.data0.datetime.datetime(0), self.broker.get_cash())
        # for d in self.stocks:
        #     print(d.fundingRate[0],d._name)
        ranks = [d for d in self.stocks if
                 len(d) > 0 and d.volume > 1
                 and d.close > 100
                 ]
        ranks.sort(key=lambda d: d.total_mv, reverse=True)  # 按市值从大到小
        ranks = ranks[0:self.p.num_volume]  # 取前num_volume名

        if len(ranks) != 0:
            for i, d in enumerate(ranks):
                print(f'选股第{i + 1}名,{d._name},total_mv值: {d.total_mv[0]},日期: {self.data0.datetime.datetime(0)},')
                data_to_save = {
                    'rank': i + 1,
                    '股票名称': d._name,
                    '总市值': d.total_mv[0],
                    '日期': self.data0.datetime.datetime(0)
                }
                self.savedf = self.savedf._append(data_to_save, ignore_index=True)
        else:  # 无债选入
            return

        data_toclose = set(self.lastRanks) - set(ranks)
        for d in data_toclose:
            print('不在本次池里：平仓', d._name, self.getposition(d).size)
            self.close(data=d)

        buypercentage = (1 - 0.02) / len(ranks)
        targetvalue = self.broker.get_cash() / len(ranks)

        ranks.sort(key=lambda d: self.broker.getvalue([d]), reverse=True)

        for d in ranks:
            # size = int(abs((self.broker.getvalue([d.data]) - targetvalue) / d.data.open[0] // 100 * 100))
            size = int(abs(targetvalue / d.open[0] // 10 * 10))
            o = self.buy(data=d, size=size)

        self.lastRanks = ranks

    def stop(self):
        # 保存 DataFrame 到 CSV 文件
        self.savedf.to_csv('可转债市值筛选_stocks.csv', index=False)


if __name__ == '__main__':

    from_idx = datetime(2022, 1, 1)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2024, 3, 15)
    print(from_idx, to_idx)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000000.0)
    #获取数据源
    df = Getdata()
    data_1 = df.kzzdata()

    data_1 = data_1[data_1['date'] > '2021-12-31']
    data_1['date'] = pd.to_datetime(data_1['date'])


    print(data_1,data_1.columns)
    data_1 = data_1.set_index('date')
    data_1['openinterest'] =1

    banks = data_1['代码'].unique().tolist()

    out=[]
    data = pd.DataFrame(index=data_1.index.unique())
    for stk_code in tqdm(banks):
        df = data_1[data_1['代码'] == stk_code]
        df = df.sort_index()
        data_ = pd.merge(data, df, left_index=True, right_index=True, how='left')
        data_.fillna(0, inplace=True)
        data_d = Dailydataextend(dataname=data_, fromdate=from_idx, todate=to_idx, timeframe=bt.TimeFrame.Days,name=stk_code)
        cerebro.adddata(data_d)
        out.append(stk_code)
    #
    print('统计数量为{}'.format(len(out)), 'Done !')

    cerebro.addstrategy(bondStrategy)
    print('add strategy DONE.')

    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')

    print('add analyzers DONE.')
    start_portfolio_value = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    end_portfolio_value = cerebro.broker.getvalue()
    pnl = end_portfolio_value - start_portfolio_value

    # 输出结果、生成报告、绘制图表
    print(f'初始本金 Portfolio Value: {start_portfolio_value:.2f}')
    print(f'最终本金和 Portfolio Value: {end_portfolio_value:.2f}')
    print(f'利润PnL: {pnl:.2f}')

    pnl = pd.Series(results[0].analyzers._TimeReturn.get_analysis())
    # 计算累计收益
    cumulative = (pnl + 1).cumprod()
    # 计算回撤序列
    max_return = cumulative.cummax()
    drawdown = (cumulative - max_return) / max_return
    # 计算收益评价指标
    import pyfolio as pf

    # 按年统计收益指标
    perf_stats_year = (pnl).groupby(pnl.index.to_period('y')).apply(
        lambda data: pf.timeseries.perf_stats(data)).unstack()
    # 统计所有时间段的收益指标
    perf_stats_all = pf.timeseries.perf_stats((pnl)).to_frame(name='all')
    perf_stats = pd.concat([perf_stats_year, perf_stats_all.T], axis=0)
    perf_stats_ = round(perf_stats, 4).reset_index()


    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    import matplotlib.ticker as ticker  # 导入设置坐标轴的模块

    plt.style.use('seaborn')  # plt.style.use('dark_background')

    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.5, 4]}, figsize=(20, 8))
    cols_names = ['date', 'Annual\nreturn', 'Cumulative\nreturns', 'Annual\nvolatility',
                  'Sharpe\nratio', 'Calmar\nratio', 'Stability', 'Max\ndrawdown',
                  'Omega\nratio', 'Sortino\nratio', 'Skew', 'Kurtosis', 'Tail\nratio',
                  'Daily value\nat risk']

    # 绘制表格
    ax0.set_axis_off()  # 除去坐标轴
    table = ax0.table(cellText=perf_stats_.values,
                      bbox=(0, 0, 1, 1),  # 设置表格位置， (x0, y0, width, height)
                      rowLoc='right',  # 行标题居中
                      cellLoc='right',
                      colLabels=cols_names,  # 设置列标题
                      colLoc='right',  # 列标题居中
                      edges='open'  # 不显示表格边框
                      )
    table.set_fontsize(13)

    # 绘制累计收益曲线
    ax2 = ax1.twinx()
    ax1.yaxis.set_ticks_position('right')  # 将回撤曲线的 y 轴移至右侧
    ax2.yaxis.set_ticks_position('left')  # 将累计收益曲线的 y 轴移至左侧
    # 绘制回撤曲线
    drawdown.plot.area(ax=ax1, label='drawdown (right)', rot=0, alpha=0.3, fontsize=13, grid=False)
    # 绘制累计收益曲线
    (cumulative).plot(ax=ax2, color='#F1C40F', lw=3.0, label='cumret (left)', rot=0, fontsize=13, grid=False)
    # 不然 x 轴留有空白
    ax2.set_xbound(lower=cumulative.index.min(), upper=cumulative.index.max())
    # 主轴定位器：每 5 个月显示一个日期：根据具体天数来做排版
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # 同时绘制双轴的图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper left', ncol=1)

    fig.tight_layout()  # 规整排版
    plt.show()
