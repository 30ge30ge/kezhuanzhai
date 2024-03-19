import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backtrader as bt
from com.insight import common
from com.insight.query import *
from com.insight.market_service import market_service
from datetime import datetime
import calendar


numstocks = 5
final_weight = []
total_codes = []

def login():
    # 登陆前 初始化
    user = "替换账号"
    password = "替换密码"
    common.login(market_service, user, password)

def risk_min(RandomPortfolios, stock_returns):
    # 找到标准差最小数据的索引值
    min_index = RandomPortfolios.Volatility.idxmin()
    # 在收益-风险散点图中突出风险最小的点
    RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
    x = RandomPortfolios.loc[min_index, 'Volatility']
    y = RandomPortfolios.loc[min_index, 'Returns']
    plt.scatter(x, y, color='red')
    # 将该点坐标显示在图中并保留四位小数
    plt.text(np.round(x, 4), np.round(y, 4), (np.round(x, 4), np.round(y, 4)), ha='left', va='bottom', fontsize=10)
    plt.show()
    # 提取最小波动组合对应的权重, 并转换成Numpy数组
    GMV_weights = np.array(RandomPortfolios.iloc[min_index, 0:numstocks])
    # 计算GMV投资组合收益
    stock_returns['Portfolio_GMV'] = stock_returns.mul(GMV_weights, axis=1).sum(axis=1)
    return GMV_weights

def sharp_max(RandomPortfolios, stock_returns):
    # 设置无风险回报率为0
    risk_free = 0
    # 计算每项资产的夏普比率
    RandomPortfolios['Sharpe'] = (RandomPortfolios.Returns - risk_free) / RandomPortfolios.Volatility
    # 绘制收益-标准差的散点图，并用颜色描绘夏普比率
    plt.scatter(RandomPortfolios.Volatility, RandomPortfolios.Returns, c=RandomPortfolios.Sharpe)
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

    # 找到夏普比率最大数据对应的索引值
    max_index = RandomPortfolios.Sharpe.idxmax()
    # 在收益-风险散点图中突出夏普比率最大的点
    RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
    x = RandomPortfolios.loc[max_index, 'Volatility']
    y = RandomPortfolios.loc[max_index, 'Returns']
    plt.scatter(x, y, color='red')
    # 将该点坐标显示在图中并保留四位小数
    plt.text(np.round(x, 4), np.round(y, 4), (np.round(x, 4), np.round(y, 4)), ha='left', va='bottom', fontsize=10)
    plt.show()

    # 提取最大夏普比率组合对应的权重，并转化为numpy数组
    MSR_weights = np.array(RandomPortfolios.iloc[max_index, 0:numstocks])
    # 计算MSR组合的收益
    stock_returns['Portfolio_MSR'] = stock_returns.mul(MSR_weights, axis=1).sum(axis=1)
    #输出夏普比率最大的投资组合的权重
    print(MSR_weights)
    return MSR_weights

def Markowitz(total_codes, stock_returns):
    # method1:探索投资组合的最有方案，使用蒙特卡洛模拟Markowitz模型

    # 设置模拟的次数
    number = 1000
    # 设置空的numpy数组，用于存储每次模拟得到的权重、收益率和标准差
    random_p = np.empty((number, 7))
    # 设置随机数种子，这里是为了结果可重复
    np.random.seed(7)

    # 循环模拟1000次随机的投资组合
    for i in range(number):
        # 生成5个随机数，并归一化，得到一组随机的权重数据
        random5 = np.random.random(5)
        random_weight = random5 / np.sum(random5)

        # 计算年平均收益率
        mean_return = stock_returns.mul(random_weight, axis=1).sum(axis=1).mean()
        annual_return = (1 + mean_return) ** 252 - 1

        # 计算年化标准差，也成为波动率
        # 计算协方差矩阵
        cov_mat = stock_returns.cov()
        # 年化协方差矩阵
        cov_mat_annual = cov_mat * 252
        # 输出协方差矩阵
        print(cov_mat_annual)
        random_volatility = np.sqrt(np.dot(random_weight.T, np.dot(cov_mat_annual, random_weight)))

        # 将上面生成的权重，和计算得到的收益率、标准差存入数组random_p中
        random_p[i][:5] = random_weight
        random_p[i][5] = annual_return
        random_p[i][6] = random_volatility

    # 将Numpy数组转化为DataF数据框
    RandomPortfolios = pd.DataFrame(random_p)
    # 设置数据框RandomPortfolios每一列的名称
    RandomPortfolios.columns = [code + '_weight' for code in total_codes] + ['Returns', 'Volatility']

    # 绘制散点图
    RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
    plt.show()

    # weights = risk_min(RandomPortfolios, stock_returns)
    weights = sharp_max(RandomPortfolios, stock_returns)

    return weights

def weight_cal(total_codes, stock_returns):
    stock_returns['time'] = pd.to_datetime(stock_returns['time']).dt.date
    stock_returns = pd.pivot(stock_returns, index="time", columns="htsc_code", values="close")
    stock_returns.columns = [col + "_daily_return" for col in stock_returns.columns]
    stock_returns = stock_returns.pct_change().dropna()

    GMV_weights = Markowitz(total_codes, stock_returns)

    return GMV_weights

def cumulative_returns_plot(name_list, stock_returns):
    for name in name_list:
        CumulativeReturns = ((1+stock_returns[name]).cumprod()-1)
        CumulativeReturns.plot(label=name)
    plt.legend()
    plt.show()

def last_day_of_month(any_day):
    """
    获取获得一个月中的最后一天
    :param any_day: 任意日期
    :return: string
    """
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)

class Select_Strategy(bt.Strategy):
    def __init__(self):
        self.codes = total_codes
    def next(self):
        today = self.data.datetime.date()
        year, month = today.year, today.month
        d, month_length = calendar.monthrange(year, month)
        if today.day == month_length:
            for i in range(len(self.codes)):
                # final_weight = [0.1, 0.5, 0.3, 0.1, 0.1]
                self.order_target_percent(target=final_weight[i], data=self.codes[i])

if __name__ == '__main__':
    login()
    start_time = datetime(2016, 3, 1)
    end_time =datetime(2017, 12, 29)
    total_codes = ["601336.SH", "601398.SH", "601318.SH", '601888.SH', '603993.SH']

    df = get_kline(htsc_code=total_codes, time=[start_time, end_time],
                   frequency="daily", fq="pre")
    stock_returns = df.copy()
    stock_returns = stock_returns[['time','htsc_code','close']]
    final_weight = weight_cal(total_codes, stock_returns)

    cerebro = bt.Cerebro()
    for code in total_codes:
        data = df[df["htsc_code"] == code]
        date_feed = bt.feeds.PandasData(dataname=data, datetime="time", fromdate=start_time, todate=end_time)
        cerebro.adddata(date_feed, name=code)
        print('添加股票数据：code: %s' % code)
    cerebro.addstrategy(Select_Strategy)
    cerebro.broker.setcash(20000.0)

    result = cerebro.run()
    print(result)
    print("value: ", cerebro.broker.get_value())
    print("cash: ", cerebro.broker.getcash())
    cerebro.plot()
