from PyQt5 import QtCore, QtWidgets, QtGui
from numpy import arange, double
import pandas as pd
import numpy as np
import datetime,requests,json,time,os
import akshare as ak
from scipy.stats import linregress
from tqdm import tqdm



import os,sys
os.chdir(sys.path[0])        #使用文件所在目录
sys.path.append(os.getcwd()) #添加工作目录到模块搜索目录列表

sortAsHeader      = '成交额占比'



class GetKZZFromWeb():
    def __init__(self) -> None:
        self.headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                        'Cache-Control': 'max-age=0',
                        'Connection': 'keep-alive',
                        'Cookie': 'kbz_newcookie=1; kbzw__user_login=7Obd08_P1ebax9aXwZusmK6pp6iVqYKvpuXK7N_u0ejF1dSesJWjx9neps_boK-S2JPYp9XcwtWR2t7WoqvNqZqpxaqYrqXW2cXS1qCasZ6umKaUmLKgzaLOvp_G5OPi2OPDpZalp5OguNnP2Ojs3Jm6y4KnkaWnrpi42c-qrbCJ8aKri5ai5-ff3bjVw7_i6Ziun66QqZeXn77Atb2toJnh0uTRl6nbxOLmnJik2NPj5tqYqpypkKaPp6WjmLTRx9Xr3piun66QqZc.; kbzw__Session=ttkf1d4jsug521au6ubqnva8b0; Hm_lvt_164fe01b1433a19b507595a43bf58262=1654683135,1654738616,1655863671,1656063540; Hm_lpvt_164fe01b1433a19b507595a43bf58262=1656329205',
                        'Host': 'www.jisilu.cn',
                        'If-Modified-Since': 'Mon, 27 Jun 2022 09:51:49 GMT',
                        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
                        'sec-ch-ua-mobile': '?0',
                        'sec-ch-ua-platform': "Windows",
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Upgrade-Insecure-Requests': '1',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        self.url = "https://www.jisilu.cn/data/cbnew/cb_list_new/?___jsl=LST___t=1656323342107"

    def GetData(self, priceLimit = 0, pricePercentLimit = 0, balanceLeft = 0):
        session = requests.Session()
        session.trust_env = False
        html = session.post(self.url,headers = self.headers)        #Get方式获取网页数据
        if html.status_code != 200:
            print("获取网络数据失败")
            return None
        #else:print("获取数据成功")
        j = json.loads(html.text)
        js = '['
        for one in j['rows']:
            if js != '[': js += ','
            js += json.dumps(one['cell'])
        js +=']'
        df = pd.read_json(js, orient='records')
        #print(df.shape)
        dfs = df[['bond_id',  'bond_nm','price', 'increase_rt','stock_nm','sprice', 'sincrease_rt', 'pb', 'convert_price','convert_value', 'premium_rt',  'dblow','year_left','curr_iss_amt','volume','turnover_rt','bond_nm_tip']]#,'adjust_condition', 'sw_cd', 'market_cd', 'btype', 'list_dt', 'qflag2','owned', 'hold', 'bond_value', 'rating_cd', 'option_value','put_convert_price', 'force_redeem_price', 'convert_amt_ratio','fund_rt', 'short_maturity_dt', 'year_left', 'curr_iss_amt', 'volume','svolume', 'turnover_rt', 'ytm_rt', 'put_ytm_rt', 'notes', 'noted','bond_nm_tip', 'redeem_icon', 'last_time', 'qstatus', 'margin_flg','sqflag', 'pb_flag', 'adj_cnt', 'adj_scnt', 'convert_price_valid','convert_price_tips', 'convert_cd_tip', 'ref_yield_info', 'adjusted','orig_iss_amt', 'price_tips', 'redeem_dt', 'real_force_redeem_price','option_tip', 'volatility_rate']]
        dfs.columns = ['转代码','转名称', '转现价', '转涨跌', '名称','正股价', '正股涨跌', '正股PB','转股价', '转股价值', '溢价率', '双低','剩余年限','剩余规模','成交额(万元)','转换手','公告']        # '债底溢价率','债券评级','期权价值','正股波动率','回售触发价','强赎触发价','转债占比','基金持仓','到期时间','剩余年限','剩余规模(亿元)','成交额(万元)','换手率','到期税前收益','回售收益','双低','操作']
        dfsmall = dfs.copy()
        dfsmall.loc[:, '转成交额'] = dfs['成交额(万元)'] / 10000
        dfsmall.loc[:, '成交额占比'] = round(dfs['成交额(万元)'] / (dfs['剩余规模'] * 10000),2)
        dflow = dfsmall.round(3).copy()

        if priceLimit > 0:
            dflow = dflow.loc[dfsmall['现价'] < priceLimit, :]  #现价低于130
        if pricePercentLimit > 0:
            dflow = dflow.loc[dflow['溢价率'] < pricePercentLimit,:]
        if balanceLeft > 0:
            dflow = dflow.loc[dflow['剩余规模'] < balanceLeft, :]
        df = dflow.sort_values(by= '成交额占比', ascending=False)

        return df[['转代码', '转名称', '转现价', '名称', '溢价率','转涨跌','剩余规模', '成交额占比']]  # .head(30)


class GetDataFromWeb:
    def __init__(self,page = 1, pagesize = 50):
        self._data = None
        self.page     = page
        self.pagesize = pagesize
        self.headers  = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'}
        self.kzz = GetKZZFromWeb()
    def urlNotion(self,page = 0,pagesize = 0):
        if page == 0:page = self.page
        if pagesize == 0:pagesize = self.pagesize
        return f'http://96.push2.eastmoney.com/api/qt/clist/get?cb=jQuery1124012058701159944296_1628131878807&pn={page}&pz={pagesize}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:90+t:3+f:!50&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152,f124,f107,f104,f105,f140,f141,f207,f208,f209,f222&_=1628131878808'
    def urlIndustry(self,page = 0,pagesize = 0):
        if page == 0:page = self.page
        if pagesize == 0:pagesize = self.pagesize
        return f'http://81.push2.eastmoney.com/api/qt/clist/get?cb=jQuery112407015799120771269_1628147949946&pn={page}&pz={pagesize}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:90+t:2+f:!50&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152,f124,f107,f104,f105,f140,f141,f207,f208,f209,f222&_=1628147949947'
    def urlBlockKlineMinute(self,secid):
        return f'http://push2.eastmoney.com/api/qt/stock/trends2/get?cb=jQuery1124017983593515879037_1628150220616&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6%2Cf7%2Cf8%2Cf9%2Cf10%2Cf11%2Cf12%2Cf13&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&iscr=0&ndays=1&_=1628150220633'
    def urlBlockKlineDailyLong(self,secid):
        return f'http://67.push2his.eastmoney.com/api/qt/stock/kline/get?cb=jQuery331007735612277817316_1628150235293&secid={secid}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&beg=0&end=20500101&smplmt=1398&lmt=1000000&_=1628150235297'
    def urlBlockKlineDailyShort(self,secid):
        return f'http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?cb=jQuery112302159331297180649_1628160598877&lmt=0&klt=101&fields1=f1%2Cf2%2Cf3%2Cf7&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61%2Cf62%2Cf63%2Cf64%2Cf65&ut=b2884a393a59ad64002292a3e90d46a5&secid={secid}&_=1628160598878'
    def urlBlockSheet(self,smallsecid):
        return f'http://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112307645366778240963_1628219069172&fid=f62&po=1&pz=50&pn=1&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5&fs=b%3A{smallsecid}&fields=f12%2Cf14%2Cf2%2Cf3%2Cf62%2Cf184%2Cf66%2Cf69%2Cf72%2Cf75%2Cf78%2Cf81%2Cf84%2Cf87%2Cf204%2Cf205%2Cf124%2Cf1%2Cf13'
    def urlChengJiaoLiangPaiHang(self,page=0,pagesize=0):
        if page == 0:page = self.page
        if pagesize == 0:pagesize = self.pagesize
        return f'http://10.push2.eastmoney.com/api/qt/clist/get?cb=jQuery112409082106482419372_1658111844541&pn={page}&pz={pagesize}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=|0|0|0|web&fid=f6&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1658111844573'
    def urlZhuLi(self,page=0,pagesize=0):
        if page == 0:page = self.page
        if pagesize == 0:pagesize = self.pagesize
        return f'https://push2.eastmoney.com/api/qt/clist/get?cb=jQuery112307336497834878335_1658456564556&fid=f62&po=1&pz={pagesize}&pn={page}&np=1&fltt=2&invt=2&ut=b2884a393a59ad64002292a3e90d46a5&fs=m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2&fields=f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124,f1,f13,f21,f4,f5,f6,f20,f7,f8,f10,f9,f23,f100,f109,f160'
    def getDataFromURL(self,url):    
        session = requests.Session()
        session.trust_env = False
        html,txt = session.get(url,headers = self.headers),''        #Get方式获取网页数据
        if html.status_code != 200:print("获取网络数据失败")
        else: txt = html.text
        return txt
    def getJsonFromURLTxt(self,txt):
        posStart,posEnd = txt.find('('),txt.rfind(')')
        if posStart < 0 or posEnd < 0:return ''
        return json.loads(txt[posStart+1:posEnd])['data']
    def getZhuLi(self,url):
        data = pd.json_normalize(self.getJsonFromURLTxt(self.getDataFromURL(url))['diff'])
        dfs =    data[['f12', 'f14',  'f2',   'f3',    'f62',      'f184',  'f21',    'f4',     'f5',    'f6',    'f20',  'f7',  'f8',   'f10', 'f9',    'f23','f100','f109','f160' ]]
        dfs.columns = ['代码','名称', '现价','涨跌幅','净流入','净占比','流通市值','涨跌额', '成交量','成交额','总市值','振幅','换手率','量比','市盈率','市净率','行业','5日', '10日']
        dfs = dfs.replace('-',0)
        dfsmall = dfs.copy()
        dfsmall['5日']  = dfsmall['5日']  - dfsmall['涨跌幅']
        dfsmall['10日'] = dfsmall['10日'] - dfsmall['涨跌幅']
        dfsmall.loc[:, '成交额']     = dfs['成交额'] / 100000000
        dfsmall.loc[:, '成交量']     = dfs['成交量'] / 10000
        dfsmall.loc[:, '总市值']     = dfs['总市值'] / 100000000
        dfsmall.loc[:, '流通市值']   = dfs['流通市值'] / 100000000
        dfsmall.loc[:, '净流入']     = dfs['净流入'] / 100000000
        dfsmall.loc[:, '占比']       = dfs['净流入'] / (dfs['流通市值']) * 100
        #去掉st,c,n
        dfsmall=dfsmall[~dfsmall['名称'].str.contains('N|C|ST|Ｂ|B')]
        dfsmall = dfsmall.dropna()
        dfsmall = dfsmall.round(2).copy()
        self._data = dfsmall.dropna().reset_index(drop=True)#           #重新设置Index，表中显示出行号   .dropna()
    def stand(self, factor):
        mean = factor.mean()
        std  = factor.std()
        return (factor - mean) / std
    def getChengJiaoLiangPaiHang(self,url):
        data = pd.json_normalize(self.getJsonFromURLTxt(self.getDataFromURL(url))['diff'])
        dfs =    data[['f12', 'f14',  'f2',    'f3',    'f4',   'f5',    'f6',   'f20',   'f21',      'f7',  'f8',  'f10', 'f9',   'f23',   'f62']]
        dfs.columns = ['代码','名称', '现价','涨跌幅','涨跌额', '成交量','成交额','总市值','流通市值','振幅','换手率','量比','市盈率','市净率','主力净流入']
        dfs = dfs.replace('-',0)
        dfsmall = dfs.copy()
        dfsmall.loc[:, '成交额']     = dfs['成交额'] / 100000000
        dfsmall.loc[:, '成交量']     = dfs['成交量'] / 10000
        dfsmall.loc[:, '总市值']     = dfs['总市值'] / 100000000
        dfsmall.loc[:, '流通市值']   = dfs['流通市值'] / 100000000
        dfsmall.loc[:, '主力净流入'] = dfs['主力净流入'] / 100000000
        dfsmall.loc[:, '成交额占比'] = dfs['成交额'] / (dfs['总市值']) * 100
        dflow = dfsmall.round(2).copy()
        self._data = dflow.sort_values(by= '成交额占比', ascending=False)


    def GetData(self, priceLimit = 0, pricePercentLimit = 0, balanceLeft = 0):
        self.getZhuLi(self.urlZhuLi(1,2000))
        self.kzzDf = self.kzz.GetData()
        self.stockDf = self._data[['代码','名称','行业','现价','涨跌幅','换手率','量比','净流入','成交额','占比']]#,'成交额'
        self._data = pd.merge(self.stockDf, self.kzzDf, how='inner', on='名称')
        self._data['转代码']=self._data['转代码'].astype(str)
        self._data['symbol'] = self._data['转代码'].apply(lambda X: 'sh' + X[:6] if X[:2] == "11" else 'sz' + X[:6])

        #=================================================排序=======================================
        self._data = self._data.sort_values(by = '占比', ascending=False).head(40)
        # ==============================================取技术指标=======================================
        momentdata =self.get_momentdata(self._data)
        self._data = pd.merge(self._data,momentdata,on='symbol')
        self._data = self._data.reindex()

        return self._data


    def get_momentdata(self, stockpool):
        stockpool = stockpool
        result = []
        for i in tqdm(stockpool['symbol']):
            try:
                df = ak.bond_zh_hs_cov_daily(symbol=i)
                df['symbol'] = i
                # 动量策略
                data_m5 = round(df['close'].rolling(5).apply(self.momentum_func).to_frame('momentum').reset_index(), 2)
                df['momentum_5'] = data_m5['momentum']
                # Rbreak突破策略
                df['pivot'] = round((df['high'].shift() + df['low'].shift() + df['close'].shift()) / 3, 2)  # '中枢点'
                df['bBreak'] = round(df['high'].shift() + 2 * (df['pivot'] - df['low'].shift()), 2)  # 突破买入价
                df['bEnter'] = round(2 * df['pivot'] - df['high'].shift(), 2)  # 反转买入价
                # DT趋势突破策略
                HH = max(df['high'][-4:-1])
                LC = min(df['close'][-4:-1])
                HC = max(df['close'][-4:-1])
                LL = min(df['low'][-4:-1])
                R = max(HH - LC, HC - LL)
                df['DualTrust'] = round(df['open'] + 0.7 * R, 2)
                result.append(df)
            except:
                # 如果出现异常，则跳过当前循环
                continue
        if result:
            df = pd.concat(result)
            df = df.drop_duplicates(subset=['symbol'], keep='last')
            return df[['symbol','momentum_5','bBreak','bEnter']]
        else:
            return None

    def momentum_func(self, the_array):
        r = np.log(the_array)
        slope, _, rvalue, _, _ = linregress(np.arange(len(r)), r)
        annualized = (1 + slope) ** 252
        return annualized * (rvalue ** 2)


    def GetDataChengJiaoLiangPaiHang(self, priceLimit = 0, pricePercentLimit = 0, balanceLeft = 0):
        self.getChengJiaoLiangPaiHang(self.urlChengJiaoLiangPaiHang(1,500))
        self.SetSelSecect()
        self._data = self._data.reindex()
        return self._data[['自选','代码','名称','现价','涨跌幅','换手率','量比','主力净流入','成交额','成交额占比']]


    def setSelfSelect(self, fileName):
        if not os.path.exists(fileName):
            #print(fileName,'文件，不存在')
            return None
        selSelect = pd.read_excel(fileName)
        if '名称' in selSelect.columns:
            return selSelect['名称'].tolist()
        return None

    def getPlates(self,url):#得到板块
        data = pd.io.json.json_normalize(self.getJsonFromURLTxt(self.getDataFromURL(url))['diff'])
        datasmall =    data[['f14',  'f2',    'f3',    'f4',    'f20',   'f8',    'f104',    'f105',   'f128',    'f140','f136',  'f13',   'f12']]
        datasmall.columns = ['板块', '最新价','涨跌幅','涨跌额','总市值','换手率','上涨家数','下跌家数','领涨股票','代码','涨跌幅','小代码','长代码']
        return datasmall
    def showMoneyInSheet(self,urlsheet):        
        data = pd.io.json.json_normalize(self.getJsonFromURLTxt(self.getDataFromURL(urlsheet))['diff'])
        datasmall =    data[['f12', 'f14', 'f2',    'f3',    'f62',       'f184',     'f66',         'f69',         'f72',       'f75']]
        datasmall.columns = ['代码','名称','最新价','涨跌幅','主力净流入','主力净占比','超大单净流入','超大单净占比','大单净流入','大单净占比']
        return datasmall
    def showKLine(self,urlKl):
        js = self.getJsonFromURLTxt(self.getDataFromURL(urlKl))['klines']
        #pdCol = ('time','open','close','high','low','val','money','a','b','c','d')
        cnt = len(js[0].split(','))
        dataA = np.array([]).reshape(0,cnt)
        for item in js:dataA = np.vstack((dataA,np.array(item.split(','))))
        data = pd.DataFrame(data = dataA,dtype = object)
        return data


class PdTable(QtCore.QAbstractTableModel):
    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data
        self.checks = {}
        self.background_colors = dict()

    def rowCount(self, parent=None):
        return self._data.shape[0]
    def columnCount(self, parent=None):
        return self._data.shape[1]
    # 显示行和列头
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        elif orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.axes[0][col]
        return None

    def checkState(self, index):
        if index in self.checks.keys():
            return self.checks[index]
        else:
            return QtCore.Qt.Unchecked

    def data(self, index, role=QtCore.Qt.DisplayRole):
        row = index.row()
        col = index.column()
        #显示复选框
        # if role == QtCore.Qt.CheckStateRole and col == 0:
        #     #self.checks[QtCore.QPersistentModelIndex(index)] = None
        #     return self.checkState(QtCore.QPersistentModelIndex(index))

        if role == QtCore.Qt.DisplayRole and col != 0:
            return str(self._data.iloc[index.row(), index.column()])#'{0}'.format(self._data[row][col])
        #显示自选颜色
        if   role == QtCore.Qt.BackgroundRole and col == 0 and self._data.iloc[index.row(), index.column()] == 1:
            return QtCore.QVariant(QtGui.QColor(QtCore.Qt.red))
        elif role == QtCore.Qt.BackgroundRole and col == 0 and self._data.iloc[index.row(), index.column()] == 2:
            return QtCore.QVariant(QtGui.QColor(QtCore.Qt.green))
        #显示专特颜色
        if   role == QtCore.Qt.BackgroundRole and col == 3 and self._data.iloc[index.row(), index.column()] == 1:
            return QtCore.QVariant(QtGui.QColor(QtCore.Qt.red))
        elif role == QtCore.Qt.BackgroundRole and col == 3 and self._data.iloc[index.row(), index.column()] == 2:
            return QtCore.QVariant(QtGui.QColor(QtCore.Qt.green))
        elif role == QtCore.Qt.BackgroundRole and col == 3 and self._data.iloc[index.row(), index.column()] == 3:
            return QtCore.QVariant(QtGui.QColor(QtCore.Qt.yellow))

        if role == QtCore.Qt.BackgroundRole and col == 0:
            ix = self.index(index.row(), 0)
            pix = QtCore.QPersistentModelIndex(ix)
            if pix in self.background_colors:
                color = self.background_colors[pix]
                return color
            elif role == QtCore.Qt.DisplayRole:
                return self.modelTableData[index.row()][index.column()]

        return None

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid():return False
        # if role == QtCore.Qt.CheckStateRole:
        #     self.checks[QtCore.QPersistentModelIndex(index)] = value
        #     return True
        # if role == QtCore.Qt.BackgroundRole and index.row()==self.target_row and self.color_enabled==True:
        #         return QtCore.QBrush(self.color_back)

        if ( 0 <= index.row() < self.rowCount() and 0 <= index.column() < self.columnCount()):
            if role == QtCore.Qt.BackgroundRole and index.isValid():
                ix = self.index(index.row(), 0)
                pix = QtCore.QPersistentModelIndex(ix)
                self.background_colors[pix] = value
                return True

        return False
    def flags(self, index):
        fl = QtCore.QAbstractTableModel.flags(self, index)
        if index.column() == 0:
            fl |= QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsUserCheckable
        return fl

    def sort(self, column, order):
        colname = self._data.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._data.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._data.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

class Thread(QtCore.QThread):
    #线程值信号
    valueChange = QtCore.pyqtSignal(int)
    #构造函数
    def __init__(self, window, parent=None):
        #super(Thread, self).__init__()
        super(Thread, self).__init__(parent)
        self.isPause = False
        self.isCancel=False
        self.cond = QtCore.QWaitCondition()
        self.mutex = QtCore.QMutex()   
        self.window = window
        self.data = GetDataFromWeb() 
    #暂停
    def pause(self):
        print("线程暂停")
        self.isPause = True
        
    #恢复
    def resume(self):
        print("线程恢复")
        self.isPause = False
        self.cond.wakeAll()
    #取消   
    def cancel(self):
        print("线程取消")
        self.isCancel=True
    
    def tradeTime(self):           #判断当前是否为交易时间
        nowTime = datetime.datetime.now()
        if nowTime.weekday() >= 5:return False  #周六，周天不工作
        nowMinutes = nowTime.hour * 60 + nowTime.minute
        if nowMinutes < 9  * 60 + 30:return False
        if nowMinutes > 11 * 60 + 30 and nowMinutes < 13 * 60:return False
        if nowMinutes > 15 * 60:return False
        return True

    def timerEvent(self):                  #定时器事件
        nowTime = datetime.datetime.now()
        if not self.tradeTime():
            #self.window.setWindowTitle("当前非交易时间" + str(nowTime))
            return
        #self.window.setWindowTitle("股票" + str(nowTime))
        self.updateData()

    def updateData(self):
        self.window.model._data = self.data.GetData()#double(self.window.editPrice.text()),double(self.window.editPricePercent.text()),double(self.window.editLeft.text()))
        self.window.view.viewport().update()#同步数据
    #运行(入口)
    def run(self):
        #for i in range(100):
        while True:
            #线程锁on
            self.mutex.lock()
            if self.isPause:
                self.cond.wait(self.mutex)
            if self.isCancel:
                self.valueChange.emit(0)
                break
            #业务代码
            #self.valueChange.emit(i)
            #self.msleep(100)
            self.timerEvent()
            self.msleep(2000)
            #线程锁off
            self.mutex.unlock()

class MyWindow(QtWidgets.QMainWindow): #主窗体类
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self._allHwnds = []
        # self.windowTrade = ControlHuaTai()
        self.firstStart = True
        self.setWindowTitle("股票")
        self.resize(2160, 1440)
        self.createMenu()
        self.createControlFace()
        self.createTableView()
        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addWidget(self.widgetControls)
        self.vbox.addWidget(self.view)
        #self.vbox.addWidget(self.widgetTip)
        self.widget=QtWidgets.QWidget()
        self.widget.setLayout(self.vbox)
        self.setCentralWidget(self.widget)


        #启动定时器
        self.timer_id = self.startTimer(1000, timerType = QtCore.Qt.VeryCoarseTimer)
        #self.thread = Worker(self)
        self.thread = Thread(self)
        self.thread.start()

    def createControlFace(self):  #创建按钮和输入框
        self.widgetControls = QtWidgets.QWidget()
        self.hHViewBox = QtWidgets.QHBoxLayout()
        self.edit = QtWidgets.QLineEdit()
        self.hHViewBox.addWidget(self.edit)

        self.labelPrice = QtWidgets.QLabel(text='每股买入金额')
        self.hHViewBox.addWidget(self.labelPrice)
        self.editMoney = QtWidgets.QLineEdit(text='30000')
        self.hHViewBox.addWidget(self.editMoney)
        self.widgetControls.setLayout(self.hHViewBox)
        self.widgetControls.height = 30



    def createTableView(self):              #创建表格
        self.GetData = GetDataFromWeb()
        df = self.GetData.GetData()#double(self.editPrice.text()), double(self.editPricePercent.text()),double(self.editLeft.text()))
        self.model = PdTable(df)
        self.view = QtWidgets.QTableView()
        self.view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)#所有列自动拉伸，充满界面
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)#设置只能选中一行
        self.view.setEditTriggers(QtWidgets.QTableView.NoEditTriggers)         #不可编辑
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows);#设置只有行选中
        self.view.verticalHeader().show()
        #self.view.setColumnWidth(0, 5)
        self.view.setModel(self.model)
        #self.view.setSortingEnabled(True)  #设置可以排序

    def createMenu(self):                       #创建菜单
        #添加menu菜单栏,注意：QMainWindow 才可以有菜单栏，QWidget没有，因此上面只能采用继承QMainWIndow
        tool = self.addToolBar("File") #这里尝试使用QmenuBar，则此时会卡死，无法完成下面appedRow操作（猜测：可能是因为本身不允许menuBar完成这种操作）
        # self.action= QtWidgets.QAction("添加", self)
        # self.action2=QtWidgets.QAction("删除", self)
        self.actionBuy = QtWidgets.QAction('买入', self)
        self.actionSell = QtWidgets.QAction('卖出', self)
        self.actionBuy5  = QtWidgets.QAction('买入前5', self)
        self.actionBuy10 = QtWidgets.QAction('买入前10', self)
        self.actionBuy15 = QtWidgets.QAction('买入前15', self)
        self.actionBuy20 = QtWidgets.QAction('买入前20', self)
        self.actionBuy30 = QtWidgets.QAction('买入前30', self)
        self.actionBuyUpdate = QtWidgets.QAction('刷新买入框', self)
        self.actionUpdate = QtWidgets.QAction('手动刷新',self)
        self.actionSaveElsx = QtWidgets.QAction('保存文件',self)
        # tool.addAction(self.action)
        # tool.addAction(self.action2)
        tool.addAction(self.actionBuy)
        tool.addAction(self.actionSell)
        tool.addAction(self.actionBuy5)
        tool.addAction(self.actionBuy10)
        tool.addAction(self.actionBuy15)
        tool.addAction(self.actionBuy20)
        tool.addAction(self.actionBuy30)
        tool.addAction(self.actionBuyUpdate)
        tool.addAction(self.actionUpdate)
        tool.addAction(self.actionSaveElsx)
        tool.actionTriggered[QtWidgets.QAction].connect(self.processtrigger)

    def tradeTime(self):           #判断当前是否为交易时间
        nowTime = datetime.datetime.now()
        if nowTime.weekday() >= 5:return False  #周六，周天不工作
        nowMinutes = nowTime.hour * 60 + nowTime.minute
        if nowMinutes < 9  * 60 + 15:return False
        if nowMinutes > 11 * 60 + 30 and nowMinutes < 13 * 60:return False
        if nowMinutes > 15 * 60:return False
        return True

    def timerEvent(self, event):                  #定时器事件
        nowTime = datetime.datetime.now()
        global sortAsHeader
        if not self.tradeTime():
            self.setWindowTitle("当前非交易时间   排序方式："+sortAsHeader + "   " + str(nowTime))
            return
        self.setWindowTitle("股票   排序方式："+sortAsHeader + "   "  + str(nowTime))


    def processtrigger(self,action):
        self.thread.pause()                            #挂起获取数据线程
        # if action.text()=="添加":print('添加')
        # elif action.text()=="删除":print('删除')
        if action.text()=='买入':self.BuyCurrent()
        elif action.text()=='卖出':print('卖出')
        elif action.text()=='买入前5': self.BuyHeadCnt(5)
        elif action.text()=='买入前10':self.BuyHeadCnt(10)
        elif action.text()=='买入前15':self.BuyHeadCnt(15)
        elif action.text()=='买入前20':self.BuyHeadCnt(20)
        elif action.text()=='买入前30':self.BuyHeadCnt(30)
        elif action.text()=='刷新买入框':self.BuyHeadCnt(30,False)
        elif action.text()=='手动刷新':self.thread.updateData()
        elif action.text()=='保存文件':
            self.GetData.saveDfToExcel()
            self.GetData.saveHead30ToTradeFlord()
        self.thread.resume()                           #恢复获取数据线程


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    #app.setStyleSheet(load_stylesheet_pyqt5())
    window = MyWindow()
    print('窗口创建')
    window.show()
    #window.thread.start()
    print('窗口显示')
    sys.exit(app.exec_())