import pandas as pd
from abc import abstractmethod
import multiprocessing as mp
import datetime as dt

def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    import pandas as pd
    #if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    #else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)

    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out:df0=df0.append(i)
    df0=df0.sort_index()
    return df0

def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out


class labeler:
    '''
    This an abstract class is used to label bar data.
    '''

    def __init__(self, bar_data):
        '''
        :param bar_data: pandas.DataFrame, the bar data to be labeled - test change
        '''
        self.bar_data = bar_data
        self.labels = None

    @abstractmethod
    def label(self):
        '''
        Label the bar data.
        '''
        raise NotImplementedError
    

class barHorizonLabeler(labeler):
    '''
    This class is used to label bar data based on the bar horizon.
    '''

    def __init__(self, bar_data : pd.DataFrame, bar_horizon):
        '''
        :param bar_data: pandas.DataFrame, the bar data to be labeled
        :param bar_horizon: int, the bar horizon (in number of bars)
        '''

        super().__init__(bar_data)
        if bar_horizon < 1:
            raise ValueError('bar_horizon must be greater than 0')
        
        if bar_horizon > len(bar_data):
            raise ValueError('bar_horizon must be less than or equal to the number of bars')
        
        if not isinstance(bar_horizon, int):
            raise TypeError('bar_horizon must be an integer')
        
        if not isinstance(bar_data, pd.DataFrame):
            raise TypeError('bar_data must be a pandas.DataFrame')
        
        self.bar_horizon = bar_horizon

    def label(self, epsilon=0.1, inplace=True, debug=False):
        '''
        Label the bar data.
        '''
        if not inplace:
            self.bar_data = self.bar_data.copy()

        if(debug):
            self.rets_values = []

        for i in range(len(self.bar_data) - self.bar_horizon):
            ret_vale = (self.bar_data.loc[i + self.bar_horizon, 'close_price'] / self.bar_data.loc[i, 'close_price']) - 1
            if(debug):
                self.up = 0
                self.down = 0
                self.rets_values.append(ret_vale)
            if ret_vale > epsilon:
                if debug:
                    self.up += 1
                self.bar_data.loc[i, 'label'] = 1
            elif ret_vale < -epsilon:
                self.bar_data.loc[i, 'label'] = -1
                if debug:
                    self.down += 1
            else:
                self.bar_data.loc[i, 'label'] = 0            

        self.bar_data = self.bar_data.dropna()

        if not inplace:
            return self.bar_data
        

class tripleBarrierLabeler(labeler):
    '''
    This class is used to label bar data based on the triple barrier method.

    Much of this is from the book Advances in Financial Machine Learning by Marcos Lopez de Prado
    '''

    def __init__(self, bar_data : pd.DataFrame):
        '''
        :param bar_data: pandas.DataFrame, the bar data to be labeled
        '''

        super().__init__(bar_data)

    def getDailyVolatility(close_data, span0=100):
        '''
        Compute the daily volatility of a stock.
        :param close_data: pandas.DataFrame, the close price of the stock
        :param span0: int, the span of the rolling window

        :return: pandas.DataFrame, the daily volatility of the stock

        From Page 44 of Advances in Financial Machine Learning by Marcos Lopez de Prado
        '''
        df0 = close_data.index.searchsorted(close_data.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        df0 = pd.Series(close_data.index[df0 - 1], index=close_data.index[close_data.shape[0] - df0.shape[0]:])
        df0 = close_data.loc[df0.index] / close_data.loc[df0.values].values - 1
        df0 = df0.ewm(span=span0).std()
        return df0
    

    def applyPtSlOnT1(close_data, events, ptSl, molecule):
        '''
        Apply stop loss/profit taking, if it takes place before t1 (end of event)
        :param close_data: pandas.DataFrame, the close price of the stock
        :param events: pandas.DataFrame, the events
        :param ptSl: list, the profit taking and stop loss limits
        :param molecule: list, the list of indices of the events

        :return: pandas.DataFrame, the events with the stop loss and profit taking limits

        From Page 44 of Advances in Financial Machine Learning
        '''


        events_ = events.loc[molecule]
        out = events_[['t1']].copy(deep=True)
        if ptSl[0] > 0:
            pt = ptSl[0] * events_['trgt']
        else:
            pt = pd.Series(index=events.index)

        if ptSl[1] > 0:
            sl = -ptSl * events_['trgt']
        else:
            sl = pd.Series(index=events.index)

        for loc, t1 in events_['t1'].fillna(close_data.index[-1]).iteritems():
            df0 = close_data[loc:t1]
            df0 = (df0 / close_data[loc] - 1) * events_.at[loc, 'side']
            out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
            out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()

        return out


    def getEvents(self, close_data, tEvents, ptS1, trgt, minRet, numThreads, t1=False):
        #) get target
        trgt = trgt.loc[tEvents]
        trgt = trgt[trgt > minRet]
        #2) get t1 (max holding period)
        if t1 is False:
            t1 = pd.Series(pd.NaT, index=tEvents)
        #3) form events object, apply stop loss on t1
        side_ = pd.Series(1., index=trgt.index)
        events = (pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt']))
        df0=mpPandasObj(func=self.applyPtSlOnT1,pdObj=('molecule',events.index),
                        numThreads=numThreads,close=close_data,events=events,
                        ptSl=ptS1)
        events['t1']=df0.dropna(how='all').min(axis=1) #pd.min ignores nan
        events=events.drop('side',axis=1)
        return events


   

    # multiprocessing snippet [20.7]


    def addVerticalBarrier(self, tEvents, close, numDays=1):
        t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
        t1=t1[t1<close.shape[0]]
        t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
        return t1
    



