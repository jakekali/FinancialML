import pandas as pd
from abc import abstractmethod

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
                self.rets_values.append(ret_vale)
            if ret_vale > epsilon:
                self.bar_data.loc[i, 'label'] = 1
            elif ret_vale < -epsilon:
                self.bar_data.loc[i, 'label'] = -1
            else:
                self.bar_data.loc[i, 'label'] = 0            

        self.bar_data = self.bar_data.dropna()

        if not inplace:
            return self.bar_data
        

class tripleBarrierLabeler(labeler):
    '''
    This class is used to label bar data based on the triple barrier method.
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

    



