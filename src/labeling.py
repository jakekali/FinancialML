import pandas as pd
from abc import abstractmethod
import multiprocessing as mp
import datetime as dt
import tqdm


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

    def getDailyVolatility(self, close_data=None, span0=100):
        '''
        Compute the daily volatility of a stock.
        :param close_data: pandas.DataFrame, the close price of the stock
        :param span0: int, the span of the rolling window

        :return: pandas.DataFrame, the daily volatility of the stock

        From Page 44 of Advances in Financial Machine Learning by Marcos Lopez de Prado
        '''
        if close_data is None:
            close_data = self.bar_data['close_price']

        df0 = close_data.index.searchsorted(close_data.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        df0 = pd.Series(close_data.index[df0 - 1], index=close_data.index[close_data.shape[0] - df0.shape[0]:])
        df0 = close_data.loc[df0.index] / close_data.loc[df0.values].values - 1
        df0 = df0.ewm(span=span0).std()
        return df0
    
    def label(self, upFactor=1, downFactor=1, horizon=60*7, binary=True):
        ''' 
        Label the bar data.
        :param upFactor: float, the factor to multiply the daily volatility by to get the upper barrier
        :param downFactor: float, the factor to multiply the daily volatility by to get the lower barrier
        :param horizon: int, the number of bars to look forward for a vertical barrier
        :param binary: bool, whether to label the data as 1, 0, or -1 or 1, {return}, -1
        :return: pandas.DataFrame, the labeled bar data
        '''

        if not isinstance(upFactor, float):
            raise TypeError('upFactor must be a float')
        
        if not isinstance(downFactor, float):
            raise TypeError('downFactor must be a float')
        
        if not isinstance(horizon, int):
            raise TypeError('horizon must be an integer')
        
        if not isinstance(binary, bool):
            raise TypeError('binary must be a boolean')
        
        if horizon < 1:
            raise ValueError('horizon must be greater than 0')
        
        if horizon > len(self.bar_data):
            raise ValueError('horizon must be less than or equal to the number of bars')
        
        if upFactor < 0:
            raise ValueError('upFactor must be greater than or equal to 0')
        
        if downFactor < 0:
            raise ValueError('downFactor must be greater than or equal to 0')
        
        if not isinstance(self.bar_data, pd.DataFrame):
            raise TypeError('bar_data must be a pandas.DataFrame')
        
        # check if the data is already labeled
        if 'Volatility' in self.bar_data.columns:
            self.bar_data.drop(columns=['Volatility', 'UpperBarrier', 'LowerBarrier'], inplace=True)
            


        self.bar_data.insert(2, "Volatility", self.getDailyVolatility(), True)
        self.bar_data.dropna(inplace=True)
        self.bar_data.insert(3, "UpperBarrier", self.bar_data['close_price'] * (1 + self.bar_data['Volatility'] * upFactor), True)
        self.bar_data.insert(4, "LowerBarrier", self.bar_data['close_price'] * (1 - self.bar_data['Volatility'] * downFactor), True)

        self.bar_data.reset_index(inplace=True, drop=True)

        for i in tqdm.tqdm(range(len(self.bar_data))):
            # Loop through variables in the horizon, and see which one is hit first

            stopped = False

            for j in range(i + 1, i + horizon):

                # If we reach the end of the data, then we can't label this data point
                if j >= len(self.bar_data):
                    break

                if self.bar_data.loc[j, 'close_price'] >= self.bar_data.loc[i, 'UpperBarrier']:
                    self.bar_data.loc[i, 'label'] = 1
                    stopped = True
                    break

                elif self.bar_data.loc[j, 'close_price'] <= self.bar_data.loc[i, 'LowerBarrier']:
                    self.bar_data.loc[i, 'label'] = -1
                    stopped = True
                    break

            if not stopped:
                if binary:
                    self.bar_data.loc[i, 'label'] = 0
                else:
                    self.bar_data.loc[i, 'label'] = self.bar_data.loc[i, 'close_price'] / self.bar_data.loc[i, 'open_price'] - 1

        self.bar_data = self.bar_data.dropna()
        return self.bar_data
            






        return self.bar_data