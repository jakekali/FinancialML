import bars
import ticksDB
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np

class labeler:
    '''
    This an abstract class is used to label bar data.
    '''

    def __init__(self, bar_data):
        '''
        :param bar_data: pandas.DataFrame, the bar data to be labeled
        '''
        self.bar_data = bar_data
        self.labels = None

    @abstractmethod
    def label(self):
        '''
        Label the bar data.
        '''
        raise NotImplementedError
    

class barHorizionLabler(labeler):
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

    def label(self, eplison=0.1, inplace=True):
        '''
        Label the bar data.
        '''
        if not inplace:
            self.bar_data = self.bar_data.copy()

        for i in range(len(self.bar_data) - self.bar_horizon):
            ret_vale = (self.bar_data.loc[i + self.bar_horizon, 'close'] / self.bar_data.loc[i, 'close']) - 1
            if ret_vale > eplison:
                self.bar_data.loc[i, 'label'] = 1
            elif ret_vale < -eplison:
                self.bar_data.loc[i, 'label'] = -1
            else:
                self.bar_data.loc[i, 'label'] = 0            

        self.bar_data = self.bar_data.dropna()

        if not inplace:
            return self.bar_data


