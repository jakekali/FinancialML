import pandas as pd
from abc import abstractmethod
from .ticksDB import ticks
from sqlalchemy import text, select, and_

class bars:
    '''
    Abstract base class for bars.
    '''
    def __init__(self, ticker, start_date, end_date, ticksDB):
        '''
        :param ticker: str, the ticker of the security
        :param start_date: datetime, the start date of the bars
        :param end_date: datetime, the end date of the bars
        :param ticksDB: ticksDB, the ticksDB object to connect to the database
        '''

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.ticksDB = ticksDB



    @abstractmethod
    def get_bars(self):
        '''
        Get the bars.
        :return: pandas.DataFrame, the bars
        '''
        pass

class timeBars(bars):
    '''
    Class for time bars.
    '''
    def __init__(self, ticker, start_date, end_date, ticksDB, interval):
        '''
        :param ticker: str, the ticker of the security
        :param start_date: datetime, the start date of the bars
        :param end_date: datetime, the end date of the bars
        :param ticksDB: ticksDB, the ticksDB object to connect to the database
        :param interval str, must be 'minute', 'hour', 'day', 'month', 'year'
        '''

        super().__init__(ticker, start_date, end_date, ticksDB)
        self.tick_id = self.ticksDB.get_ticker_id(ticker)
        if interval not in ['minute', 'hour', 'day', 'month', 'year']:
            raise ValueError('interval must be \'minute\', \'hour\', \'day\', \'month\', \'year\'')
        self.interval = interval

    def get_ask_bars(self):
        '''
        Get the bars.
        :return: pandas.DataFrame, the bars
        '''
        stmt = (text(f''' 
        with datas as 
            (
            select 
                ttime,
                ask_price,
                ask_volume,
                FIRST_VALUE(ask_price) OVER (PARTITION BY date_trunc('{self.interval}', ttime)  ORDER BY ttime) AS first_ask_price,
                LAST_VALUE(ask_price) OVER (PARTITION BY date_trunc('{self.interval}', ttime) ORDER BY ttime) AS last_ask_price
            from ticks
            where ids = {self.tick_id} and ask_price > 0 and ttime>='{self.start_date}' and ttime<='{self.end_date}'
            )
        select 
            date_trunc('{self.interval}', ttime) AS minu, 
            MAX(ask_price) AS max_ask_price, 
            MIN(ask_price) AS min_ask_price, 
            SUM(ask_volume) AS total_ask_volume, 
            avg(datas.first_ask_price) as open_price,
            avg(last_ask_price) as close_price
        from datas
        group by 1
       	order by minu asc;
        '''))

        result = pd.read_sql(stmt, self.ticksDB.conn)

        return result

    def get_bid_bars(self):
        '''
        Get the bars.
        :return: pandas.DataFrame, the bars
        '''
        stmt = (text(f''' 
        with datas as 
            (
            select 
                ttime,
                bid_price,
                bid_volume,
                FIRST_VALUE(bid_price) OVER (PARTITION BY date_trunc('{self.interval}', ttime)  ORDER BY ttime) AS first_bid_price,
                LAST_VALUE(bid_price) OVER (PARTITION BY date_trunc('{self.interval}', ttime) ORDER BY ttime) AS last_bid_price
            from ticks
            where ids = {self.tick_id} and bid_price > 0 and ttime>='{self.start_date}' and ttime<='{self.end_date}'
            )
        select 
            date_trunc('{self.interval}', ttime) AS minu, 
            MAX(bid_price) AS max_bid_price, 
            MIN(bid_price) AS min_bid_price, 
            SUM(bid_volume) AS total_bid_volume, 
            avg(datas.first_bid_price) as open_price,
            avg(last_bid_price) as close_price
        from datas
        group by 1
       	order by minu asc;
        '''))

        result = pd.read_sql(stmt, self.ticksDB.conn)

        return result


class dynamicBars(bars):
    '''
    Abstract base class for 'dynamic' bars.
    Dynamic bars are bars that are not fixed in time, but are defined by a certain condition(s),
    e.g. the bars are defined by the number of ticks or the number of volume.

    To create these bars, we need implement using a 'rolling' window, locally, in python.
    '''
    window_size = None

    def __init__(self, ticker, start_date, end_date, ticksDB, window_size):
        '''
        :param ticker: str, the ticker of the security
        :param start_date: datetime, the start date of the bars
        :param end_date: datetime, the end date of the bars
        :param ticksDB: ticksDB, the ticksDB object to connect to the database
        :param window_size: int, the window size
        '''
        super().__init__(ticker, start_date, end_date, ticksDB)
        if window_size < 1:
            raise ValueError('window_size must be >= 1')
        self.window_size = window_size

    def get_ticks(self, ask=True):
        '''
        Yields ticks in batches of size window_size.
        :param ask: bool, if True, get ask ticks, else get bid ticks
        :return: pandas.DataFrame, the ask ticks
        '''
        self.tick_id = self.ticksDB.get_ticker_id(self.ticker)

        currentOffSet = 0
        while True:
            if ask:
                query = select(ticks.ttime, ticks.ask_price, ticks.ask_volume, ticks)
            else:
                query = select(ticks.ttime, ticks.bid_price, ticks.bid_volume, ticks)
            
            query = query.where(and_(ticks.ids == self.tick_id, ticks.ttime >= self.start_date, ticks.ttime <= self.end_date))
            query = query.order_by(ticks.ttime)
            query = query.limit(self.window_size)
            query = query.offset(currentOffSet)
            result = self.ticksDB.session.query(ticks).from_statement(query).all()
           
            currentOffSet += self.window_size
            if len(result) == 0:
                break

            yield result

class volumeBar(dynamicBars):
    '''
    Volume bars are bars that are defined by the amount of volume and a particular cutoff.
    '''
    cutoff = 10000

    def __init__(self, ticker, start_date, end_date, ticksDB, window_size, cutoff):
        '''
        :param ticker: str, the ticker of the security
        :param start_date: datetime, the start date of the bars
        :param end_date: datetime, the end date of the bars
        :param ticksDB: ticksDB, the ticksDB object to connect to the database
        :param window_size: int, the window size
        :param cutoff: int, the cutoff for the volume
        '''
        super().__init__(ticker, start_date, end_date, ticksDB, window_size)
        if cutoff < 1:
            raise ValueError('cutoff must be >= 1')
        self.cutoff = cutoff


    def get_bars(self, ask=True):
        '''
        Get the bars.
        :return: pandas.DataFrame, the bars
        '''
        
        current_volume = 0
        current_open = 0

        bar = {'time': None, 'open': None, 'high': None, 'low': None, 'close': None, 'volume': None}
        bars = []

        prices = []

        return_pd = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        for ticks in self.get_ticks(ask=ask):
            if len(ticks) == 0:
                break
            
            for(tick) in ticks:
                if current_volume == 0:
                    if ask:
                        current_open = tick.ask_price
                    else:
                        current_open = tick.bid_price

                if current_volume < self.cutoff:
                    if ask:
                        current_volume += tick.ask_volume
                    else:
                        current_volume += tick.bid_volume

                if ask:
                    prices.append(tick.ask_price)  
                else:
                    prices.append(tick.bid_price)

                if current_volume >= self.cutoff:
                    if ask:
                        bar['time'] = tick.ttime
                        bar['open'] = current_open
                        bar['high'] = max(prices)
                        bar['low'] = min(prices)

                        bar['close'] = ticks[-1].ask_price
                        bar['volume'] = current_volume

                    else:  
                        bar['time'] = tick.ttime
                        bar['open'] = current_open
                        bar['high'] = max(prices)
                        bar['low'] = min(prices)

                        bar['close'] = ticks[-1].bid_price
                        bar['volume'] = current_volume

                    current_volume = 0
                    current_open = 0
                    prices = []
                    bars.append(bar.copy())
                
        return_pd = pd.concat([return_pd, pd.DataFrame(bars)], ignore_index=True)
        return return_pd


class dollarBar(dynamicBars):
    '''
    Volume bars are bars that are defined by the amount of volume and a particular cutoff.
    '''
    cutoff = 10000

    def __init__(self, ticker, start_date, end_date, ticksDB, window_size, cutoff):
        '''
        :param ticker: str, the ticker of the security
        :param start_date: datetime, the start date of the bars
        :param end_date: datetime, the end date of the bars
        :param ticksDB: ticksDB, the ticksDB object to connect to the database
        :param window_size: int, the window size
        :param cutoff: int, the cutoff for the volume * price
        '''
        super().__init__(ticker, start_date, end_date, ticksDB, window_size)
        if cutoff < 1:
            raise ValueError('cutoff must be >= 1')
        self.cutoff = cutoff


    def get_bars(self, ask=True):
        '''
        Get the bars.
        :return: pandas.DataFrame, the bars
        '''
        
        current_volume_price = 0
        current_open = 0

        bars = []
        bar = {'time': None, 'open': None, 'high': None, 'low': None, 'close': None, 'volume': None}

        prices = []

        return_pd = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        for ticks in self.get_ticks(ask=ask):
            if len(ticks) == 0:
                break  
            
            for(tick) in ticks:

                if current_volume_price == 0:
                    if ask:
                        current_open = tick.ask_price
                    else:
                        current_open = tick.bid_price


                if current_volume_price < self.cutoff:
                    if ask:
                        current_volume_price += tick.ask_volume * tick.ask_price
                    else:
                        current_volume_price += tick.bid_volume * tick.bid_price

                if ask:
                    prices.append(tick.ask_price)  
                else:
                    prices.append(tick.bid_price)

                if current_volume_price >= self.cutoff:
                    if ask:
                        bar['time'] = tick.ttime
                        bar['open'] = current_open
                        bar['high'] = max(prices)

                        bar['low'] = min(prices)
                        bar['close'] = ticks[-1].ask_price
                        bar['volume'] = current_volume_price

                    else:  
                        bar['time'] = tick.ttime
                        bar['open'] = current_open
                        bar['high'] = max(prices)
                        bar['low'] = min(prices)
                        bar['close'] = ticks[-1].bid_price
                        bar['volume'] = current_volume_price

                    current_volume_price = 0
                    current_open = 0
                    prices = []
                    bars.append(bar.copy())
                    bar = {'time': None, 'open': None, 'high': None, 'low': None, 'close': None, 'volume': None}

                
        return_pd = pd.concat([return_pd, pd.DataFrame(bars)], ignore_index=True)
        return return_pd




class tickBars(dynamicBars):
    '''
    Tick bars are bars that are defined by the number of ticks and a particular cutoff.
    '''
    cutoff = 10000

    def __init__(self, ticker, start_date, end_date, ticksDB, cutoff):
        '''
        :param ticker: str, the ticker of the security
        :param start_date: datetime, the start date of the bars
        :param end_date: datetime, the end date of the bars
        :param ticksDB: ticksDB, the ticksDB object to connect to the database
        :param window_size: int, the window size
        :param cutoff: int, the cutoff for the number of ticks
        '''
        if cutoff < 1:
            raise ValueError('cutoff must be >= 1')
        self.cutoff = cutoff
        super().__init__(ticker, start_date, end_date, ticksDB, cutoff)


    def get_bars(self, ask=True):
        """
        Get the bars.
        :return: pandas.DataFrame, the bars
        """

        bars = []
        bar = {'time': None, 'open': None, 'high': None, 'low': None, 'close': None, 'volume': None}

        for ticks in self.get_ticks(ask=ask):
            if len(ticks) == 0:
                break
            else:
                if ask:
                    bar['time'] = ticks[-1].ttime
                    bar['open'] = ticks[0].ask_price
                    bar['high'] = max([tick.ask_price for tick in ticks])
                    bar['low'] = min([tick.ask_price for tick in ticks])
                    bar['close'] = ticks[-1].ask_price
                    bar['volume'] = sum([tick.ask_volume for tick in ticks])
                else:
                    bar['time'] = ticks[-1].ttime
                    bar['open'] = ticks[0].bid_price
                    bar['high'] = max([tick.bid_price for tick in ticks])
                    bar['low'] = min([tick.bid_price for tick in ticks])
                    bar['close'] = ticks[-1].bid_price
                    bar['volume'] = sum([tick.bid_volume for tick in ticks])

                bars.append(bar.copy())
                bar = {'time': None, 'open': None, 'high': None, 'low': None, 'close': None, 'volume': None}

        return pd.DataFrame(bars)
