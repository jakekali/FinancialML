import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime, text, Float, ForeignKey, select
from sqlalchemy.orm import declarative_base, relationship, backref, sessionmaker, mapped_column
from datetime import datetime

Base = declarative_base()

class ticksDB:
    '''
    This class is used to connect to the database and add tick data and securities to the database.
    '''
    def __init__(self, engine=None):
        '''
        If an engine is not provided, a default engine will be created.
        :param engine: sqlalchemy.engine.base.Engine to connect to the database
        '''
        self.engine = engine

    def connect(self):
        '''
        Connect to the database using the engine provided in the constructor, and establish a session.
        '''
        if self.engine is None:
            self.engine = create_engine('postgresql+psycopg2://postgres:password@localhost/FML')
        self.conn = self.engine.connect()
        Base.metadata.create_all(c.engine)

        Session = sessionmaker(c.conn)  
        self.session = Session()

    def disconnect(self):
        '''
        Close the connection to the database, and dispose of the session.
        '''
        self.conn.close()
        self.engine.dispose()

    def add_tickData_from_dataframe(self, ticker, dataframe, description=None, chunksize=1000):
        '''
        Add tick data to the database from a pandas dataframe.
        :param ticker: str, the ticker of the security
        :param dataframe: pandas.DataFrame, the dataframe containing the tick data. 
        The dataframe should contain the following columns: ttime, bid_price, ask_price, bid_volume, ask_volume
        :param description: str, the description of the security
        :param chunksize: int > 0, the number of rows to add to the database at a time
        '''

        # check if ticker exists
        if not self.tickerExists(ticker):
            # add ticker to securities table
            new_ticker = securities(ticker=ticker, name=description)
            self.session.add(new_ticker)

        self.session.commit()
            
        # get ticker id
        ticker_id = self.get_ticker_id(ticker)

        #ensure that the dataframe has the correct columns
        dataframe['ids'] = int(ticker_id.id)
        dataframe['ttime'] = pd.to_datetime(dataframe['ttime'], format='%Y%m%d %H:%M:%S:%f')
        dataframe = dataframe[['ids', 'ttime', 'bid_price', 'ask_price', 'bid_volume', 'ask_volume']]

        # add tick data to ticks table
        input_df = dataframe
        input_df.to_sql('ticks', self.engine, if_exists='append', index=False, chunksize=chunksize, method='multi')
        
    def tickerExists(self, ticker):
        '''
        Check if a ticker exists in the database.
        :param ticker: str, the ticker to check
        :return: bool, True if the ticker exists, False otherwise
        '''
        stmt = select(securities).where(securities.ticker == ticker)
        return self.session.execute(stmt).fetchone() is not None

    def get_ticker_id(self, ticker):
        '''
        Get the id of a ticker.
        :param ticker: str, the ticker to get the id of
        :return: int, the id of the ticker
        '''
        stmt = select(securities).where(securities.ticker == ticker)
        tick_id = self.session.execute(stmt).fetchone()[0]
        return tick_id.id

class securities(Base):
    '''
    This class is used to represent the securities table in the database, and uses the sqlalchemy ORM.
    '''
    __tablename__ = 'securities'
    id = mapped_column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, unique=True)
    name = Column(String)
    
class ticks(Base):
    '''
    This class is used to represent the ticks table in the database, and uses the sqlalchemy ORM.
    '''
    __tablename__ = 'ticks'
    id = Column(Integer, primary_key=True)
    ids = mapped_column(ForeignKey('securities.id'))
    ttime = Column(DateTime, nullable=False, index=True)
    bid_price = Column(Float, nullable=False)
    ask_price = Column(Float, nullable=False)
    bid_volume = Column(Float, nullable=False)
    ask_volume = Column(Float, nullable=False)

c = ticksDB()
c.connect()


# new_ticker = securities(ticker='USA500.IDX', name='S&P 500')
# c.session.add(new_ticker)
# c.session.commit()
# sp_data = pd.read_csv('/home/jacob/code/FinancialML/data-tickStory/USA500IDXUSD.csv')
# sp_data.columns = ['ttime', 'bid_price', 'ask_price', 'bid_volume', 'ask_volume', 'null', 'null2']

# df = pd.DataFrame({'ttime': [datetime.now()], 'bid_price': [100], 'ask_price': [101], 'bid_volume': [1000], 'ask_volume': [1001]})
# c.add_tickData_from_dataframe('USA500.IDX', sp_data, 'S&P 500')

