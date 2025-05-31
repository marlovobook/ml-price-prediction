import pandas as pd
class DataProcessor:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
        import vectorbt as vbt
        price_data = vbt.YFData.download(self.symbols, start=self.start_date, end=self.end_date, interval='1D', missing_index='drop')
        self.data = price_data.data[self.symbols[0]]
        self.data.index = pd.to_datetime(self.data.index).strftime('%Y-%m-%d')
        return self.data

    def preprocess_data(self):
        import pandas_ta as ta
        self.data['RSI'] = ta.rsi(self.data['Close'], length=14)
        self.data['SMA'] = ta.sma(self.data['Close'], length=50)
        self.data['EMA'] = ta.ema(self.data['Close'], length=50)
        self.data['ATR'] = ta.atr(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], length=14)
        macd_df = ta.macd(self.data['Close'])
        self.data['MACD'] = macd_df['MACD_12_26_9']
        self.data['MACD_signal'] = macd_df['MACDs_12_26_9']
        self.data.dropna(inplace=True)
        return self.data

    def create_trading_signal(self):
        self.data['Signal'] = 0
        buy_count = 0

        for i in range(2, len(self.data) - 1):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i - 1] and self.data['Close'].iloc[i - 1] > self.data['Close'].iloc[i - 2]:
                if buy_count < 3:
                    self.data.at[self.data.index[i + 1], 'Signal'] = 1
                    buy_count += 1
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i - 1] and self.data['Close'].iloc[i - 1] < self.data['Close'].iloc[i - 2]:
                if buy_count > 0:
                    self.data.at[self.data.index[i + 1], 'Signal'] = -1
                    buy_count = 0
                else:
                    self.data.at[self.data.index[i + 1], 'Signal'] = 0

        return self.data