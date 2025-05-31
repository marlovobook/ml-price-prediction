class Portfolio:
    def __init__(self, initial_cash=10000, fees=0.001):
        self.initial_cash = initial_cash
        self.fees = fees
        self.positions = {}
        self.cash = initial_cash
        self.total_value = initial_cash
        self.history = []

    def buy(self, symbol, price, size):
        cost = price * size * (1 + self.fees)
        if cost <= self.cash:
            self.cash -= cost
            if symbol in self.positions:
                self.positions[symbol] += size
            else:
                self.positions[symbol] = size
            self.history.append((symbol, 'buy', size, price))
        else:
            print("Not enough cash to buy.")

    def sell(self, symbol, price, size):
        if symbol in self.positions and self.positions[symbol] >= size:
            revenue = price * size * (1 - self.fees)
            self.cash += revenue
            self.positions[symbol] -= size
            if self.positions[symbol] == 0:
                del self.positions[symbol]
            self.history.append((symbol, 'sell', size, price))
        else:
            print("Not enough shares to sell.")

    def calculate_total_value(self, current_prices):
        total_value = self.cash
        for symbol, size in self.positions.items():
            total_value += current_prices[symbol] * size
        self.total_value = total_value
        return total_value

    def get_performance_metrics(self):
        return {
            'initial_cash': self.initial_cash,
            'final_value': self.total_value,
            'profit_loss': self.total_value - self.initial_cash,
            'positions': self.positions,
            'history': self.history
        }