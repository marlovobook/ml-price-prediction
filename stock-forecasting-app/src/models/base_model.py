class BaseModel:
    def train(self, X, y):
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def predict(self, X):
        raise NotImplementedError("Predict method must be implemented by subclasses.")

    def evaluate(self, y_true, y_pred):
        raise NotImplementedError("Evaluate method must be implemented by subclasses.")