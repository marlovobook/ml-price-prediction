from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, params=None):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**(params if params else {}))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score, classification_report
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, report