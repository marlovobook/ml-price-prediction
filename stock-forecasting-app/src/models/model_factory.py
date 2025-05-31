class ModelFactory:
    def create_model(self, model_type: str):
        if model_type == 'xgboost':
            from .xgboost_model import XGBoostModel
            return XGBoostModel()
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized.")