from stock.entity.artifact_entity import RegressionMetricArtifact
from stock.exception import StockException
import os,sys
import numpy as np

def get_regression_score(y_true,y_pred)->RegressionMetricArtifact:
    try:
        model_rmse = float(np.sqrt(np.mean(((y_pred - y_true) ** 2))))
        

        regression_metric = RegressionMetricArtifact(rmse=model_rmse)
        return regression_metric
    except Exception as e:
        raise StockException(e,sys)
