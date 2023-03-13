from stock.exception import StockException
from stock.logger import logging
from stock.utils.main_utils import load_numpy_array_data
from stock.entity.artifact_entity import ModelTrainerArtifact,ModelEvaluationArtifact
from stock.entity.artifact_entity import DataTransformationArtifact
from stock.entity.config_entity import ModelEvaluationConfig
import os,sys
try :
    from stock.ml.metric.regression_matric import get_regression_score
except Exception as e:
    raise StockException(e,sys)
from stock.ml.model.estimator import SensorModel
from stock.utils.main_utils import save_object,load_object,write_yaml_file
from stock.ml.model.estimator import ModelResolver
from stock.constant.training_pipeline import TARGET_COLUMN
from stock.ml.model.estimator import TargetValueMapping
import pandas  as  pd
import numpy as np


class ModelEvaluation:


    def __init__(self,model_eval_config:ModelEvaluationConfig,
                    data_transformation_artifact:DataTransformationArtifact,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:
            
            self.model_eval_config=model_eval_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
        except Exception as e:
            raise StockException(e,sys)
    
    def split_timeseries_data(self ,train_arr : np.ndarray  ):

                training_data_len = int(np.ceil( len(train_arr) * .95 ))
            
                train_data = train_arr[0:int(training_data_len), :]
            
                # Split the data into x_train and y_train data sets
                x_train = []
                y_train = []
    
                for i in range(60, len(train_data)):
                    x_train.append(train_data[i-60:i, 0])
                    y_train.append(train_data[i, 0])
                    if i<= 61:
                        print(x_train)
                        print(y_train)
                        print()
    
                # Convert the x_train and y_train to numpy arrays 
                x_train, y_train = np.array(x_train), np.array(y_train)
    
                # Reshape the data
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                # x_train.shape
                
                return x_train, y_train


    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #valid train and test file dataframe
            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr =  load_numpy_array_data(test_file_path)

            x_train,y_train = self.split_timeseries_data(train_arr)
            x_test,y_test = self.split_timeseries_data(test_arr)


            all_data = np.concatenate((x_train, x_test), axis=0)
            y_true = np.concatenate((y_train, y_test), axis=0)
            
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted=True


            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=None, 
                    best_model_path=None, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact, 
                    best_model_metric_artifact=None)
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)
            
            y_trained_pred = train_model.predict(all_data)
            y_latest_pred  =latest_model.predict(all_data)

            trained_metric = get_regression_score(y_true, y_trained_pred)
            latest_metric = get_regression_score(y_true, y_latest_pred)

            improved_accuracy = trained_metric.rmse-latest_metric.rmse
            if self.model_eval_config.change_threshold < improved_accuracy:
                #0.02 < 0.03
                is_model_accepted=True
            else:
                is_model_accepted=False

            
            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=improved_accuracy, 
                    best_model_path=latest_model_path, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=trained_metric, 
                    best_model_metric_artifact=latest_metric)

            model_eval_report = model_evaluation_artifact.__dict__

            #save the report
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise StockException(e,sys)
