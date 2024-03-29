import sys
import os
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from stock.constant.training_pipeline import TARGET_COLUMN
from stock.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from stock.entity.config_entity import DataTransformationConfig
from stock.exception import StockException
from stock.logger import logging
from stock.utils.main_utils import save_numpy_array_data, save_object
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig):
        """
        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise StockException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise StockException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            robust_scaler = RobustScaler()
            imputer = IterativeImputer(missing_values=np.nan,add_indicator=True)
            preprocessor = Pipeline(
                steps=[
                    ("Imputer", imputer),
                    ("RobustScaler", robust_scaler)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise StockException(e, sys) from e

    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            preprocessor = self.get_data_transformer_object()

            train_df = train_df[[TARGET_COLUMN]]
            test_df  = test_df[[TARGET_COLUMN]]

            preprocessor_object = preprocessor.fit(train_df)
            transformed_input_train_feature = preprocessor_object.transform(train_df)
            transformed_input_test_feature = preprocessor_object.transform(test_df)

            train_arr =  np.array(transformed_input_train_feature) 
            test_arr =  np.array(transformed_input_test_feature) 

            X_train_series = train_arr.reshape((train_arr.shape[0], train_arr.shape[1], 1))
            X_test_series = test_arr.reshape((test_arr.shape[0], 1))



            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=X_train_series)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=X_test_series)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        
        except Exception as e:
            raise StockException(e, sys) from e
