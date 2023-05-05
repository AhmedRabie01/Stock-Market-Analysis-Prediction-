from stock.utils.main_utils import load_numpy_array_data
from stock.exception import StockException
from stock.logger import logging
from stock.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from stock.entity.config_entity import ModelTrainerConfig
import os,sys
from stock.ml.metric.regression_matric import get_regression_score
from stock.ml.model.estimator import StockModel
from stock.utils.main_utils import save_object,load_object
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
        data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise StockException(e,sys)

    def perform_hyper_paramter_tunig(self):...
        
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
    
    def train_model(self,x_train,y_train, num_epochs=10, batch_size=32, validation_split=0.2):
        try:
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=validation_split)

            return model
        except Exception as e:
            raise e
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr =  load_numpy_array_data(test_file_path)

            # splitting the data 
            x_train,y_train = self.split_timeseries_data(train_arr)
            x_test,y_test = self.split_timeseries_data(test_arr)
            
            # model prediction 
            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)
            
            
            rmse_train = get_regression_score(y_train ,y_train_pred )

            
            if rmse_train.rmse >= self.model_trainer_config.expected_rmse:
                raise Exception("Trained model is not good to provide expected accuracy")
            
            y_test_pred = model.predict(x_test)
            rmse_test = get_regression_score(y_test,y_test_pred)
            

            #Overfitting and Underfitting
            diff = abs(rmse_train.rmse-rmse_test.rmse)
            
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            sensor_model = StockModel(preprocessor=preprocessor,model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)

            #model trainer artifact

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
            train_metric_artifact=rmse_train,
            test_metric_artifact=rmse_test)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise StockException(e,sys)