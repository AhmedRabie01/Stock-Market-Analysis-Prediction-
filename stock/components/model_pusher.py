from stock.exception import StockException
from stock.logger import logging
from stock.entity.artifact_entity import ModelPusherArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from stock.entity.config_entity import ModelEvaluationConfig,ModelPusherConfig
import os,sys
from stock.ml.metric.classification_metric import get_classification_score
from stock.utils.main_utils import save_object,load_object,write_yaml_file
import datetime
import shutil

class ModelPusher:

    def __init__(self,
                model_pusher_config:ModelPusherConfig,
                model_eval_artifact:ModelEvaluationArtifact):

        try:
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
        except  Exception as e:
            raise StockException(e, sys)
    

    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            trained_model_path = self.model_eval_artifact.trained_model_path
            
            # Add version number to model file name
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_file_name = f"model_{current_time}.pkl"
            model_file_path = os.path.join(self.model_pusher_config.model_file_path, model_file_name)


            #Creating model pusher dir to save model
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)

            #saved model dir
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            #prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path, model_file_path=model_file_path)
            return model_pusher_artifact
        except  Exception as e:
            raise StockException(e, sys)