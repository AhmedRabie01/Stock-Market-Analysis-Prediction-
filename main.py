import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, robust_scale
from stock.configuration.mongo_db_connection import MongoDBClient
from stock.exception import StockException
import os,sys
from stock.logger import logging
from stock.pipeline.training_pipeline import TrainPipeline
import os
from stock.utils.main_utils import read_yaml_file
from stock.constant.training_pipeline import SAVED_MODEL_DIR
from fastapi import FastAPI
from stock.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from stock.ml.model.estimator import ModelResolver
from stock.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from stock.constant.training_pipeline import TARGET_COLUMN
import matplotlib.pyplot as plt
import io
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")


env_file_path=os.path.join(os.getcwd(),"env.yaml")

def set_env_variable(env_file_path):

    if os.getenv('MONGO_DB_URL',None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL']=env_config['MONGO_DB_URL']



app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:

        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")
        
 # Define your API endpoint for prediction
@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df = df[['Date', 'Close']]  # Assuming 'Date' and 'Close' are the relevant columns
        df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime object
        robust_scaler = RobustScaler()
        scaled_data = robust_scaler.fit_transform(df[['Close']])
        x_test= np.array(scaled_data)

        # Reshape x_test to match the expected input shape of the model
        x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available")
        
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        print('model_found')

        # predict
        y_test_pred = model.predict(x_test)
        predictions = robust_scaler.inverse_transform(y_test_pred)
        predicted_column = predictions.reshape(-1)
        
        # Convert the predicted data to a CSV file
        predicted_data = pd.DataFrame({'Date': df['Date'], 'Predictions': predicted_column})
        csv_file = io.StringIO()
        predicted_data.to_csv(csv_file, index=False)

        # Generate the plot
        plt.figure(figsize=(16, 6))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(df['Date'], df['Close'], label='Close')
        plt.plot(df['Date'], predicted_column, label='Predictions')
        plt.legend(loc='lower right')
        plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees for better visibility
        plt.tight_layout()

        # Convert the plot to a PNG image
        img_file = io.BytesIO()
        plt.savefig(img_file, format='png')
        img_file.seek(0)

        # Return the plot as a streaming response
        return StreamingResponse(iter([img_file.getvalue()]), media_type="image/png")
        
    except Exception as e:
        raise Response(f"Error Occured! {e}")


def main():
        try:
            set_env_variable(env_file_path)
            training_pipeline = TrainPipeline()
            training_pipeline.run_pipeline()
        except Exception as e:
            print(e)
            logging.exception(e)

if __name__=="__main__":
    #main()
    # set_env_variable('env.yaml')
    main()
    app_run(app, host=APP_HOST, port=APP_PORT)
