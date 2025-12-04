# Hull Tactical - Market Prediction
This guide outlines the steps to participate in the Kaggle competition for Hull Tactical - Market Prediction.
<br>[Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)

# 1. Join Kaggle Competition
- Create a Kaggle account if you don't have one.
- Join the "Hull Tactical - Market Prediction" competition.
- Familiarize yourself with the rules, evaluation metric (Log Loss), and data format.

# 2. Notebook Setup
- Create a new notebook in Kaggle or your preferred environment.
- Add the competition dataset (/kaggle/input/hull-tactical-market-prediction/train.csv) to your notebook.

# 3. Environment Settings
- LGBM: Set Accelerator to "None". (But, it doesn't matter if "GPU T4 x2".)
- LSTM: Set Accelerator to "GPU T4 x2".

# 4. Run Notebook
- Execute the cells in your notebook sequentially to train models and generate predictions.
- Ensure your final submission. In this Kaggle competition, you don't submit a CSV file. Instead, you need to create and submit a prediction function as shown below.
```
import polars as pl
import kaggle_evaluation.default_inference_server

def predict(test: pl.DataFrame):
    # your prediction code
    return 0

inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)
```

