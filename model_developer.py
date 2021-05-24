import lightgbm as lgb
import catboost as cat
import pandas as pd
from data_processor_time import


def data_generator(data):
    """
    drop : windspeed, wind direction
    preparation list = mist, dust
    """
    train_data = pd.DataFrame()

    # train data list
    train_data['year'] = data['Forecast_time'].dt.year
    train_data['month'] = data['Forecast_time'].dt.month
    train_data['date'] = data['Forecast_time'].dt.day
    train_data['hour'] = data['Forecast_time'].dt.hour
    train_data['temp'] = data['Temperature']
    train_data['precipitation'] = data['Precipitation']
    train_data['precrate'] = data['PrecRate']
    train_data['humidity'] = data['Humidity']
    train_data['cloud'] = data['Cloud']

    # statistical values

    #

def cat_model(data):
    train_x, train_y, val_x, val_y = data_splitter(data)
    model = lgb.LGBMRegressor()

    return models


def lgb_model(data):
    train_x, train_y, val_x, val_y = data_splitter(data)
    model = cat.CatBoostRegressor()

    return models

