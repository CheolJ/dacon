import lightgbm as lgb
import catboost as cat
import pandas as pd
import numpy as np
from data_processor_time import data_splitter
import matplotlib.pyplot as plt

times = [14, 17, 20]


def sola_nmae(answer, pred, capacity):
    absolute_error = np.abs(answer - pred)

    absolute_error /= capacity

    target_idx = np.where(answer >= capacity * 0.1)

    nmae = 100 * absolute_error[target_idx].mean()

    return nmae


def data_generator(raw_data, location):
    """
    drop : windspeed, wind direction
    preparation list = mist, dust
    raw feature         : year, month, date, hour, temp, precipitation, precrate, humidity, cloud, energy
    statistical feature : temp_6, temp_12, temp_18, temp_24, t_avg3, t_avg_5, t_avg7
                          humidity_6, humidity_12, humidity_18, humidity_24

    """
    temp = {}
    
    for time in times:
        train_data = pd.DataFrame()
        data = raw_data[time]

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
        train_data['energy'] = data[location]

        # statistical values_temp
        train_data['temp_6'] = data['Temperature'] - data['Temperature'].shift(6)
        train_data['temp_12'] = data['Temperature'] - data['Temperature'].shift(12)
        train_data['temp_18'] = data['Temperature'] - data['Temperature'].shift(18)
        train_data['temp_24'] = data['Temperature'] - data['Temperature'].shift(24)

        train_data['t_avg3'] = data['Temperature'].rolling(window=3, min_periods=1, center=True).mean()
        train_data['t_avg5'] = data['Temperature'].rolling(window=5, min_periods=1, center=True).mean()
        train_data['t_avg7'] = data['Temperature'].rolling(window=7, min_periods=1, center=True).mean()

        # Statistical values_humidity
        train_data['h_avg3'] = data['Humidity'].rolling(window=3, min_periods=1, center=True).mean()
        train_data['h_avg5'] = data['Humidity'].rolling(window=5, min_periods=1, center=True).mean()
        train_data['h_avg7'] = data['Humidity'].rolling(window=7, min_periods=1, center=True).mean()

        #train_test split
        start = '2015-01-01 01:00:00'
        middle = '2021-02-01 00:00:00'
        end = '2021-02-28 23:00:00'
        
        start_index = train_data[data['Forecast_time'] == start].index[0]
        middle_index = train_data[data['Forecast_time'] == middle].index[0]
        end_index = train_data[data['Forecast_time'] == end].index[0]

        test_data = train_data.loc[middle_index:end_index, :].copy()
        train = train_data.loc[start_index:middle_index, :]
        
        train.dropna(inplace=True)
        
        train_x = train[train.columns.difference(['energy'])]
        train_y = train['energy']
        
        val_x = train_x[-24*30:]
        val_y = train_y[-24*30:]
        
        train_x = train_x[:-24*30]
        train_y = train_y[:-24*30]
        
        test_x = test_data[test_data.columns.difference(['energy'])]
        
        temp[time] = {'train_x' : train_x,
                      'train_y': train_y,
                      'val_x': val_x,
                      'val_y': val_y,
                      'test_x': test_x}
        
    return temp
    


def lgb_model(raw_data, capacity, location):
    
    
    def nmae_10(y_pred, dataset):
        y_true = dataset.get_label()
    
        absolute_error = abs(y_true - y_pred)
        absolute_error /= capacity
    
        target_idx = np.where(y_true >= capacity * 0.1)
    
        nmae = 100 * absolute_error[target_idx].mean()
    
        return 'score', nmae, False
    
    
    models = {}
     
    params = {
        'learning_rate': 0.05,
        'objective': 'regression',
        'metric': 'mae',
        'seed': 42
    }
    
    for time in times:
        data = raw_data[time]
        train_x = data['train_x'].to_numpy()
        train_y = data['train_y'].to_numpy()
        val_x = data['val_x'].to_numpy()
        val_y = data['val_y'].to_numpy()
        
        train_dataset = lgb.Dataset(train_x, train_y)
        val_dataset = lgb.Dataset(val_x, val_y)
        
        models[time] = lgb.train(params, train_dataset, 10000, val_dataset, feval=nmae_10, verbose_eval=100, early_stopping_rounds=100)
        pred = models[time].predict(val_x)
        
        plt.figure(figsize=(20,5))
        plt.title(location + '/ forecast_time : ' + str(time))
        plt.plot(val_y, label = 'true')
        plt.plot(pred, label = 'pred')
        plt.legend()
        plt.show()
        #print('CV Score : ', sola_nmae(val_y, pred, capacity))
        
    
    return models

