import numpy as np
import pandas as pd

times = [14, 17, 20]
locations = [ 'dangjin_floating', 'dangjin_warehouse', 'dangjin', 'ulsan']

def to_date(x):
    return pd.DateOffset(hours=x)


def interpolation(fcst):
    # 4462:337960, 342251

    """
                Forecast time  forecast  ...  WindDirection  Cloud
4462  2014-12-31 14:00:00      10.0  ...          325.0    3.0
4463  2014-12-31 14:00:00      13.0  ...          324.0    3.0

              Forecast time  forecast  ...  WindDirection  Cloud
342100  2021-02-27 20:00:00      52.0  ...           42.0    4.0
342101  2021-02-27 23:00:00       4.0  ...           92.0    4.0
342102  2021-02-27 23:00:00       7.0  ...          117.0    3.0

    """

    fcst = fcst.iloc[4462:342101]
    fcst['Forecast_time'] = pd.to_datetime(fcst['Forecast time'])
    fcst_inter = {}

    for time in times:
        # Selecting Forecast data by the forecast time
        num = 24 - time
        data = fcst[fcst['Forecast_time'].dt.hour == time]
        data = data[(data['forecast'] >= num) & (data['forecast'] <= num + 24)]
        data['Forecast_time'] = data['Forecast_time'] + data['forecast'].map(to_date)

        # Interpolating time
        date_inter = pd.DataFrame()
        date_inter['Forecast_time'] = pd.date_range(start='2015-01-01 00:00:00', end='2021-03-01 00:00:00', freq='H')
        date_inter = pd.merge(date_inter, data, on='Forecast_time', how='outer')

        # Interpolating data
        data_inter = date_inter.interpolate()
        data_inter.drop(['Forecast time','forecast'], axis=1, inplace=True)
        fcst_inter[time] = data_inter

    return fcst_inter

def energy_checker(energy):

    checker = energy.loc[(energy.time.dt.hour >= 9) & (energy.time.dt.hour <= 16), :]
    dangjin_f_check = checker[checker['dangjin_floating'] == 0]['dangjin_floating'].replace(0, np.NaN)
    dangjin_w_check = checker[checker['dangjin_warehouse'] == 0]['dangjin_warehouse'].replace(0, np.NaN)
    dangjin_check = checker[checker['dangjin'] == 0]['dangjin'].replace(0, np.NaN)
    ulsan_check = checker[checker['ulsan'] == 0]['ulsan'].replace(0, np.NaN)

    re_energy = energy.copy()

    for i in range(len(dangjin_f_check.index)):
        re_energy['dangjin_floating'][dangjin_f_check.index[i]] = dangjin_f_check[dangjin_f_check.index[i]]

    for i in range(len(dangjin_w_check.index)):
        re_energy['dangjin_warehouse'][dangjin_w_check.index[i]] = dangjin_w_check[dangjin_w_check.index[i]]

    for i in range(len(dangjin_check.index)):
        re_energy['dangjin'][dangjin_check.index[i]] = dangjin_check[dangjin_check.index[i]]

    for i in range(len(ulsan_check.index)):
        re_energy['ulsan'][ulsan_check.index[i]] = ulsan_check[ulsan_check.index[i]]


    return re_energy


def merger(data, energy):

    merged_data = {}
    energy.rename(columns={'time': 'Forecast_time'}, inplace=True)

    for location in locations:

        data_imsi_dic = {}
        data_imsi = pd.DataFrame()

        if location == 'ulsan':
            tmp_fcst = data[location]

        else:
            tmp_fcst = data['dangjin']

        tmp_energy = energy[['Forecast_time', location]]

        for time in times:
            data_imsi = pd.merge(tmp_fcst[time], tmp_energy, on='Forecast_time', how='outer')
            #data_imsi.to_csv('check/' + location + '_' + str(time) + '.csv')
            data_imsi_dic[time] = data_imsi
            
        merged_data[location] = data_imsi_dic

    return merged_data


def data_splitter(data):
    return train_x, train_y, val_x, val_y