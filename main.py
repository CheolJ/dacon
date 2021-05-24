import pandas as pd
import data_processor_time as dt

locations = [ 'dangjin_floating', 'dangjin_warehouse', 'dangjin', 'ulsan']
tiems = [14, 17, 20]
# Read_files
energy = pd.read_csv('data/new_energy.csv')
energy.time = pd.to_datetime(energy.time)
data_dangjin = pd.read_csv('data/dangjin_fcst_revision.csv')
data_ulsan = pd.read_csv('data/ulsan_fcst_revision.csv')

# Columns used for data analysis

dangjin = dt.interpolation(data_dangjin)
ulsan = dt.interpolation(data_ulsan)
fcst_data = {'dangjin': dangjin,
             'ulsan': ulsan}

# Energy checker / skip by revised energy
# new_energy = dt.energy_checker(energy)

# Data merger
merged_data = dt.merger(fcst_data, new_energy)

# Model builder
import model_developer as mdv
cat_models = {}
lgb_models = {}
for location in locations:

    cat_models[location] = mdv.cat_model(merged_data[location])
    lgb_models[location] = mdv.lgb_model(merged_data[location])


# Stacked ensemble

