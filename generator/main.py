import pandas as pd
import data_processor_time as dt
import model_developer as mdv

locations = ['dangjin_floating', 'dangjin_warehouse', 'dangjin', 'ulsan']
tiems = [14, 17, 20]
generator_capacity = [1000, 700, 1000, 500]

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
energy = dt.energy_checker(energy)

# Data merger
merged_data = dt.merger(fcst_data, energy)

# data_split
split_data = {}

for location in locations:
    split_data[location] = mdv.data_generator(merged_data[location], location)

# Model build
cat_models = {}
lgb_models = {}

for location, capacity in zip(locations, generator_capacity):
    lgb_models[location] = mdv.lgb_model(split_data[location], capacity, location)
    
# Stacked ensemble