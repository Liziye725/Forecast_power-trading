
# All the features: ['all', 'actual', 'entsoe', 'weather_t', 'weather_i', 'holiday', 'weekday', 'hour', 'month']
model_cat_id = "01"
feature = ['actual', 'entsoe']

# LSTM layer configuration
layer_conf = [ True, True, True]
cells = [[ 5, 10, 20, 30, 50, 75, 100, 125, 150], [0, 10, 20, 50], [0, 10, 15, 20]]
dropout = [0, 0.1, 0.2]
batch_size = [8]
timesteps = [1]

import os
# os.environ["KERAS_BACKEND"] = "torch"
# os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print(keras.__version__)

import sys
import math
import itertools
import datetime as dt
from decimal import *
import pytz
import time as t
import pandas as pd
import numpy as np
from pandas import read_csv
from numpy import newaxis
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as stattools
from tabulate import tabulate
import math
import keras
from keras import backend as K
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (9, 5)
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from lstm_load import data, lstm

# %% [markdown]
# rmse: Root Mean Square Error - 模型预测值与实际值之间的差异
# 
# mae: Mean Absolute Error - 衡量预测值与实际值的差异。
# 
# mape: Mean Absolute Percentage Error - 衡量预测误差相对于实际值的百分比
# 
# train_loss - 存储训练集上的损失值。损失函数是用于衡量模型预测误差的标准。
# 
# valid_loss - 存储验证集上的损失值。用于评估模型在未见数据上的表现


abspath = os.path.abspath('../data/fulldataset.csv')
loc_tz = pytz.timezone('Europe/Zurich')
split_date = loc_tz.localize(dt.datetime(2017,2,1,0,0,0,0))
validation_split = 0.2
epochs = 30
verbose = 0
results = pd.DataFrame(columns=['module_name', 'config', 'dropout', 'train_loss', 'train_rmse', 'train_mae', 'train_mape', 'valid_loss', 'valid_rmse', 'valid_mae', 'valid_mape', 'test_rmse', 'test_mae', 'test_mape', 'epochs', 'batch_train', 'input_shape', 'total_time', 'time_step', 'splits'])
early_stopping = True
min_delta = 0.006
patience = 2

# UTC: Coordinated Universal Time
# - BST: British Summer Time  -  UTC + 1
# - CEST: Central European Summer Time  -  UTC + 2
# 夏天 - 英国比丹麦晚一小时，英国4pm，丹麦5pm
# 
# ------ Winter ------
# -   UTC + 0
# - CET: Central European Time  -  UTC + 1
# 
# 时间按时区转换
# 这要用到datetime模块的astimezone方法来实现。如下所示，开始生成本地时间，然后在转成utc时间。

dt.datetime.now(pytz.timezone('CET'))
utc = pytz.timezone('UTC')
cet = pytz.timezone('CET')
now_time = dt.datetime.now()
utc_time = utc.normalize(now_time.astimezone(tz=utc))
cet_time = cet.normalize(now_time.astimezone(tz=cet))
print('now:', now_time, '\nutc:', utc_time, '\ncet:', cet_time)


def generate_combinations(model_name=None, layer_conf=None, cells=None, dropout=None, batch_size=None, timesteps=None):
    models = []
    layer_conb = list(itertools.product(*cells))
    configs = [layer_conb, dropout, batch_size, timesteps]
    combinations = list(itertools.product(*configs))

    for ix, comb in enumerate(combinations):
        m_name = model_name
        m_name += str(ix + 1)

        layers = []
        for idx, level in enumerate(comb[0]):
            return_sequence = True
            if all(size == 0 for size in comb[0][idx + 1:]) == True:
                return_sequence = False
            if (idx + 1) == len(comb[0]):
                return_sequence = False
            if level > 0:
                layers.append({'type': 'lstm', 'cells': level, 'dropout': comb[1], 'stateful': layer_conf[idx], 'ret_seq': return_sequence })
                m_name += '_1-' + str(comb[1])
        if comb[1] > 0:
            m_name += '_d-' + str(comb[1])
        model_config = {
            'name': m_name,
            'layers': layers,
            'batch_size': comb[2],
            'timesteps': comb[3]
        }
        models.append(model_config)

        print('==================')
        print(tabulate([
            ['Number of model configs generated', len(combinations)]],
            tablefmt="jira", numalign="right", floatfmt=".3f"))
        return models


result_dir = '../results/notebook_' + model_cat_id + '/'
plot_dir = '../plots/notebook_' + model_cat_id + '/'
model_dir = '../models/notebook_' + model_cat_id + '/'
os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
output_table = result_dir + model_cat_id + '_results_' + t.strftime("%Y%m%d") + '.csv'
test_output_table = result_dir + model_cat_id + '_test_results' + t.strftime("%Y%m%d") + '.csv'

models = []
models = generate_combinations(
    model_name=model_cat_id + '_', layer_conf=layer_conf, cells=cells, dropout=dropout,
    batch_size=batch_size,timesteps=[1]
)

print(models)

df = data.load_dataset(path=abspath, modules=feature)
df_scaled = df.copy()
df_scaled = df_scaled.dropna()
floats = [key for key in dict(df_scaled.dtypes) if dict(df_scaled.dtypes)[key] in ['float64']]
scaler =  StandardScaler()
scaled_columns = scaler.fit_transform(df_scaled[floats]) # noraml distribution
df_scaled[floats] = scaled_columns
df_train = df_scaled.loc[(df_scaled.index < split_date)].copy()
df_test = df_scaled.loc[df_scaled.index >= split_date].copy()


y_train = df_train['actual'].copy()
X_train = df_train.drop('actual', axis=1).copy()
y_test = df_test['actual'].copy()
X_test = df_test.drop('actual', axis=1).copy()


#### Training models on all configuration
start_time = t.time()

def validate_layers(layers):
    for layer in layers:
        if 'cells' not in layer:
            raise KeyError(f"Missing 'cells' key in layer: {layer}")

results = pd.DataFrame()

for idx, m in enumerate(models):
    stopper = t.time()
    print('======= Model {}/{} ========'.format(idx+1, len(models)))
    print(tabulate([['Starting with model', m['name']], ['Starting time', dt.datetime.fromtimestamp(stopper)]], 
                   tablefmt="jira", numalign="right", floatfmt=".3f"))
    try:
        validate_layers(m['layers'])
        print("Validated layers:", m['layers'])  # Debugging output
        
        model = lstm.create_model(layers=m['layers'], batch_size=m['batch_size'], 
                                  timesteps=m['timesteps'], features=X_train.shape[1])
        print("Model created successfully.")  # Debugging output
        
        history = lstm.train_model(model=model, mode='fit', y=y_train, X=X_train, 
                                   batch_size=m['batch_size'], timesteps=m['timesteps'], epochs=epochs,
                                   rearrange=False, validation_split=validation_split, verbose=verbose,
                                   early_stopping=early_stopping, min_delta=min_delta, patience=patience)
        print("Training completed.")  # Debugging output
        
        min_loss = np.min(history.history['val_loss'])
        min_idx = np.argmin(history.history['val_loss'])
        min_epoch = min_idx + 1

        if verbose > 0:
            print('_________________________')
            print(tabulate([['Minimum validation loss at epoch', min_epoch, 'Time: {}'.format(t.time()-stopper)],
                            ['Training loss & MAE', history.history['loss'][min_idx], history.history['mean_absolute_error'][min_idx] ],
                            ['Validation loss & mae', history.history['val_loss'][min_idx], history.history['val_mean_absolute_error'][min_idx]],
                            ], tablefmt="jira", numalign="right", floatfmt=".3f"))
            print('_________________________')

        result = pd.DataFrame([{'model_name': m['name'], 'config': m, 'train_loss': history.history['loss'][min_idx], 'train_rmse': 0,
                   'train_mae': history.history['mean_absolute_error'][min_idx], 'train_mape': 0,
                   'valid_loss': history.history['val_loss'][min_idx], 'valid_rmse': 0,
                   'valid_mae': history.history['val_mean_absolute_error'][min_idx], 'valid_mape':0,
                   'test_rmse':0, 'test_mae': 0, 'test_mape': 0, 'epochs': '{}/{}'.format(min_epoch, epochs), 'batch_train': m['batch_size'],
                   'input_shape': (X_train.shape[0], timesteps, X_train.shape[1]), 'total_time': t.time()-stopper,
                   'time_step': 0, 'splits': str(split_date), 'dropout': m['layers'][0]['dropout']                
                   }])
        
        results = pd.concat([results, result], ignore_index=True)
        model.save(model_dir + m['name'] + '.h5')
        results.to_csv(output_table, sep=';')
        K.clear_session()
        tf.reset_default_graph()

    except KeyError as e:
        print(f"Configuration error: {e}")
        continue    

    except BaseException as e:
        print('======= ERROR {}/{} ======='.format(idx+1, len(models)))
        print(tabulate([['Model:', m['name']], ['Config:', m]], tablefmt="jira", numalign="right", floatfmt=".3f"))
        print('Error: {}'.format(e))
        result = pd.DataFrame([{'model_name': m['name'], 'config':m, 'train_loss': str(e)}])
        results = pd.concat([results, result], ignore_index=True)
        results.to_csv(output_table, sep=';')
        continue


start_time = t.time()
for idx, m in enumerate(models):
    stopper = t.time()
    print('========================= Model {}/{} ========================='.format(idx+1, len(models)))
    print(tabulate([['Starting with model', m['name']], ['Starting time', dt.datetime.fromtimestamp(stopper)]],
                   tablefmt="jira", numalign="right", floatfmt=".3f"))
    try:
        # Creating the Keras Model
        model = lstm.create_model(layers=m['layers'], batch_size=m['batch_size'], 
                          timesteps=m['timesteps'], features=X_train.shape[1])
        # Training...
        history = lstm.train_model(model=model, mode='fit', y=y_train, X=X_train, 
                                   batch_size=m['batch_size'], timesteps=m['timesteps'], epochs=epochs, 
                                   rearrange=False, validation_split=validation_split, verbose=verbose, 
                                   early_stopping=early_stopping, min_delta=min_delta, patience=patience)

        # Write results
        min_loss = np.min(history.history['val_loss'])
        min_idx = np.argmin(history.history['val_loss'])
        min_epoch = min_idx + 1
        
        if verbose > 0:
            print('______________________________________________________________________')
            print(tabulate([['Minimum validation loss at epoch', min_epoch, 'Time: {}'.format(t.time()-stopper)],
                        ['Training loss & MAE', history.history['loss'][min_idx], history.history['mean_absolute_error'][min_idx]  ], 
                        ['Validation loss & mae', history.history['val_loss'][min_idx], history.history['val_mean_absolute_error'][min_idx] ],
                       ], tablefmt="jira", numalign="right", floatfmt=".3f"))
            print('______________________________________________________________________')
        
        
        result = [{'model_name': m['name'], 'config': m, 'train_loss': history.history['loss'][min_idx], 'train_rmse': 0,
                   'train_mae': history.history['mean_absolute_error'][min_idx], 'train_mape': 0,
                   'valid_loss': history.history['val_loss'][min_idx], 'valid_rmse': 0, 
                   'valid_mae': history.history['val_mean_absolute_error'][min_idx],'valid_mape': 0, 
                   'test_rmse': 0, 'test_mae': 0, 'test_mape': 0, 'epochs': '{}/{}'.format(min_epoch, epochs), 'batch_train':m['batch_size'],
                   'input_shape':(X_train.shape[0], timesteps, X_train.shape[1]), 'total_time':t.time()-stopper, 
                   'time_step':0, 'splits':str(split_date), 'dropout': m['layers'][0]['dropout']
                  }]
        results = results.append(result, ignore_index=True)
        
        # Saving the model and weights
        model.save(model_dir + m['name'] + '.h5')
        
        # Write results to csv
        results.to_csv(output_table, sep=';')
        
        #if not os.path.isfile(output_table):
            #results.to_csv(output_table, sep=';')
        #else: # else it exists so append without writing the header
        #    results.to_csv(output_table,mode = 'a',header=False, sep=';')
        
        K.clear_session()
        import tensorflow as tf
        tf.reset_default_graph()
        
    # Shouldn't catch all errors, but for now...
    except BaseException as e:
        print('=============== ERROR {}/{} ============='.format(idx+1, len(models)))
        print(tabulate([['Model:', m['name']], ['Config:', m]], tablefmt="jira", numalign="right", floatfmt=".3f"))
        print('Error: {}'.format(e))
        result = pd.DataFrame([{'model_name': m['name'], 'config': m, 'train_loss': str(e)}])
        results = pd.concat([results, result], ignore_index=True)
        results.to_csv(output_table,sep=';')
        continue


import time as t
stopper = t.time()
# Process
t.sleep(2)
total_time = t.time() - stopper
print(f"Total time elapsed: {total_time} seconds")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# print(history.history.keys())
loss_history = history.history['loss']
accuracy_history = history.history['accuracy']

for epoch in range(len(loss_history)):
    print(f"Epoch {epoch+1}: Loss = {loss_history[epoch]}, Accuracy = {accuracy_history[epoch]}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))

model = Sequential()
model.add(Input(shape=(10,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
