# -*- coding: utf-8 -*-
"""
Created on Monday 21 April 2025
@author: Seyid Amjad Ali
""" 
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score, max_error,mean_squared_log_error, median_absolute_error, mean_poisson_deviance, mean_gamma_deviance
from lightgbm import LGBMRegressor
# preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, QuantileTransformer, PowerTransformer, Binarizer
import warnings

warnings.filterwarnings("ignore")

input_filename = 'Data.xlsx'
output_filename = 'lightGBM_Data(SurvivalPercetage).txt'

data_orj = pd.read_excel(input_filename)

# Preprocessing
stdsc = StandardScaler()

data_pp = data_orj
print(data_pp)

# Survival Percetage -- 4,      Fresh Weight -- 5,      Dry Weight -- 6
# Total Phenolic -- 7,          Total Flavonoid -- 8
# Total Triterpenoid -- 9,      DPHH -- 10,      SOD -- 11
# CAT -- 12,      APX -- 13
X = data_pp.iloc[:,[0, 1, 2]].values
y = data_pp.iloc[:,[4]].values

# Scaling
X_scaled = stdsc.fit_transform(X)
    
# LOOCV
loo = LeaveOneOut()
loo.get_n_splits(X_scaled)
n_samples, n_features = X_scaled.shape
    
file_object = open(output_filename,'w')   
file_object.write('IterationNumber' + '           MSE' +'              MAE'+'              MAPE'+'              R2'+'              ExpVar'+'              MLSE'+'              MedAE'+'             Model' + '\n')
file_object.close()
    
# Hyperparameters
boosting = ['gbdt', 'dart']
num_leaves = [4, 8, 32] 
learning_rate = [0.01, 0.05, 0.1]
n_estimators = [1024]
bagging_freq = [1]
min_data_in_leaf = [5, 20, 50]
colsample_bytree = [0.05, 1.0]
min_child_weight = [1e-5, 1e-3, 1e-1, 1, 1e1]
#feature_fraction = [0.85, 0.95]     #  [0.85, 0.95] 
#bagging_fraction = [0.85, 0.95] 
min_gain_to_split = [0.0, 0.5, 1.0, 2.0] 
early_stopping_round= [250]
verbosity  = -1

data1 = []
iteration = 0

for bo in boosting:
    for nl in num_leaves:
        for lr in learning_rate:
            for ne in n_estimators:
                for bf in bagging_freq:
                    for mdl in min_data_in_leaf:
                        for cst in colsample_bytree:
                            try:
                                predict_loo = []
                                #lightGMB = LGBMRegressor()
                                lightGMB = LGBMRegressor(                                    
                                                        objective = 'regression',
                                                        metric = 'rmse',
                                                        boosting = bo,
                                                        num_leaves = nl,
                                                        learning_rate = lr,
                                                        n_estimators = ne,                                                                  
                                                        bagging_freq = bf,                                                                           
                                                        min_data_in_leaf = mdl,
                                                        colsample_bytree = cst,
                                                        verbosity = verbosity)
                                for train_index, test_index in loo.split(X_scaled):
                                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                                    y_train, y_test = y[train_index], y[test_index]
                                    
                                    lightGMB.fit(X_train, y_train.ravel())
                                    preds = lightGMB.predict(X_test)
                                    predict_loo.append(round(float(preds), 4))
                                
                                
                                predict_loo_tot = np.array(predict_loo)
                                                        
                                mse_lightGMB = np.reshape(mean_squared_error(y, predict_loo_tot), (1,1))
                                mae_lightGMB = np.reshape(mean_absolute_error(y, predict_loo_tot), (1,1))
                                mape_lightGMB = np.reshape(mean_absolute_percentage_error(y, predict_loo_tot), (1,1))
                                r2_lightGMB = np.reshape(r2_score(y, predict_loo_tot), (1,1))
                                expVar_lightGMB = np.reshape(explained_variance_score(y, predict_loo_tot), (1,1))
                                msle_lightGMB = np.reshape(mean_squared_log_error(y, predict_loo_tot), (1,1))
                                medAE_lightGMB = np.reshape(median_absolute_error(y, predict_loo_tot), (1,1))
                                                
                                lightGMB.fit(X, y.ravel())
                                predict_full = np.reshape(lightGMB.predict(X),(n_samples, 1))
                                mse_lightGMB_full = np.reshape(mean_squared_error(y, predict_full), (1,1))
                                mae_lightGMB_full = np.reshape(mean_absolute_error(y, predict_full), (1,1))
                                mape_lightGMB_full = np.reshape(mean_absolute_percentage_error(y, predict_full), (1,1))
                                r2_lightGMB_full = np.reshape(r2_score(y, predict_full), (1,1))
                                expVar_lightGMB_full = np.reshape(explained_variance_score(y, predict_full), (1,1))
                                msle_lightGMB_full = np.reshape(mean_squared_log_error(y, predict_full), (1,1))
                                medAE_lightGMB_full = np.reshape(median_absolute_error(y, predict_full), (1,1))
                                                
                                data = {'MSE': mse_lightGMB,
                                        'MAE': mae_lightGMB,
                                        'MAPE': mape_lightGMB,
                                        'R2': r2_lightGMB,
                                        'Explained Var': expVar_lightGMB,
                                        'MSLE': msle_lightGMB,
                                        'MedAE': medAE_lightGMB,
                                        'lightGMB_Regressor': lightGMB,
                                        'predicted values': predict_loo_tot}
                                
                                data1.append(data)
                                iteration = iteration + 1
                                print(iteration)
                                
                                if r2_lightGMB > 0.0:
                                    print("lightGMB_Regressor LOO:", mse_lightGMB, mae_lightGMB, mape_lightGMB, r2_lightGMB, expVar_lightGMB, msle_lightGMB, medAE_lightGMB)
                                    print("lightGMB_Regressor Full:", mse_lightGMB_full, mae_lightGMB_full, mape_lightGMB_full, r2_lightGMB_full, expVar_lightGMB_full, msle_lightGMB_full, medAE_lightGMB_full)
                                    print(lightGMB)
                                    
                                file_object = open(output_filename,'a')
                                file_object.write(repr(iteration) + '                   ' +
                                                  repr(round(float(data['MSE']), 5)) + '         ' +
                                                  repr(round(float(data['MAE']), 5)) + '         ' +
                                                  repr(round(float(data['MAPE']), 5)) + '         ' +
                                                  repr(round(float(data['R2']), 5)) + '          ' +
                                                  repr(round(float(data['Explained Var']), 5)) + '          ' +
                                                  repr(round(float(data['MSLE']), 5)) + '          ' +
                                                  repr(round(float(data['MedAE']), 5)) + '          ' +
                                                  "".join((str(data['lightGMB_Regressor']).replace("\n","")).split()) + '            ' +
                                                  str(data['predicted values'].reshape(1, n_samples)).replace("\n"," ")+ '\n' )
                                file_object.close()
                                
                            except:
                                print("Unsuccessful Model: ", lightGMB)
                                pass
                                        
                                        
maximum_r2 = []
minimum_mse = []
minimum_mae = []
minimum_mape = []
maximum_expVar = []
minimum_msle = []
minimum_medAE = []

for i in range(len(data1)):
    maximum_r2.append(round(float(data1[i]['R2']), 4))
    minimum_mse.append(round(float(data1[i]['MSE']), 4))
    minimum_mae.append(round(float(data1[i]['MAE']), 4))
    minimum_mape.append(round(float(data1[i]['MAPE']), 4))
    maximum_expVar.append(round(float(data1[i]['Explained Var']), 4))
    minimum_msle.append(round(float(data1[i]['MSLE']), 4))
    minimum_medAE.append(round(float(data1[i]['MedAE']), 4))
    
print('Largest R2 value:', np.max(maximum_r2))
print('Smallest MSE value:', np.min(minimum_mse))
print('Smallest MAE value:', np.min(minimum_mae))
print('Smallest MAPE value:', np.min(minimum_mape))
print('Largest Explained Var value:', np.max(maximum_expVar))
print('Smallest MSLE value:', np.min(minimum_msle))
print('Smallest MedAE value:', np.min(minimum_medAE))

print('Largest R2 index: ', np.where(maximum_r2 == np.max(maximum_r2)))
print('Smallest MSE index: ', np.where(minimum_mse == np.min(minimum_mse)))
print('Smallest MAE index: ', np.where(minimum_mae == np.min(minimum_mae)))
print('Smallest MAPE index: ', np.where(minimum_mape == np.min(minimum_mape)))
print('Largest Explained Var index: ', np.where(maximum_expVar == np.max(maximum_expVar)))
print('Smallest MSLE index: ', np.where(minimum_msle == np.min(minimum_msle)))
print('Smallest MedAE index: ', np.where(minimum_medAE == np.min(minimum_medAE)))


file_object = open(output_filename, 'a')
file_object.write('R2 : ' + repr(np.max(maximum_r2)) + '\n' +
                  'MSE : ' + repr(np.min(minimum_mse)) + '\n' +
                  'MAE : ' + repr(np.min(minimum_mae)) + '\n' +
                  'MAPE : ' + repr(np.min(minimum_mape)) + '\n' +
                  'Explained Var : ' + repr(np.max(maximum_expVar)) + '\n' +
                  'MSLE : ' + repr(np.min(minimum_msle)) + '\n' +
                  'MedAE : ' + repr(np.min(minimum_medAE)) + '\n' +
                  'R2 indices : ' + repr(np.where(maximum_r2 == np.max(maximum_r2))) + '\n' +
                  'MSE indices : ' + repr(np.where(minimum_mse == np.min(minimum_mse))) + '\n' +
                  'MAE indices : ' + repr(np.where(minimum_mae == np.min(minimum_mae))) + '\n' +
                  'MAPE indices : ' + repr(np.where(minimum_mape == np.min(minimum_mape))) + '\n' +
                  'Explained Var indices : ' + repr(np.where(maximum_expVar == np.max(maximum_expVar))) + '\n'
                  'MSLE indices : ' + repr(np.where(minimum_msle == np.min(minimum_msle))) + '\n'+
                  'MedAE indices : ' + repr(np.where(minimum_medAE == np.min(minimum_medAE)))) 
file_object.close()

print('End of Simulation')
