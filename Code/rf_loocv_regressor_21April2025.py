# -*- coding: utf-8 -*-
"""
Created on Monday 21 April 2025
@author: Seyid Amjad Ali
"""
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score, max_error,mean_squared_log_error, median_absolute_error, mean_poisson_deviance, mean_gamma_deviance
# from sklearn.metrics import mean_absolute_error, max_error, mean_absolute_percentage_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
# preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, QuantileTransformer, PowerTransformer, Binarizer
import warnings

warnings.filterwarnings("ignore")

input_filename = 'Data.xlsx'
output_filename = 'rf_Data(SurvivalPercetage).txt'

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

file_object = open(output_filename, 'w')   
file_object.write('IterationNumber' + '           MSE' +'              MAE'+'              MAPE'+'              R2'+'              ExpVar'+'              MLSE'+'              MedAE'+'             Model' + '\n')
file_object.close()

# Hyperparameters
bootstrap = [True, False] 
ccp_alpha = [0.0]
criterion = ['mse']
#criterion = ['squared_error']
max_depth = [10, 20, 30] 
max_features = ['auto', 'sqrt', 'log2'] 
max_leaf_nodes = None
max_samples = None 
min_impurity_decrease = [0.0]
#min_impurity_split = None 
min_samples_leaf = [1, 2, 4]
min_samples_split = [2, 5, 10] 
min_weight_fraction_leaf = [0.0, 1, 2]
n_estimators = [10, 100, 200, 400] 
n_jobs = None 
oob_score = False
random_state = None 
verbose = False 
warm_start = False

data1 = []
iteration = 0

for bs in bootstrap:
    for ca in  ccp_alpha:
        for cr in  criterion:
              for mf in max_features:
                      for mi in min_impurity_decrease:
                        for msl in min_samples_leaf:
                          for mss in min_samples_split:
                            for mw in min_weight_fraction_leaf:
                              for ne in n_estimators:
                                for md in max_depth:
                                    try:
                                                                                                      
                                        predict_loo = []    
                                          
                                        rf = RandomForestRegressor(
                                                                  bootstrap = bs,
                                                                  ccp_alpha = ca,
                                                                  criterion = cr,
                                                                  max_depth = md,
                                                                  max_features = mf,
                                                                  max_leaf_nodes = max_leaf_nodes,
                                                                  max_samples = max_samples,
                                                                  min_impurity_decrease = mi,
                                                                  #min_impurity_split = min_impurity_split,
                                                                  min_samples_leaf = msl,
                                                                  min_samples_split = mss,
                                                                  min_weight_fraction_leaf = mw,
                                                                  n_estimators = ne,
                                                                  n_jobs = n_jobs,
                                                                  oob_score = oob_score,
                                                                  random_state = random_state,
                                                                  verbose = verbose,
                                                                  warm_start = warm_start)
                                        
                                        for train_index, test_index in loo.split(X_scaled):
                                            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                                            y_train, y_test = y[train_index], y[test_index]
                                        
                                            rf.fit(X_train, y_train.ravel())
                                            preds = rf.predict(X_test)
                                            predict_loo.append(round(float(preds), 4))
                                            
                                        predict_loo_tot = np.array(predict_loo)
                                          
                                        mse_rf = np.reshape(mean_squared_error(y, predict_loo_tot), (1,1))
                                        mae_rf = np.reshape(mean_absolute_error(y, predict_loo_tot), (1,1))
                                        mape_rf = np.reshape(mean_absolute_percentage_error(y, predict_loo_tot), (1,1))
                                        r2_rf = np.reshape(r2_score(y, predict_loo_tot), (1,1))
                                        expVar_rf = np.reshape(explained_variance_score(y, predict_loo_tot), (1,1))
                                        msle_rf = np.reshape(mean_squared_log_error(y, predict_loo_tot), (1,1))
                                        medAE_rf = np.reshape(median_absolute_error(y, predict_loo_tot), (1,1))
                                        
                                        rf.fit(X, y.ravel())
                                        predict_full = np.reshape(rf.predict(X),(n_samples, 1))
                                        mse_rf_full = np.reshape(mean_squared_error(y, predict_full), (1,1))
                                        mae_rf_full = np.reshape(mean_absolute_error(y, predict_full), (1,1))
                                        mape_rf_full = np.reshape(mean_absolute_percentage_error(y, predict_full), (1,1))
                                        r2_rf_full = np.reshape(r2_score(y, predict_full), (1,1))
                                        expVar_rf_full = np.reshape(explained_variance_score(y, predict_full), (1,1))
                                        msle_rf_full = np.reshape(mean_squared_log_error(y, predict_full), (1,1))
                                        medAE_rf_full = np.reshape(median_absolute_error(y, predict_full), (1,1))
                                        
                                        
                                        data = {'MSE': mse_rf,
                                                'MAE': mae_rf,
                                                'MAPE': mape_rf,
                                                'R2': r2_rf,
                                                'Explained Var': expVar_rf,
                                                'MSLE': msle_rf,
                                                'MedAE': medAE_rf,
                                                'rfRegressor': rf,
                                                'predicted values': predict_loo_tot}
                                        
                                        data1.append(data)
                                        
                                        iteration = iteration + 1
                                        print(iteration)
                                        
                                        if r2_rf > 0.0:
                                            print("rfRegressor LOO:", mse_rf, mae_rf, mape_rf, r2_rf, expVar_rf, msle_rf, medAE_rf)
                                            print("rfRegressor Full:", mse_rf_full, mae_rf_full, mape_rf_full, r2_rf_full, expVar_rf_full, msle_rf_full, medAE_rf_full)
                                            print(rf)
                                        
                                        file_object = open(output_filename,'a')     
                                        file_object.write(repr(iteration) + '                   ' + 
                                                         repr(round(float(data['MSE']), 5)) + '         ' +
                                                         repr(round(float(data['MAE']), 5)) + '         ' +
                                                         repr(round(float(data['MAPE']), 5)) + '         ' +
                                                         repr(round(float(data['R2']), 5)) + '          ' +
                                                         repr(round(float(data['Explained Var']), 5)) + '          ' +
                                                         repr(round(float(data['MSLE']), 5)) + '          ' +
                                                         repr(round(float(data['MedAE']), 5)) + '          ' +
                                                          "".join((str(data['rfRegressor']).replace("\n","")).split()) + '            '+
                                                          str(data['predicted values'].reshape(1, n_samples)).replace("\n"," ")+ '\n' )
                                        file_object.close() 
                                    
                                    except:
                                        print("Unsuccessful Model: ", rf)
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
