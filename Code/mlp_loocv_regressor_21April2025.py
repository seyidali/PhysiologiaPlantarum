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
from sklearn.neural_network import MLPRegressor
# Preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, QuantileTransformer, PowerTransformer, Binarizer
import warnings

warnings.filterwarnings("ignore")

input_filename = 'Data.xlsx'
output_filename = 'mlp_Data(SurvivalPercetage).txt'

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
hidden_layer_sizes = [(25, 25, 25), (50, 50, 50)]  
activation = ['relu', 'tanh', 'logistic', 'identity']
solver = ['adam', 'lbfgs', 'sgd'] 
alpha = 0.0001 
batch_size = 'auto' 
learning_rate = ['constant', 'adaptive', 'invscaling']
learning_rate_init = [0.001, 0.01] 
power_t = 0.5 
max_iter = 5000   # 5000 
shuffle = False 
random_state = None 
tol = 0.0001 
verbose = False 
warm_start = False 
momentum = [0.9, 0.3] 
nesterovs_momentum = True 
early_stopping = False 
validation_fraction = 0.1 
beta_1 = [0.9, 0.1] 
beta_2 = [0.999, 0.1] 
#epsilon = [1e-08, 1e-04]  
epsilon = [1e-08]           # epsilon = 1e-08 ok
n_iter_no_change = 10 
max_fun = 15000

data1 = []
iteration = 0

for hls in hidden_layer_sizes:
    for ac  in activation:
        for sl  in solver:
            for lr  in learning_rate:
                for lri in learning_rate_init:
                    for mt  in momentum:
                        for bt1 in beta_1:
                            for bt2 in beta_2:
                                for ep  in epsilon:
                                    try:
                                        
                                        predict_loo = []
                                
                                        mlp_model = MLPRegressor(
                                                                hidden_layer_sizes = hls, 
                                                                activation = ac, 
                                                                solver = sl, 
                                                                alpha = alpha, 
                                                                batch_size = batch_size, 
                                                                learning_rate = lr, 
                                                                learning_rate_init = lri, 
                                                                power_t = power_t, 
                                                                max_iter = max_iter, 
                                                                shuffle = shuffle, 
                                                                random_state = random_state, 
                                                                tol = tol, 
                                                                verbose = verbose, 
                                                                warm_start = warm_start, 
                                                                momentum = mt, 
                                                                nesterovs_momentum = nesterovs_momentum, 
                                                                early_stopping = early_stopping, 
                                                                validation_fraction = validation_fraction, 
                                                                beta_1 = bt1, 
                                                                beta_2 = bt2, 
                                                                epsilon = ep, 
                                                                n_iter_no_change = n_iter_no_change, 
                                                                max_fun = max_fun)
                                        
                                        for train_index, test_index in loo.split(X_scaled):
                                            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                                            y_train, y_test = y[train_index], y[test_index]
                                         
                                        
                                            mlp_model.fit(X_train, y_train.ravel())
                                            preds = mlp_model.predict(X_test)
                                            predict_loo.append( round(float(preds), 4) )
                                            
                                        predict_loo_tot = np.array(predict_loo)    
                                    
                                        mse_mlp = np.reshape( mean_squared_error(y, predict_loo_tot), (1, 1) )
                                        mae_mlp = np.reshape( mean_absolute_error(y, predict_loo_tot), (1, 1) )
                                        mape_mlp = np.reshape( mean_absolute_percentage_error(y, predict_loo_tot), (1, 1) )
                                        r2_mlp = np.reshape( r2_score(y, predict_loo_tot), (1, 1) )
                                        expVar_mlp = np.reshape(explained_variance_score(y, predict_loo_tot), (1,1))
                                        msle_mlp = np.reshape(mean_squared_log_error(y, predict_loo_tot), (1,1))
                                        medAE_mlp = np.reshape(median_absolute_error(y, predict_loo_tot), (1,1))
                                    
                                         
                                        mlp_model.fit(X, y.ravel())
                                        predict_full = np.reshape( mlp_model.predict(X),(n_samples, 1) )
                                        mse_mlp_full = np.reshape( mean_squared_error(y, predict_full), (1, 1) )
                                        mae_mlp_full = np.reshape( mean_absolute_error(y, predict_full), (1, 1) )
                                        mape_mlp_full = np.reshape( mean_absolute_percentage_error(y, predict_full), (1, 1) )
                                        r2_mlp_full = np.reshape( r2_score(y, predict_full), (1, 1) )
                                        expVar_mlp_full = np.reshape(explained_variance_score(y, predict_full), (1,1))
                                        msle_mlp_full = np.reshape(mean_squared_log_error(y, predict_full), (1,1))
                                        medAE_mlp_full = np.reshape(median_absolute_error(y, predict_full), (1,1))
                                    
                                        data = {'MSE': mse_mlp, 
                                                'MAE': mae_mlp,
                                                'MAPE': mape_mlp,  
                                                'R2': r2_mlp,
                                                'Explained Var': expVar_mlp,
                                                'MSLE': msle_mlp,
                                                'MedAE': medAE_mlp,
                                                'mlpRegressor': mlp_model, 
                                                'predicted values': predict_loo_tot}
                                    
                                        iteration = iteration + 1
                                        print(iteration)
                                    
                                        data1.append(data)
                                        #results_iter = pd.DataFrame(data1)
                                    
                                    
                                        if r2_mlp > 0.0:
                                            print("mlpRegressor LOO:", "mse:", mse_mlp, mae_mlp, mape_mlp, r2_mlp, expVar_mlp, msle_mlp, medAE_mlp)
                                            print("mlpRegressor Full:", "mse full:", mse_mlp_full, mae_mlp_full, mape_mlp_full, r2_mlp_full, expVar_mlp_full, msle_mlp_full, medAE_mlp_full)
                                            print(mlp_model)
                                    
                                        
                                        file_object = open(output_filename, 'a')     
                                        file_object.write(repr(iteration) + '                   ' + 
                                                          repr( round(float(data['MSE']), 5) ) + '         ' +
                                                          repr(round(float(data['MAE']), 5)) + '         ' +
                                                          repr(round(float(data['MAPE']), 5)) + '         ' +
                                                          repr( round(float(data['R2']), 5) ) + '          ' +
                                                          repr(round(float(data['Explained Var']), 5)) + '          ' +
                                                         repr(round(float(data['MSLE']), 5)) + '          ' +
                                                         repr(round(float(data['MedAE']), 5)) + '          ' +
                                                          "".join((str(data['mlpRegressor']).replace("\n","")).split()) + '            '+
                                                          str(data['predicted values'].reshape(1, n_samples)).replace("\n"," ")+ '\n' )
                                        file_object.close()
                                    
                                    except:
                                        print("Unsuccessful Model: ", mlp_model)
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
