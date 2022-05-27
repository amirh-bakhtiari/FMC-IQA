# Enable Intel(R) Extension for Scikit-learn
# from sklearnex import patch_sklearn
# patch_sklearn()
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import DatasetHandler as dh


def nn_regressor(input_size: int):
    '''Use a multi layer neural network as a regressor
    
    :param input_size: dimension of input samples
    :return: regressor model
    '''
    
    model = models.Sequential()
    model.add(Dense(4096, kernel_initializer='normal', activation='relu', input_shape=(input_size,)))#layer 1
    model.add(BatchNormalization())
    model.add(Dense(2048, kernel_initializer='normal', activation='relu'))#layer 2
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))#layer 3
    model.add(Dropout(0.2))

    model.add(Dense(512, kernel_initializer='normal', activation='relu'))#layer 4
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))#layer 5
    model.add(Dropout(0.2))

    model.add(Dense(128, kernel_initializer='normal', activation='relu'))#layer 6
    model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer='normal', activation='linear'))#layer 7
    
    
    callback = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=1,mode="min",baseline=None,restore_best_weights=True)
    model.compile(optimizer= optimizers.Adam(learning_rate=4e-5),loss='mean_absolute_error', metrics=['mean_absolute_error'])
    
    return model
    
    

def live_dataset_regression(X, y):
    '''Train an SVR Using the video level features and their corresponding scores,
       predict the scores of test videos using the trained SVR. Finally calculate the
       Spearman and Pearson correlation for target and predicted scores.
       
    :param X: an array of video level features of all videos in the dataset
    :param y: an array scores of all videos in the dataset
    '''
    
    video_data = '/media/amirh/Programs/Projects/VQA_Datasets/LIVE_SD/live_video_quality_seqs.txt'
    dmos_data = '/media/amirh/Programs/Projects/VQA_Datasets/LIVE_SD/live_video_quality_data.txt'
    # Get the list of video sequences
    video_list, _ = dh.get_live_info(video_data, dmos_data)
    
    # Turn y into a 2D array to match StandardScalar() input
    y = np.array(y).reshape(-1, 1)
    
    # There are 10 pristine videos in LIVE VQA dataset and 15 different distorted videos are made from each,
    # totally 150 videos. Divide the video features into 10 groups in orders, so that the same group will not
    # appear in two different folds
    groups = np.empty(150, dtype='u1')
    video_groups = []
    
    for i in range(0, 150, 15):
        groups[i: i + 15] = i / 15
        # Add the name of the first video in each group
        video_groups.append(video_list[i])
               
    gss = GroupShuffleSplit(n_splits=50, train_size=0.8)
    
    SROCC_coef, SROCC_p, PLCC = [], [], []
    
    for train_idx, test_idx in gss.split(X, y, groups):
        # Split train validation set
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # print the video names fallen into test split
        for group in set(groups[test_idx]):
            print(f'video group {group} = {video_groups[group]}', end=',  ')
        
        # Feature scaling
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        y_train = sc_y.fit_transform(y_train)
        
        # Set the regressor to SVR
        regressor = SVR(kernel='rbf')
        
        # Train the regresoor model
        regressor.fit(X_train, y_train.squeeze())
        
        # Predict the scores for X_test videos features
        y_pred = regressor.predict(X_test)
        # Turn y_pred into a 2D array to match StandardScalar() input
        y_pred = y_pred.reshape(-1, 1)
        
        y_test = y_test.reshape(-1, 1)
        
        # Inverse transform the predicted values to get the real values
        y_pred = sc_y.inverse_transform(y_pred)
        
        # Calculate the Spearman rank-order correlation
        coef, p = spearmanr(y_test.squeeze(), y_pred.squeeze())
        
        # Calculate the Pearson correlation
        corr, _ = pearsonr(y_test.squeeze(), y_pred.squeeze())
        
        SROCC_coef.append(coef)
        SROCC_p.append(p)
        PLCC.append(corr)
        
        print(f'\nSpearman correlation = {coef:.4f} with p = {p:.4f},  Pearson correlation = {corr:.4f}')
        print('*' * 50)
        
    # set the precision of the output for numpy arrays & suppress the use of scientific notation for small numbers
    with np.printoptions(precision=4, suppress=True):
        print(f'SROCC_coef = {np.array(SROCC_coef)}')
        print(f'SROCC_p = {np.array(SROCC_p)}')
        print(f'PLCC = {np.array(PLCC)}')
        
        
        
        
    
    