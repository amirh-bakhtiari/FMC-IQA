import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.mstats import spearmanr
from scipy.stats.mstats import pearsonr
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVR
from sklearnex.svm import SVR

import DatasetHandler as dh

try:
    import tensorflow
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras import optimizers

    def nn_regressor(X_train, y_train):
        '''Use a multi layer neural network as a regressor

        :param X_train: input samples' features
        :param y_train: target scores
        :return: a trained regressor model and the training history
        '''
        input_size = X_train.shape[1]

        model = Sequential()
        model.add(Dense(4096, kernel_initializer='normal', activation='relu', input_shape=(input_size,)))#layer 1
        model.add(BatchNormalization())
        model.add(Dense(2048, kernel_initializer='normal', activation='relu'))#layer 2
        # model.add(BatchNormalization())
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

        callback=tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=5,
                                                          verbose=1,mode="min",baseline=None,
                                                          restore_best_weights=True)
        model.compile(optimizer= optimizers.Adam(learning_rate=4e-5),loss='mean_squared_error',
                      metrics=['mean_squared_error'])

        history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1,
                            verbose=0, callbacks=[callback])

        return model, history
except ModuleNotFoundError:
    pass
    

def live_dataset_regression(X, y, regression_method='svr'):
    '''Train an SVR Using the video level features and their corresponding scores from
       LIVE VQA dataset, predict the scores of test videos using the trained SVR. 
       Finally calculate the SROCC & PLCC.
       
    :param X: an array of video level features of all videos in the dataset
    :param y: an array scores of all videos in the dataset
    :param regression_method: 'svr' for SVR or 'nn' for multi layer neural network
    :return: SROCC_coef, SROCC_p, PLCC
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
        
        if regression_method == 'svr':
            # Set the regressor to SVR
            regressor = SVR(kernel='rbf', epsilon=0.3)
            # Train the regresoor model
            regressor.fit(X_train, y_train.squeeze())
        elif regression_method == 'nn':
            # Set the regressor to neural network
            tensorflow.keras.backend.clear_session()
            regressor, history = nn_regressor(X_train, y_train.squeeze())

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
        
    return SROCC_coef, SROCC_p, PLCC
        
        
        
def konvid1k_dataset_regression(X, y, Xc=None, yc=None, regression_method='svr'):
    '''Train an SVR Using the video level features and their corresponding scores from
       Konvid1k VQA dataset, predict the scores of test videos using the trained SVR. 
       Finally calculate the SROCC & PLCC.
    
    :param X: an array of video level features of all videos in the dataset
    :param y: an array scores of all videos in the dataset
    :param regression_method: 'svr' for SVR or 'nn' for multi layer neural network
    :return: SROCC_coef, SROCC_p, PLCC 
    '''
    # Turn y into a 2D array to match StandardScalar() input
    y = np.array(y).reshape(-1, 1)
    
    SROCC_coef, SROCC_p, PLCC = [], [], []
    CROSS_SROCC, CROSS_PLCC = None, None
    # if cross dataset validation is required
    if Xc is not None:
        CROSS_SROCC, CROSS_PLCC = [], []
        # Turn y into a 2D array to match StandardScalar() input
        # yc = np.array(yc).reshape(-1, 1)
        # sc = StandardScaler()
        # ycs = sc.fit_transform(yc)
        ycs = yc / 100.0
    
    # Repeat K-fold cross validation 10 times
    for _ in range(20):
        
        # Use K-Fold Cross Validation for evaluation
        kfold = KFold(n_splits=5, shuffle=True)
        
        for train_idx, test_idx in kfold.split(X):
            # print(f'Test index = {test_idx}')
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Feature Scaling
            sc_X = StandardScaler()
            sc_y = StandardScaler()
            
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            y_train = sc_y.fit_transform(y_train)
            
            if regression_method.lower() == 'svr':
                # Set the regressor to SVR
                regressor = SVR(kernel='rbf', epsilon=0.3)
                # Train the SVR
                regressor.fit(X_train, y_train.squeeze())
            elif regression_method.lower() == 'nn':
                # Set the regressor to neural network
                tensorflow.keras.backend.clear_session()
                regressor, history = nn_regressor(X_train, y_train.squeeze())
                
            # Predict the scores for X_test videos features
            y_pred = regressor.predict(X_test)
            
            # Turn y_pred into a 2D array to match StandardScalar() input
            y_pred = y_pred.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            # Reverse the transform to get the real y_pred
            y_pred = sc_y.inverse_transform(y_pred)
            
            # Calculate the Spearman rank-order correlation
            coef, p = spearmanr(y_test.squeeze(), y_pred.squeeze())

            # Calculate the Pearson correlation
            corr, _ = pearsonr(y_test.squeeze(), y_pred.squeeze())
          
            SROCC_coef.append(coef)
            SROCC_p.append(p)
            PLCC.append(corr)
                        
            # print(f'Target Mos = {y_test.squeeze()}')
            # print(f'Predicted Mos = {y_pred.squeeze()}')
            print(f'\nSpearman correlation = {coef:.4f} with p = {p:.4f},  Pearson correlation = {corr:.4f}')
            print('-' * 50)
            
            # if cross dataset validation is required
            if Xc is not None:
                Xcs = sc_X.transform(Xc)
                
                yc_pred = regressor.predict(Xcs)
                
                # Turn y_pred into a 2D array to match StandardScalar() input
                yc_pred = yc_pred.reshape(-1, 1)
                # Reverse the transform to get the real yc_pred
                yc_pred = sc_y.inverse_transform(yc_pred)
                
                yc_pred -= 1
                yc_pred /= 4.0
              
                # Calculate the Spearman rank-order correlation
                coef_c, pc = spearmanr(ycs.squeeze(), yc_pred.squeeze())
                # Calculate the Pearson correlation
                corr_c, _ = pearsonr(ycs.squeeze(), yc_pred.squeeze())
                
                CROSS_SROCC.append(coef_c)
                CROSS_PLCC.append(corr_c)
                
                print(f'\nCross Dataset SROCC = {coef_c:.4f} with p = {pc:.4f}, Cross Dataset PLCC = {corr_c:.4f}')
                print('*' * 50)
                 
            
                # Plot the correlation between ground-truth and predicted scores             
                sns.set(style='darkgrid')
                scatter_plot = sns.relplot(x=yc.squeeze(), y=yc_pred.squeeze() * 100,
                                           kind='scatter', height=7, aspect=1.2, palette='coolwarm').set(
                                           xlabel='Cross Dataset Score', ylabel='Predicted Score');
                scatter_plot.savefig(f'plots/{len(SROCC_coef)}_{abs(coef):.4f}_c.png')
                
            # Plot the correlation between ground-truth and predicted scores             
            sns.set(style='darkgrid')
            scatter_plot = sns.relplot(x=y_test.squeeze(), y=y_pred.squeeze(),
                                       kind='scatter', height=7, aspect=1.2, palette='coolwarm').set(
                                       xlabel='Ground-truth MOS', ylabel='Predicted Score');
            
            
            scatter_plot.savefig(f'plots/{len(SROCC_coef)}_{abs(coef):.4f}.png')
        
    return SROCC_coef, SROCC_p, PLCC, CROSS_SROCC, CROSS_PLCC
    
        
def regression(X, y, Xc=None, yc=None, regression_method='svr', dataset='LIVE'):
    '''Get video features and scores and call the respective regressor according to the dataset name
    
    :param X: an array of video level features of all videos in the dataset
    :param y: an array scores of all videos in the dataset
    :param Xc: an array of video level features of all videos in the cross dataset
    :param yc: an array scores of all videos in the cross dataset
    :param regression_method: 'svr' for SVR or 'nn' for multi layer neural network
    '''
    
    print(f'{X.shape = } {y.shape = }')
    if Xc is not None:
        print(f'{Xc.shape = } {yc.shape = }')
    
    if dataset.lower() == 'live':
        SROCC_coef, SROCC_p, PLCC = live_dataset_regression(X, y, regression_method='svr')
    elif (dataset.lower() == 'konvid1k' or dataset.lower() == 'koniq10k' or dataset.lower() == 'clive'):
        SROCC_coef, SROCC_p, PLCC, CROSS_SROCC, CROSS_PLCC= konvid1k_dataset_regression(X, y, Xc, yc,
                                                                                        regression_method='svr')
        
    # set the precision of the output for numpy arrays & suppress the use of scientific notation for small numbers
    with np.printoptions(precision=4, suppress=True):
        print(f'SROCC_coef = {np.array(SROCC_coef)}')
        print(f'SROCC_coefs average = {np.mean(np.abs(SROCC_coef))}')
        print(f'SROCC_coefs median = {np.median(np.abs(SROCC_coef))}')
        print(f'SROCC_p = {np.array(SROCC_p)}')
        print(f'PLCC = {np.array(PLCC)}')
        print(f'PLCCs average = {np.mean(np.abs(PLCC))}')
        print(f'PLCCs median = {np.median(np.abs(PLCC))}')
        if Xc is not None:
            print(f'SROCC_coef = {np.array(CROSS_SROCC)}')
            print(f'SROCC_coefs average = {np.mean(np.abs(CROSS_SROCC))}')
            print(f'SROCC_coefs median = {np.median(np.abs(CROSS_SROCC))}')
            print(f'PLCC = {np.array(CROSS_PLCC)}')
            print(f'PLCCs average = {np.mean(np.abs(CROSS_PLCC))}')
            print(f'PLCCs median = {np.median(np.abs(CROSS_PLCC))}')
            
        
                
            
            
            
            
            
            
            
    