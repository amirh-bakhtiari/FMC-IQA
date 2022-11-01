import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.mstats import spearmanr
from scipy.stats.mstats import pearsonr
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# from sklearnex.svm import SVR

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
    
def normalize_cross_scores(cross_dataset, yc):
    '''Normalize the cross dataset scores to be in the range [0, 1]
    '''
    
    # Normalize the cross dataset scores
    if cross_dataset == 'clive':
        ycn = yc / 100.0
    elif cross_dataset == 'koniq10k' or dataset == 'kadid10k':
        ycn = yc - 1
        ycn /= 4.0
    
    return ycn

def calc_correlation(y_gt, y_pred, sc, dataset=None, delimiter='-'):
    '''Calculate SROCC with p and PLCC
    '''
    # Turn y_pred into a 2D array to match StandardScalar() input
    y_pred = y_pred.reshape(-1, 1)
    # Inverse transform the predicted values to get the real values
    y_pred = sc.inverse_transform(y_pred)
    
    # If dataset name has been given, normalize the results to compare with the cross dataset scores
    if dataset == 'koniq10k' or dataset == 'kadid10k':
        y_pred -= 1
        y_pred /= 4.0
    elif dataset == 'clive':
        y_pred /= 100.0

    # Calculate the Spearman rank-order correlation
    srocc, p = spearmanr(y_gt.squeeze(), y_pred.squeeze())

    # Calculate the Pearson correlation
    plcc, _ = pearsonr(y_gt.squeeze(), y_pred.squeeze())
    
    text = f'Spearman correlation = {srocc:.4f} with p = {p:.4f},  Pearson correlation = {plcc:.4f}\n'
    with open('correlation.txt', 'a') as writer:
        writer.write(text)
        writer.write(delimiter * 70 + '\n')
    
    return srocc, p, plcc

def plot_correlation(y_gt, y_pred, num, srocc, cross_dataset=None):
    '''Plot SROCC of predicted and ground-truth scores
    '''
    
    if cross_dataset == 'clive':
        y_pred *= 100
        file_path = f'plots/{num}_{abs(srocc):.4f}_{cross_dataset}.png'
    elif cross_dataset == 'koniq10k' or cross_dataset == 'kadid10k':
        y_pred *= 4
        y_pred += 1
        file_path = f'plots/{num}_{abs(srocc):.4f}_{cross_dataset}.png'
    else:
        file_path = f'plots/{num}_{abs(srocc):.4f}.png'
        
    # Plot the correlation between ground-truth and predicted scores             
    sns.set(style='darkgrid')
    scatter_plot = sns.relplot(x=y_gt.squeeze(), y=y_pred.squeeze(),
                               kind='scatter', height=7, aspect=1.2, palette='coolwarm').set(
                               xlabel='Ground-truth MOS', ylabel='Predicted Score');
     
    plt.close()
    scatter_plot.savefig(file_path)

def synthetic_dataset_regression(X, y, dist_per_ref, Xc=None, yc=None, dataset='kadid0k',
                                 cross_dataset=None, regression_method='svr'):
    '''Train a regressor Using the image/video features and their corresponding scores from
       a synthetically distorted IQA/VQA dataset, predict the scores of test data. 
       Finally calculate the SROCC & PLCC.
       
    :param X: an array of features of all images/videos in the dataset
    :param y: an array scores of all images/videos in the dataset
    :param regression_method: 'svr' for SVR or 'nn' for multi layer neural network
    :return: SROCC_coef, SROCC_p, PLCC
    '''
            
    # Turn y into a 2D array to match StandardScalar() input
    y = np.array(y).reshape(-1, 1)
    
    # There are some pristine images/videos in a synthetic dataset and len() different distorted videos are made from each,
    # totally 150 videos. Divide the video features into 10 groups in orders, so that the same group will not
    # appear in two different folds
    groups = np.empty(len(y), dtype='u1')
    
    for i in range(0, len(y), dist_per_ref):
        groups[i: i + dist_per_ref] = i // dist_per_ref
               
    gss = GroupShuffleSplit(n_splits=100, train_size=0.8)
    
    SROCC_coef, SROCC_p, PLCC = [], [], []
    
    # if cross dataset validation is required
    if Xc is not None:
        CROSS_SROCC, CROSS_PLCC = [], []
        ycn = normalize_cross_scores(cross_dataset, yc)
    else:
        CROSS_SROCC, CROSS_PLCC = None, None
   
    for train_idx, test_idx in gss.split(X, y, groups):
        # Split train validation set
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

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
        
        srocc, p, plcc = calc_correlation(y_test, y_pred, sc_y)
        
        SROCC_coef.append(srocc)
        SROCC_p.append(p)
        PLCC.append(plcc)
                            
        # if cross dataset validation is required
        if Xc is not None:
            Xcs = sc_X.transform(Xc)

            yc_pred = regressor.predict(Xcs)

            srocc, _, plcc = calc_correlation(ycn, yc_pred, sc_y, dataset, '*')
            
            CROSS_SROCC.append(srocc)
            CROSS_PLCC.append(plcc)

            # Plot the correlation between ground-truth and predicted scores             
            plot_correlation(yc, yc_pred, len(SROCC_coef), srocc, cross_dataset)
        
        # Plot the correlation between ground-truth and predicted scores             
        plot_correlation(y_test, y_pred, len(SROCC_coef), srocc)
        
    return SROCC_coef, SROCC_p, PLCC, CROSS_SROCC, CROSS_PLCC

def authentic_dataset_regression(X, y, Xc=None, yc=None, dataset='koniq10k', cross_dataset=None, regression_method='svr'):
    '''Train a regressor Using the features and their corresponding scores from
       an authentically distorted IQA/VQA dataset, predict the scores of test data. 
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
        ycn = normalize_cross_scores(cross_dataset, yc)
    else:
        CROSS_SROCC, CROSS_PLCC = None, None
  
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
            
            srocc, p, plcc = calc_correlation(y_test, y_pred, sc_y)
          
            SROCC_coef.append(srocc)
            SROCC_p.append(p)
            PLCC.append(plcc)
                    
            # if cross dataset validation is required
            if Xc is not None:
                Xcs = sc_X.transform(Xc)
                
                yc_pred = regressor.predict(Xcs)
                
                srocc, _, plcc = calc_correlation(ycn, yc_pred, sc_y, dataset, '*')
                
                CROSS_SROCC.append(srocc)
                CROSS_PLCC.append(plcc)
            
                # Plot the correlation between ground-truth and predicted scores             
                plot_correlation(yc, yc_pred, len(SROCC_coef), srocc, cross_dataset)
            
            # Plot the correlation between ground-truth and predicted scores             
            plot_correlation(y_test, y_pred, len(SROCC_coef), srocc)
                                
    return SROCC_coef, SROCC_p, PLCC, CROSS_SROCC, CROSS_PLCC
    
        
def regression(X, y, Xc=None, yc=None, regression_method='svr', dataset='koniq10k', cross_dataset=None):
    '''Get video features and scores and call the respective regressor according to the dataset name
    
    :param X: an array of video level features of all videos in the dataset
    :param y: an array scores of all videos in the dataset
    :param Xc: an array of video level features of all videos in the cross dataset
    :param yc: an array scores of all videos in the cross dataset
    :param regression_method: 'svr' for SVR or 'nn' for multi layer neural network
    '''
    
    dataset = dataset.lower()
    # Total number of distorted images/videos per each reference for synthetic datasets
    synth_dist_per_ref = {'tid2013': 120,
                          'kadid10k': 125,
                          'scid': 45}
    
    # assert bool(Xc) == bool(cross_dataset), 'either features value or name of the cross dataset has not been provided.'
    
    print(f'{X.shape = } {y.shape = }')
    if Xc is not None:
        print(f'{Xc.shape = } {yc.shape = }')
    
    if dataset in synth_dist_per_ref:
        SROCC_coef, SROCC_p, PLCC, CROSS_SROCC, CROSS_PLCC = synthetic_dataset_regression(X, y,
                                                                                          synth_dist_per_ref[dataset],
                                                                                          Xc, yc, dataset,
                                                                                          cross_dataset,
                                                                                          regression_method)
    else:
        SROCC_coef, SROCC_p, PLCC, CROSS_SROCC, CROSS_PLCC = authentic_dataset_regression(X, y, Xc, yc,
                                                                                          dataset, 
                                                                                          cross_dataset,
                                                                                          regression_method)
    with open('correlation.txt', 'a') as writer:    
    # set the precision of the output for numpy arrays & suppress the use of scientific notation for small numbers
        with np.printoptions(precision=4, suppress=True):
            writer.write(f'SROCC_coef = {np.array(SROCC_coef)}\n')
            writer.write(f'SROCC_coefs average = {np.mean(np.abs(SROCC_coef)): .4f}\n')
            writer.write(f'SROCC_coefs median = {np.median(np.abs(SROCC_coef)): .4f}\n')
            writer.write(f'SROCC_p = {np.array(SROCC_p)}\n')
            writer.write(f'PLCC = {np.array(PLCC)}\n')
            writer.write(f'PLCCs average = {np.mean(np.abs(PLCC)): .4f}\n')
            writer.write(f'PLCCs median = {np.median(np.abs(PLCC)): .4f}\n')
            if Xc is not None:
                writer.write(f'SROCC_coef = {np.array(CROSS_SROCC)}\n')
                writer.write(f'SROCC_coefs average = {np.mean(np.abs(CROSS_SROCC)): .4f}\n')
                writer.write(f'SROCC_coefs median = {np.median(np.abs(CROSS_SROCC)): .4f}\n')
                writer.write(f'PLCC = {np.array(CROSS_PLCC)}\n')
                writer.write(f'PLCCs average = {np.mean(np.abs(CROSS_PLCC)): .4f}\n')
                writer.write(f'PLCCs median = {np.median(np.abs(CROSS_PLCC)): .4f}\n')
            
    print('Done!')        
        
                
            
            
            
            
            
            
            
    