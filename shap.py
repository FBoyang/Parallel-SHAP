

import numpy as np
import pandas as pd
from itertools import combinations 
from sklearn.model_selection import cross_validate
from numpy.random import binomial
from itertools import permutations 
from sklearn import svm
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score as mis
from sklearn.metrics import adjusted_mutual_info_score as amis
import warnings
from sklearn import datasets
from numpy.random import choice
warnings.filterwarnings('ignore')
import multiprocessing
from sklearn import tree
import time
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor 
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder



class DistSHAP:
    def __init__(self, X, y, ytype='str',method='shap'):
        self.X = X
        self.y = y
        self.ytype = ytype
        self.method = method

    def featcompute(self, params):
        order, ID, index = params
        # each permutation has a unique ID
        X = self.X
        y = self.y
        ytype = self.ytype
        m = len(order)
        x_vars = np.zeros(m)
        if ytype == 'str':
            if index == 0:
                y_pred_proba = pd.Series(y).value_counts().sort_index().values
                y_pred_proba = y_pred_proba/np.sum(y_pred_proba)
                v_i = -entropy(y_pred_proba)
            else:
                v_i = self.cateVal(X[:,order[:index]], y)
                
        else:
            if index == 0:
                v_i = 0
            else:
                v_i = self.varVal(X[:,order[:index]], y) 
        if index == 0:
            return ID, -1, v_i
        else:
            return ID,order[index-1],v_i


    def cateVal(self, X, y):
        clf = GradientBoostingClassifier(n_estimators=20)
        return cross_validate(clf, X, y, cv=3,n_jobs=-1, return_train_score=False, scoring='neg_log_loss')['test_score'].mean()

    def varVal(self, X, y):
        regr = GradientBoostingRegressor(n_estimators=20)
        return cross_validate(regr, X, y, cv=3,n_jobs=-1, return_train_score=False, scoring='explained_variance')['test_score'].mean()


    def ParallSampler(self, sample_size=100, processor=1,debug=False):
        method = self.method
        X = self.X
        y = self.y
        M = X.shape[1]
        ytype = self.ytype
        if ytype == 'str':
            y = y.astype('str')
        else:
            y = preprocessing.scale(y)
        M = X.shape[1]
        val_list = []
        arr = np.empty((0,M))

        for i in range(sample_size):
            arr = np.vstack([arr, np.random.permutation(np.arange(M))])

        arr = np.unique(arr,axis=0)
        arr = arr.astype(int)
        
        params = []
        tsample_size = arr.shape[0]
        for i in range(tsample_size):
            order = arr[i]
            # index = M+1 because take into account the empty set
            for index in range(M+1):
                params.extend([(order,i,index)])
    
        if debug:
            for param in params:
                print(self.featcompute(param))
        else:
            pool = multiprocessing.Pool(processes=processor)
            all_vals = pool.map(self.featcompute, params)
            pool.close()
            pool.join()

        all_vals = pd.DataFrame(data=all_vals, columns=['ID', 'position', 'score'])
        featImps = np.empty((0,M))
        for i in range(tsample_size):
            group = all_vals[all_vals.ID == i].score.values
            featImp=group[1:]-group[:-1]
            order_index = np.argsort(all_vals[all_vals.ID == i].position.values[1:])
            featImp = featImp[order_index]
            featImps = np.vstack([featImps,featImp])

        featImp = np.mean(featImps, axis=0)
        return featImp
    
    def SHAPplot(self, feat_imp, featname, dataname = 'data', target_name=None, multiclass=False, save=False, path=''):
        if multiclass:
            if target_name == None:
                print('please provide target name when ploting multiclass feature importance')
                return

        else:
            plt.rcdefaults()
            fig, ax = plt.subplots(1)
            # Example data
            ylab = featname
            y_pos = np.arange(len(ylab))
            sort_index = np.argsort(feat_imp)[::-1]
            print(sort_index)
            shift=np.zeros(len(ylab))
            ax.barh(y_pos, feat_imp[sort_index], align='center', left=shift)
            shift += feat_imp[sort_index]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(np.array(ylab)[sort_index])
            ax.invert_yaxis()  # labels read top-to-bottom
            # ax.legend(feat_names)    
            ax.set_xlabel('Feature Importance')
            ax.set_title('SHAP')
            if save:
                fig.savefig(path+f'/SHAP_{dataname}.pdf',bbox_inches='tight')
