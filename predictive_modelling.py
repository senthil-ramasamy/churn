from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the csv file

df=pd.read_csv('data.csv')

#checking input data

print df.head()

print np.unique(df['TotalCharges'])

print df.dtypes

#isolate data

churn_result=df['Churn']

y=np.where(churn_result=='Yes',1,0);

#convert string to number

churnspace=df

churnspace=churnspace.replace(['Yes','No','No internet service','No phone service'],[1,0,0,0])
    
churnspace=churnspace.replace(['Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check'],[1,2,3,4])

churnspace=churnspace.replace(['DSL','Fiber optic','Male','Female'],[1,2,1,0])

churnspace=churnspace.replace(['Month-to-month','One year','Two year'],[1,2,3])

#checking converted data

print churnspace.head()

#convert object to numeric in dataframe

churnspace = churnspace.apply(pd.to_numeric, errors='coerce')

print churnspace.dtypes

#checking nan values

print np.any(np.isnan(churnspace))

X=churnspace.as_matrix().astype(np.float)

# Normalisation

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print "Feature space holds %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(y)


#doing kfold and cross validation

from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=3,shuffle=True)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


#applying different classifier algorithm

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)

print "Support vector machines:"
print "%.3f" % accuracy(y, run_cv(X,y,SVC))
print "Random forest:"
print "%.3f" % accuracy(y, run_cv(X,y,RF))
print "K-nearest-neighbors:"
print "%.3f" % accuracy(y, run_cv(X,y,KNN))



#drawing confusion matrix

from sklearn.metrics import confusion_matrix

y = np.array(y)
class_names = np.unique(y)

confusion_matrices = [
    ( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC)) ),
    ( "Random Forest", confusion_matrix(y,run_cv(X,y,RF)) ),
    ( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN)) ),
]


conMatrix = confusion_matrix(y,run_cv(X,y,SVC))

# Pyplot code not included to reduce clutter
import matplotlib.pyplot
import pylab as pl
#%matplotlib inline

def draw_confusion_matrices(confusion_matrices, class_names):
    labels = list(class_names)

    for cm in confusion_matrices:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.matshow(cm[1])
        pl.title('Confusion Matrix\n(%s)\n' % cm[0])
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        pl.xlabel('Predicted Class')
        pl.ylabel('True Class')

        for i,j in ((x,y) for x in xrange(len(cm[1])) for y in xrange(len(cm[1][0]))):
            ax.annotate(str(cm[1][i][j]), xy=(i,j), color='white')

        pl.show()
        

draw_confusion_matrices(confusion_matrices,class_names)
print confusion_matrices
print conMatrix






#moving to probability

def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob



import warnings
warnings.filterwarnings('ignore')

# Use 10 estimators so predictions are all multiples of 0.1
pred_prob = run_prob_cv(X, y, RF, n_estimators=10)
pred_churn = pred_prob[:,1]
is_churn = y == 1

# Number of times a predicted probability is assigned to an observation
counts = pd.value_counts(pred_churn)

# calculate true probabilities
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)

# pandas-fu
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
print counts

from ggplot import *
#%matplotlib inline

baseline = np.mean(is_churn)
g=ggplot(counts,aes(x='pred_prob',y='true_prob',size='count')) + \
    geom_point(color='blue') + \
    stat_function(fun = lambda x: x, color='red') + \
    stat_function(fun = lambda x: baseline, color='green') + \
    xlim(-0.05,  1.05) + ylim(-0.05,1.05) + \
    ggtitle("Random Forest") + \
    xlab("Predicted probability") + ylab("Relative frequency of outcome")

print g
