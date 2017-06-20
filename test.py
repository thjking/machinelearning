import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from  sklearn.linear_model import SGDClassifier

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli'
                , 'Mitoses', 'Class']

data = pd.read_csv('E:\Python machine learning code\Datasets\Breast-Cancer\\breast-cancer-wisconsin.csv', names=column_names)

data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')

X_train,X_test,Y_train,Y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

#print Y_train.value_counts()
#print X_test.value_counts()

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression()
sgdc = SGDClassifier()

lr.fit(X_train,Y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train,Y_train)
sgdc_y_predict = sgdc.predict(X_test)