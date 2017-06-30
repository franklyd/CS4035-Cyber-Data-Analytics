# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:58:31 2017

@author: xps
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

file_name = 'capture20110817.binetflow'

df = pd.read_csv(file_name,sep=',')

# Assign simple labels
def assgin_label(label):
    if label.find('From-Normal')!=-1:
        return 0
    elif label.find('From-Botnet')!=-1:
        return 1 
    else:
        return -1
df['label'] = df['Label'].apply(lambda x: assgin_label(x))
sns.countplot(x="label", data=df, palette="Greens_d")
# imbalanced: 184987 botnet, 29893 normal
new_df = df[df['label']!=-1]

data = new_df[['Dur','Proto','TotPkts','TotBytes','SrcBytes']]
# One-Hot Encoding for Proto
data = pd.get_dummies(data)
X = data
y = new_df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
#clf = LogisticRegression()#class_weight='balanced')
clf = RandomForestClassifier(class_weight='balanced')
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_pred=y_pred,y_true=y_test).ravel()
print 'precision:', float(tp)/(tp+fp)
print 'recall',float(tp)/(tp+fn)

# Host level 
clf = RandomForestClassifier(n_estimators = 1000,class_weight='balanced')
clf.fit(X,y)
new_df['y_pred'] = clf.predict(X)
wrong_df = new_df[new_df['y_pred']!=new_df['label']]
