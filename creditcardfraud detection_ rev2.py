# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:43:56 2020

@author: ankit
"""

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import seaborn as sns
from sklearn.model_selection import train_test_split


data = pd.read_csv("D:\Data Science Projects\Credit Card fraud Detection\creditcardfraud\creditcard.csv")
data.head()
data.shape[0]
data.shape[1]
data1= data[data["Class"]==1]
data0= data[data["Class"]==0]
data["Time"]
#******Exploratory analysis*******

# Stats summary 
data.info()

x= data.describe()# decribe() only applicable for numerical functions

# finding missing values
Total = data.isnull().sum()
data.isnull().count()
percent = (data.isnull().sum()/data.isnull().count())*100
missvalues=pd.concat([Total,percent],axis=1,keys=['Total','Percent']).transpose()
missvalues
##### there is no missing value

## class wise histogram
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(data1["Amount"], bins = bins)
ax1.set_title('Fraud')
plt.yscale('log')
ax2.hist(data0["Amount"], bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();

data1["Amount"].describe()
data0["Amount"].describe()

### class wise boxplots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data, palette="PRGn",showfliers=False)
plt.show()

# visualising imbalamces in data
cntx={"class":["normal","fraud"],"Transactions":[data0.shape[0],data1.shape[0]]}
cnt1x= pd.DataFrame(cntx)
sns.barplot(data = cnt1x,x='class',y='Transactions')
plt.show()
## scatter plot showing both classes
plt.scatter(data1["Time"], data1["Amount"], c='red', s=200, label='1',alpha=0.8)
plt.scatter(data0["Time"], data0["Amount"], c='green', s=40, label='0',alpha =0.09)
plt.xlabel('Time')
plt.ylabel('Amount')
plt.legend()
plt.show()

plt.scatter(data1["Time"], data1["Amount"], c='red', s=40, label='1',alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Amount')
plt.legend()
plt.show()

plt.scatter(data0["Time"], data0["Amount"], c='green', s=40, label='0',alpha =0.09)
plt.xlabel('Time')
plt.ylabel('Amount')
plt.legend()
plt.show()

## Density Plot for transactions# During night time normal transactions are less....but fraud transactions are as usual

plt.figure(figsize = (14,4))
plt.title('Credit Card Transactions Time Density Plot')
sns.set_color_codes("pastel")
sns.distplot(data0["Time"],kde=True,bins=480)
sns.distplot(data1["Time"],kde=True,bins=480)
plt.show()

###Correlation matrix
plt.figure(figsize = (16,16))
plt.title('Credit Card Transactions features correlation plot')
corrl = data.corr()
sns.heatmap(corrl,xticklabels=corrl.columns,yticklabels=corrl.columns,linewidths=.1,cmap="Reds")
plt.show()

### class wise variables density plot
var = data.columns.values

i = 0
t0 = data.loc[data['Class'] == 0]
t1 = data.loc[data['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(14,24))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
    
plt.show();



## bw???

###****Model Training

### train test split
x_train,x_test,y_train,y_test=train_test_split(data.iloc[:,0:30],data.iloc[:,30],test_size=0.2,random_state=2018)
### train and validation test split
x_train1,x_val,y_train1,y_val=train_test_split(x_train,y_train,test_size=0.2,random_state=2018)

type(y_train)
x_train.shape[0]
pd.DataFrame(y_train)
counttrain1 = y_train[y_train==1]
countval1=y_val[y_val==1]
y_train.shape[0]
counttrain1.shape[0]

percenttrain1=counttrain1.shape[0]/y_train.shape[0]*100
percentval1=countval1.shape[0]/y_val.shape[0]*100
percentorgdata=data1.shape[0]/data.shape[0]*100


##Decision Tree

####
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

## cross validation for finding tuning parameter

x=[]
for i in range(10,120,10):
    dtree = DecisionTreeClassifier(random_state=0, max_leaf_nodes=i) ## you can define Class weights here to deal with imbalance data
    dtree.fit(x_train1,y_train1)
    pred = dtree.predict(x_val)
    df1 = f1_score(y_val, pred)
    print ('f1 for nodes'+ str(i)+ 'is'+ str(df1)) 
    x.append(str(df1))
    
nodes = range(10,120,10)
plt.plot(nodes,x)
plt.xlabel('# of Nodes')
plt.ylabel('f1')
plt.show()

nodes[x.index(max(x))] # no of nodes for max recall value, tuned parameter for decision tree

## implement tuned decision tree on test set
dtree = DecisionTreeClassifier(random_state=0,max_leaf_nodes=nodes[x.index(max(x))])
dtree.fit(x_train1,y_train1)
#dpred = dtree.predict(x_test) predict with 0.5 threshold
dthresh = 0.1
dprob=dtree.predict_proba(x_test)[:,1]
dpredprob = (dprob>= dthresh).astype(bool) # setting threshold for probability
d_acc = accuracy_score(y_test,dpredprob)
d_prec = precision_score(y_test, dpredprob)
d_rec= recall_score(y_test, dpredprob)
d_f1= f1_score(y_test,dpredprob) ## metrics to be used for evaluating performance of models



## Confusion Matrix  for test data
cm = sklearn.metrics.confusion_matrix(y_test, dpredprob) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.title('Decision Tree  '+ 'F1 : ' + str(format(d_f1prob, '.3f')) + ', Recall : '+ str(format(d_recprob, '.3f'))+ ' Threshold Prob : ' + str(dthresh))
plt.show()



### Random forest
from sklearn.ensemble import RandomForestClassifier

rclf= RandomForestClassifier(n_estimators=100,max_features='auto', random_state=0)
rclf.fit(x_train, y_train)
#rpred=rclf.predict(x_test)
rthresh = 0.1
rprob = rclf.predict_proba(x_test)[:,1]
rpred = (rprob >=rthresh).astype(bool)

r_acc = accuracy_score(y_test,rpred)
r_prec = precision_score(y_test, rpred)
r_rec = recall_score(y_test, rpred)
r_f1 = f1_score(y_test, rpred)

## Confusion Matrix  for test data
r_cm = sklearn.metrics.confusion_matrix(y_test, rpred) # rows = truth, cols = prediction
r_df_cm = pd.DataFrame(r_cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(r_df_cm, annot=True, fmt='g')
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.title('Random forest  '+ 'F1 : ' + str(format(r_f1, '.3f')) + ', Recall : '+ str(format(r_rec, '.3f')) + ' Threshold Prob : ' + str(rthresh) )
plt.show()


## feature Importance
predictors = list(x_train.columns.values)
#rclf.feature_importances_
rtmp=pd.DataFrame({'Feature':predictors,'Importance':rclf.feature_importances_})
rtmp=rtmp.sort_values(by='Importance',ascending=False)
plt.figure(figsize=(18,10))
plt.title('Features importance',fontsize=14)
s=sns.barplot(x='Feature',y='Importance',data=rtmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()



        
### New XGBoost Training
import xgboost as xgb

import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics 


# min_child_weight is the minimum weight (or number of samples if all samples have a weight of 1) required in order to create a new node in the tree. A smaller min_child_weight allows the algorithm to create children that correspond to fewer samples, thus allowing for more complex trees, but again, more likely to overfit.

## tuning max_depth and Child weights
cv_params = {'max_depth': [1,3,5], 'min_child_weight': [1,3]}    # parameters to be tries in the grid search
fix_params = {'learning_rate': 0.2, 'n_estimators': 100,'early_stopping_rounds': 10, 'objective': 'binary:logistic'}   #other parameters, fixed for the moment 
csv = GridSearchCV(xgb.XGBClassifier(**fix_params), cv_params, scoring = 'f1', cv = 5)
csv.fit(x_train, y_train)
csv.best_params_
##fixing {'max_depth': 5, 'min_child_weight': 3}

## tuning tree sample size and factions of columns
cv_params = {'subsample': [0.8,1],'colsample_bytree': [0.8,1]}
fix_params = {'learning_rate': 0.2, 'n_estimators': 100,'early_stopping_rounds': 10, 'objective': 'binary:logistic', 'max_depth': 5, 'min_child_weight':3}
csv = GridSearchCV(xgb.XGBClassifier(**fix_params), cv_params, scoring = 'f1', cv = 5) 
csv.fit(x_train, y_train)
csv.grid_scores_
csv.best_params_
##fixing {'colsample_bytree': 1, 'subsample': 0.8}

## tuning Learning rate
cv_params = {'learning_rate': [0.05, 0.1, 0.2, 0.3]}
fix_params['subsample'] = 0.8
fix_params['colsample_bytree'] = 1
csv = GridSearchCV(xgb.XGBClassifier(**fix_params), cv_params, scoring = 'f1', cv = 5) 
csv.fit(x_train, y_train)
csv.grid_scores_
csv.best_params_
#{'learning_rate': 0.2}

fix_params['learning_rate'] = 0.2
params_final =  fix_params
#params_final : {'colsample_bytree': 1,'learning_rate': 0.2,'max_depth': 5,'min_child_weight': 3, 'n_estimators': 100,'objective': 'binary:logistic','subsample': 0.8,'validate_parameters': 1} 
print(params_final)

dtrain = xgb.DMatrix(x_train, label= y_train)
dtest = xgb.DMatrix(x_test, label= y_test)

xgb_final = xgb.train(params_final, dtrain, num_boost_round = 100)
xgbprob = xgb_final.predict(dtest)
xgbthresh = 0.1
xy_pred = (xgbprob>=xgbthresh).astype(bool)

## metrics
x_acc = accuracy_score(y_test,xy_pred)
x_prec = precision_score(y_test, xy_pred)
x_rec = recall_score(y_test, xy_pred)
x_f1 = f1_score(y_test, xy_pred)

## Confusion Matrix  for test data
x_cm = sklearn.metrics.confusion_matrix(y_test, pd.DataFrame(xy_pred)) # rows = truth, cols = prediction
x_df_cm = pd.DataFrame(x_cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(x_df_cm, annot=True, fmt='g')
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.title('XGBoost Prediction  '+ 'F1 : ' + str(format(x_f1, '.3f')) + ', Recall : '+ str(format(x_rec, '.3f')) + ' Threshold Prob : ' + str(xgbthresh))
plt.show()

#### ROC and recall precision curve
from sklearn.metrics import precision_recall_curve
lr_precision, lr_recall, thresholdPR = precision_recall_curve(y_test,xgbprob )
#lr_auc = roc_auc_score(y_test,xy_predprob)
lr_fpr, lr_tpr, thresholdroc= sklearn.metrics.roc_curve(y_test,xgbprob)
figure2, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(lr_recall, lr_precision, marker='.', label='Xgboost')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.legend()
ax2.plot(lr_fpr, lr_tpr, marker='.', label='XGboost')
ax2.set_xlabel('FPR')
ax2.set_ylabel('TPR')
ax2.legend()
plt.show()


#Voting for Class
modelcmp =pd.DataFrame({'DT_prob':dprob,'RF_prob':rprob,'xgb_prob':xgbprob})
dthresold= 0.1
rfthresold =0.1
xgbthresold=0.1
modelcmp["DT_ Class"]=(modelcmp["DT_prob"]>dthresold)*1
modelcmp["RF_ Class"]=(modelcmp["RF_prob"]>rfthresold)*1
modelcmp["XGB_ Class"]=(modelcmp["xgb_prob"]>xgbthresold)*1
modelcmp["Max_Voteclass"]= ((modelcmp["DT_ Class"] +modelcmp["RF_ Class"]+modelcmp["XGB_ Class"])>1.5)*1

## evaluation matrix
v_acc = accuracy_score(y_test,modelcmp[["Max_Voteclass"]])
v_prec = precision_score(y_test,modelcmp[["Max_Voteclass"]])
v_rec = recall_score(y_test, modelcmp[["Max_Voteclass"]])
v_f1 = f1_score(y_test, modelcmp[["Max_Voteclass"]])


## Confusion Matrix  for test data
v_cm = sklearn.metrics.confusion_matrix(y_test,modelcmp[["Max_Voteclass"]] ) # rows = truth, cols = prediction
v_df_cm = pd.DataFrame(v_cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(v_df_cm, annot=True, fmt='g')
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.title('Max Vote Class Prediction  '+ 'F1 : ' + str(format(v_f1, '.3f')) + ', Recall : '+ str(format(v_rec, '.3f')) + ' Threshold Prob(DT,RF,XGB) : ' + str(dthresold)+ ' , ' +str(rfthresold)+ ' , ' +str(xgbthresold))
plt.show()

## comparing Models Evaluation metrics

Metrics = [ (d_rec, d_prec, d_f1 , d_acc) ,
             (r_rec, r_prec, r_f1 , r_acc) ,
             (x_rec, x_prec, x_f1 , x_acc ) ,
             (v_rec, v_prec, v_f1 , v_acc) ]

Models_comparison= pd.DataFrame(Metrics, columns = ['Recall' , 'Precision', 'F1' , 'Accuracy'], index=['Decision Tree', 'Random Forest', 'XGBoost' , 'Max Vote Model'])
Models_comparison = Models_comparison.sort_values(by='Recall',ascending=False)

