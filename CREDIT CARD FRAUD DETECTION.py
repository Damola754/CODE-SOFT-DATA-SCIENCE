#!/usr/bin/env python
# coding: utf-8

# TASK 5: CREDIT CARD FRAUD DETECTION
# 
# Author: Yogesh Baghel
# 
# Domain: Data Science
# 
# Name: Bobade Adedamola Timilehin
# 
# Batch: June batch A56
# 
# Task: CREDIT CARD FRAUD DETECTION 
#     Build a machine learning model to identify fraudulent credit card transactions.
#     preprocess and normalize the transaction data, handle class imbalance issues, and split the data into training and testing sets.
#     Train a classification algorithm, such as logisitc regression or random forest, to classify transactions as fraudulent or genuine.
#     Evaluate the model's performance using metrics like precision, recall,and f1-score, and consider techniques like oversampling or undersampling for improving results.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy
from sklearn. metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED =42
LABELS = ["Normal", "Fraud"]


# In[3]:


#importing dataset
df = pd.read_csv('creditcard.csv')


# In[4]:


df.head()


# In[5]:


#data infromation
df.info()


# In[6]:


#checking for null values


# In[7]:


df.isnull().values.any()


# In[8]:


count_classes = pd.value_counts(df['Class'], sort= True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[9]:


#Assigning fraud as 1 to the class cloumn and normal to 0


# In[10]:


fraud = df[df['Class']==1]
normal= df[df['Class']==0]


# In[11]:


print(fraud.shape,normal.shape)


# In[13]:


#Checking the unique data 
df['Class'].unique()


# In[9]:


#how different are the amount of money used in different transaction classes?
fraud.Amount.describe()


# In[10]:


normal.Amount.describe()


# In[11]:


f, (ax1, ax2)=plt.subplots(2, 1, sharex=True)
f. suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[ ]:


#Checking how fraudulent transactions occur more often during certain time frame?


# In[12]:


f, (ax1, ax2)=plt.subplots(2, 1, sharex=True)
f. suptitle('Time of transaction vs Amount by class')
bins = 50
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show();


# In[ ]:


#taking some sample of the data


# In[13]:


df1=df.sample(frac = 0.1, random_state=1)
df1.shape


# In[14]:


df.shape


# In[ ]:


# to determine the number of fraud and valid transactions in the dataset


# In[15]:


fraud = df1[df1['Class']==1]
Valid = df1[df1['Class']==0]
outlier_fraction = len(fraud)/float(len(Valid))


# In[16]:


print(outlier_fraction)
print("fraud cases : {}" .format(len(fraud)))
print("Valid cases : {}" .format(len(Valid)))


# In[17]:


#create independent and dependent features
columns = df1.columns.tolist()
#filter the column to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]
#store the variable we are predicting
target = "Class"
#define a random state
state = np.random.RandomState(42)
X = df1[columns]
Y = df1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
#print the shapes of X & Y
print(X.shape)
print(Y.shape)


# In[18]:


#correlation
import seaborn as sns
#get correlation of each features in dataset
corrmat =df1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(30,30))
#plot heat map
g = sns.heatmap(df[top_corr_features].corr(),annot=True)


# In[ ]:


Model Prediction

Isolation Forest Algorithm


# In[19]:


#define the outlier detection methods

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X),
                                        contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                               leaf_size=30, metric='minkowski',
                                               p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05,
                                                           max_iter=-1)
}


# In[20]:


type(classifiers)


# In[ ]:


n_outliers = len(fraud)
for i, (clf_name, clf) in enumerate(classifiers.items()): 
    # Fitting the data and tagging classifier
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":  
        clf.fit(X)
        y_pred = clf.predict(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
        # Prediction value to 0 for valid transactions, 1 for fraud transactions
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        n_errors = (y_pred != Y).sum() 
        print("{}: {}".format(clf_name, n_errors))
        print("Accuracy Score:")
        print(accuracy_score(Y, y_pred))
        print("Classification Report:")
        print(classification_report(Y, y_pred))


# Using Logistics Regression model

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('creditcard.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[ ]:





# In[5]:


df.info()


# In[ ]:





# In[6]:


df.isnull().sum()


# In[ ]:





# In[7]:


df['Class'].value_counts()


# In[ ]:





# In[ ]:





# In[8]:


normal=df[df.Class == 0]
fraud= df[df.Class == 1]


# In[9]:


print(normal.shape)
print(fraud.shape)


# In[ ]:





# In[10]:


fraud.Amount.describe()


# In[11]:


normal.Amount.describe()


# In[ ]:





# In[12]:


df.groupby('Class').mean()


# In[ ]:





# In[13]:


normal_sample=normal.sample(n=492)


# In[ ]:





# In[14]:


new_df =pd.concat([normal_sample, fraud], axis=0)


# In[15]:


new_df.head()


# In[16]:


new_df.tail()


# In[17]:


new_df['Class'].value_counts()


# In[18]:


new_df.groupby('Class').mean()


# In[ ]:





# In[19]:


X=new_df.drop(columns='Class', axis=1)
Y=new_df['Class']


# In[20]:


print(X)


# In[21]:


print(Y)


# Logistics Regression Model

# In[50]:


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[51]:


print(X.shape, X_train.shape, X_test.shape)


# In[ ]:





# In[52]:


model = LogisticRegression()


# In[ ]:





# In[53]:


model.fit(X_train, Y_train)


# In[ ]:





# In[54]:


X_train_prediction = model.predict(X_train)
training_data_accuracy =accuracy_score(X_train_prediction, Y_train)


# In[55]:


print('Accuracy on the training data:', training_data_accuracy)


# In[56]:


# Compute the confusion matrix for training data
conf_matrix = confusion_matrix(Y_train, X_train_prediction)

print(conf_matrix)


# In[57]:


X_test_prediction = model.predict(X_test)
test_data_accuracy =accuracy_score(X_test_prediction, Y_test)


# In[58]:


print('Accuracy on the test data:', test_data_accuracy)


# In[59]:


#compute confusion matrix for test data
print("Matrix:", confusion_matrix(Y_test, X_test_prediction))


# In[ ]:





# In[60]:


from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score, roc_curve


# In[61]:


print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))


# In[ ]:





# In[62]:


#Evaluation metrics
print("\nClassification Report:")
print(classification_report(Y_train, X_train_prediction))


# comparison result
# 
# Accuracy
# 
# Training Data Accuracy: 0.9504 (95.04%)
# Test Data Accuracy: 0.9340 (93.40%)
# 
# 
# Interpretation and Summary
# 
# Accuracy:
# 
# The accuracy on the training data (95.04%) is slightly higher than the accuracy on the test data (93.40%). This is generally expected as models usually perform better on the data they were trained on.
# 
# Confusion Matrix:
# 
# For the training data, the model made 10 false positives and 29 false negatives.
# For the test data, the model made 5 false positives and 8 false negatives.
# 
# 
# Precision, Recall, and F1-Score:
# 
# Precision is high for both classes in training (0.93 for class 0 and 0.97 for class 1) and test data (0.92 for class 0 and 0.95 for class 1). This indicates that the model is good at not labeling a negative sample as positive.
# Recall is also high for both classes in training (0.97 for class 0 and 0.93 for class 1) and test data (0.95 for class 0 and 0.92 for class 1). This indicates that the model is good at finding all the positive samples.
# The F1-Score combines precision and recall, and it is consistently high for both classes in both datasets, indicating a balanced performance.
# 
# 
# Conclusion
# 
# The model demonstrates strong performance on both the training and test datasets. The metrics suggest that the model generalizes well, as there is only a slight drop in performance from the training set to the test set.
# 
# 
# Balanced Performance: The precision, recall, and F1-scores are balanced across both classes, indicating the model handles both classes well without significant bias.
# 
# 
# Overall, the model is reliable and performs well, maintaining high accuracy, precision, recall, and F1-scores on unseen data, which is crucial for practical applications.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




