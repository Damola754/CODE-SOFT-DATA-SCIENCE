#!/usr/bin/env python
# coding: utf-8

# TASK 3: IRIS FLOWER CLASSIFICATION
# 
# Author: Yogesh Baghel
# 
# Domain: Data Science
# 
# Name: Bobade Adedamola Timilehin
# 
# Batch: June batch A56
# 
# Aim: Use the iris dataset to develop a model that can classify iris flowers into different species based on thier sepal and petal measurements.
# 
# Objective: To train a machine learning model that can learn from these measurments and accurately classify the iris flowers into their respective species    
# 
# Task: IRIS CLASSIFICATION PREDICTION
# 
# The iris flower dataset  consist of three species: setosa, versicolor, and virginica. These species can be distingushed based on their measurements.

# Importing libraries

# In[1]:


from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


# Importing datset

# In[2]:


df=pd.read_csv('IRIS.csv')


# In[3]:


df.head(5)


# In[4]:


df.tail()


# In[5]:


df.describe()


# In[6]:


df.iloc[50:80]


# In[7]:


#checking for missing value
df.isnull().sum()


# In[8]:


# Count the occurrences of each species
species_counts = df['species'].value_counts()

species_counts


# In[9]:



# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Distribution of Iris Species')
plt.show()


# In[10]:


#Using scatterplot to visualize species wrt sepal(length, width) and petal(length, width)
sns.pairplot(df, hue="species")
plt.show()


# In[11]:


# dropping "species" column and assigning other columns as feature variables 
x = df.drop("species", axis = 1)
print(x)


# In[12]:


# assigning species as target variable
y = df["species"]


# In[13]:


y


# In[14]:


#shape of the data (row, column)
df.shape


# In[15]:


#Checking for null values in the dataset
df.isnull().values.any()


# In[16]:


#spliting data into test and train by using sklearn library
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[17]:


#The prediction(KNeighborsClassifier) model
Knn = KNeighborsClassifier(n_neighbors = 3)
Knn.fit(x_train, y_train)


# In[18]:


y_pred = Knn.predict(x_test)


# In[19]:


print("Accuracy on the test data:", accuracy_score(y_test, y_pred).round(2))


# In[20]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[21]:


new_df= pd.DataFrame({"sepal_length" :[7.0], "sepal_width":3.5, "petal_length":4.7, "petal_width": 1.5})


# In[22]:


prediction = Knn.predict(new_df)


# In[23]:


prediction[0]


# In[24]:


df.iloc[50:]


# logistic Regression Model

# In[25]:


# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# In[26]:


df=pd.read_csv('IRIS.csv')


# In[44]:


#Separating features (X) and target labels (y)
X= df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y=df [['species']].values


# In[46]:


y


# In[47]:


# Reshape y to be a 1D array
y = y.ravel()


# In[48]:


y


# In[49]:



#Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[50]:


#Creating the Logistic Regression model
logistic_model = LogisticRegression(multi_class='ovr', solver='liblinear') 


#Train the model on the training data
logistic_model.fit(X_train, y_train)

#predictions on the testing data
predictions = logistic_model.predict(X_test)


# In[51]:


#Evaluating model performance
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))


# In[ ]:





# In[52]:


#compute confusion matrix for test data
from sklearn.metrics import confusion_matrix
print("Matrix:", confusion_matrix(predictions, y_test))


# Interpretation
# 
# Precision, Recall, and F1-Score:
# 
# Iris-setosa: Achieves perfect scores across precision, recall, and F1-score, indicating the model classifies this class with 100% accuracy.
# Iris-versicolor: Has perfect precision (1.00) and a recall of 0.92, meaning it correctly identifies all Iris-versicolor samples but misses one instance (hence the recall of 92%).
# Iris-virginica: Shows strong performance with precision of 0.93 and perfect recall of 1.00, indicating that while it misses a few samples when predicting this class, it identifies all true instances.
#     
#     
# Accuracy:
# 
# The overall accuracy is 97.78%, indicating that the model correctly predicts the class of 44 out of 45 samples.
# 
# Confusion Matrix:
# 
# [[19 0 0], [0 12 0], [0 1 13]]:
# Iris-setosa: All 19 instances correctly classified.
# Iris-versicolor: 12 out of 13 instances correctly classified, with one misclassified as Iris-virginica.
# Iris-virginica: 13 out of 13 instances correctly classified, with no misclassifications.
#     
# Summary
# High Performance: The model demonstrates high performance across all metrics, with particularly outstanding results for Iris-setosa (perfect scores) and strong results for Iris-virginica.
# Slight Misclassification: There is a minor issue with Iris-versicolor, where one instance is misclassified as Iris-virginica. However, this does not significantly affect the overall high accuracy and performance metrics.
#     
# Conclusion
# Model Efficacy: The logistic regression model used is highly effective in classifying the three iris species, achieving nearly perfect accuracy.
# Robust Classification: The classification report and confusion matrix suggest that the model generalizes well and performs robustly on the test data.
# 
#     
# Overall, the model is highly reliable for classifying the iris species and can be confidently used for this task.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:




