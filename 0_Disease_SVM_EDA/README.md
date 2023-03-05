# Data Science - Support Vector Machine (SVM) Multiclass Classification

In this project, I will be talking about a Machine Learning Model called Support Vector Machine (SVM). It is a very powerful and versatile supervised Machine Learning model, superivsed meaning sample data should be labeled. Algorithm works perfectly for almost any supervised problem and it can be used for **classification** or **regression problems**. There is a slight difference in the implementation of these algorithms, *Support Vector Classifier* and *Support Vector Regressor*.

# Support Vector Machine (SVM)
SVM is well suited for classification of complex but small or medium sized datasets. To generalize, the objective is to find a hyperplane that maximizes the separation of the data points to their potential classes in an n-dimensional space. The data points with the minimum distance to the hyperplane  are called Support Vectors.

- **The One-to-Rest approach** - separate between every two classes. Each SVM would predict membership in one of the **m** classes. This means the separation takes into account only the points of the two classes in the current split. Thus, the red-blue line tries to maximize the separation only between blue and red points and It has nothing to do with green points.

- **The One-to-One approach** - separate between a class and all others at once, meaning the separation takes all points into account, dividing them into two groups; a group for the class points and a group for all other points. Thus, the green line tries to maximize the separation between green points and all other points at once.

<div align="center">
  Example of 3 classes classification
</div>

<p align="center">
  <img width="600" height="300" src="https://github.com/sulova/Data_Science_Disease_SVM/blob/main/SVM.PNG ">
</p>

**The advantages of SVM**
 - Effective in high dimensional spaces
 - Effective in cases where number of dimensions is greater than the number of samples
 - Memory efficient as it uses a subset of training points in the decision function (called support vectors)
 
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

**The disadvantages of SVM include**
 - If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial
 - Do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation
 - Do not give the best performance for handling text structures as compared to other algorithms that are used in handling text data. 

Training Support Vector Machines for Multiclass Classification

```python
# import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
```

Test & Train split 20:80
```python
# Test & Train split 
filename = 'Data/disease.tsv'
df = pd.read_table(filename, sep='\t') 

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
df 

```

Frequency distribution of classes & Visualizing Outcome Train Distribution 
```python
train_outcome = pd.crosstab(index=train["Class"],  # Make a crosstab
                              columns="count")      # Name the count column
# Visualizing Outcome Distribution 
temp = train["Class"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })

#df.plot(kind='pie',labels='labels',values='values', title='Activity Ditribution',subplots= "True")

labels = df['labels']
sizes = df['values']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','cyan','lightpink']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()                              

```
Check for missing values in the dataset
```python
print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")
```

Visualizing Outcome Distribution
```python
temp = train["Class_column"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
labels = df['labels']
sizes = df['values']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','cyan','lightpink']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
```
Seperating Predictors and Outcome values from train and test sets
```python
X_train = pd.DataFrame(train.drop(['Class_column','subject'],axis=1))
Y_train_label = train.Class_column.values.astype(object)
X_test = pd.DataFrame(test.drop(['Class_column','subject'],axis=1))
Y_test_label = test.Class_column.values.astype(object)
```
Dimension of Train and Test set 
```python
print("Dimension of Train dataset",X_train.shape)
print("Dimension of Test dataset",X_test.shape,"\n")
```
Transforming non numerical labels into numerical labels
```python
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
```
Encoding train labels 
```python
encoder.fit(Y_train_label)
Y_train = encoder.transform(Y_train_label)
# encoding test labels 
encoder.fit(Y_test_label)
Y_test = encoder.transform(Y_test_label)
```

Total Number of Continous and Categorical features in the training set
```python
num_cols = X_train._get_numeric_data().columns
print("Number of numeric features:",num_cols.size)
```
Libraries to Build Ensemble Model : Random Forest Classifier.  Create the parameter grid based on the results of random search 
```python
params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
                    
names_of_predictors = list(X_train.columns.values)

# Scaling the Train and Test feature set 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Performing CV to tune parameters for best SVM fit 
```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
svm_model = GridSearchCV(SVC(), params_grid, cv=5)
svm_model.fit(X_train_scaled, Y_train)
```
 View the accuracy score
```python
print('Best score for training data:', svm_model.best_score_,"\n") 

# View the best parameters for the model found using grid search
print('Best C:',svm_model.best_estimator_.C,"\n") 
print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

final_model = svm_model.best_estimator_
Y_pred = final_model.predict(X_test_scaled)
Y_pred_label = list(encoder.inverse_transform(Y_pred))
```
Making the Confusion Matrix
```Python
#print(pd.crosstab(Y_test_label, Y_pred_label, rownames=['Actual Activity'], colnames=['Predicted Activity']))
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(Y_test_label,Y_pred_label))
print("\n")
print(classification_report(Y_test_label,Y_pred_label))

print("Training set score for SVM: %f" % final_model.score(X_train_scaled , Y_train))
print("Testing  set score for SVM: %f" % final_model.score(X_test_scaled  , Y_test ))

svm_model.score
```
