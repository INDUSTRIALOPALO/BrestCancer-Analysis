
# coding: utf-8

# # Clasificador de tumores

# Cargamos las diferentes librerías

# In[4]:

import numpy as np # linear algebra
import scipy as sp # 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # subplots
import seaborn as sns
import os
get_ipython().magic(u'matplotlib inline')
import csv



# In[5]:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from subprocess import check_output
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier


# In[6]:

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model


# Cargamos los datos
# 

# In[7]:

data = pd.read_csv('C:/Users/LEONARDO/Google Drive/MAESTRIA/SEGUNDO SEMESTRE/MACHINE LEARNING/Investigacion/data.csv')


# Información básica de los datos

# In[8]:

data.info()


# Los datos vienen agrupados en tres categorías: 1) Media (mean), 2) Desviación o variabilidad (se) y 3) valores inferiores (worst). Eliminamos la columna sin nombre ni dimensión y la columna de ID ya que no aportan al estudio

# In[9]:

data.drop("Unnamed: 32",axis=1,inplace=True) #Eliminación de la variable desconocida


# In[10]:

data.drop("id",axis=1,inplace=True) #Eliminación de la variable de código de identificación


# In[11]:

data.columns #Se listan las variables


# In[12]:

data.info() #se lista cada columna (nombre de la variable) con el tipo de variable


# Teniendo en cuenta que son 3 categorías de variables, se divide el dataframe en las categorías Mean, Se, y Worst (media, desviación y valor inferior)

# In[13]:

features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)


# A partir de la descripción de las variables es posible hacer una aproximación a su dispersión y distribución (teniendo en cuenta los quartiles)

# In[14]:

data.describe()


# In[15]:

# lets get the frequency of cancer stages
sns.countplot(data['diagnosis'],label="Count")


# In[16]:

corr = data[features_mean].corr() # .corr is used for find corelation
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'PuBuGn') 


# In[17]:

corr = data[features_se].corr() # .corr is used for find corelation
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_se, yticklabels= features_se,
           cmap= 'PuBuGn') 


# In[18]:

corr = data[features_worst].corr() # .corr is used for find corelation
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_worst, yticklabels= features_worst,
           cmap= 'PuBuGn') 


# In[ ]:




# In[19]:

data.drop("perimeter_mean",axis=1,inplace=True)


# In[20]:

data.drop("area_mean",axis=1,inplace=True)


# In[21]:

data.drop("compactness_mean",axis=1,inplace=True)
data.drop("concavity_mean",axis=1,inplace=True)


# In[22]:

type(data)


# Existe una fuerte correlaciónentre el radio promedio (radius_mean), el perímetro promedio (perimeter_mean) y el área promedio (area_mean). Por tanto se simplificará a una sola variable.
# Por otra parte, la compactación media (compactness_mean), la concavidad media (concavity_mean  y el punto de concavidad medio(concavepoint_mean) también presentan un alto grado de correlación.

# In[23]:

df = pd.read_csv('C:/Users/LEONARDO/Google Drive/MAESTRIA/SEGUNDO SEMESTRE/MACHINE LEARNING/Investigacion/data.csv')
type(df)


# In[24]:

data.describe()
data


# In[25]:

data1 = data.loc[:,['diagnosis','radius_mean','texture_mean', 'smoothness_mean', 'concave points_mean', 'symmetry_mean','fractal_dimension_mean']]


# In[26]:

data1.describe()


# In[27]:

data1


# In[28]:

df=data1
df.describe()


# data1.loc[:,['radius_mean','texture_mean', 'smoothness_mean', 'concave points_mean', 'symmetry_mean','fractal_dimension_mean']]

# In[29]:

g = sns.PairGrid(data1,hue='diagnosis')
g = g.map_diag(plt.hist, alpha =0.75)
g = g.map_offdiag(plt.scatter, s = 4)


# In[30]:

#prediction_var = [data1.columns[1:7]]


# Con el fin de analizar los datos de manera más sintetizada, se realizará un análisis de componentes principales en dos dimensiones. Teniendo en cuenta que si existen  varaibles altamente correlacionadas estas se unirán, se trabajará con todo el dataset
# 

# In[186]:

df2  = pd.read_csv('C:/Users/LEONARDO/Google Drive/MAESTRIA/SEGUNDO SEMESTRE/MACHINE LEARNING/Investigacion/data.csv')


# In[187]:

df_std = StandardScaler().fit_transform(df2.drop(['id','diagnosis','Unnamed: 32'], axis = 1))
pca = PCA(n_components=2)
pca.fit(df_std)
TwoD_Data = pca.transform(df_std)
PCA_df = pd.DataFrame()
PCA_df['PCA_1'] = TwoD_Data[:,0]
PCA_df['PCA_2'] = TwoD_Data[:,1]


plt.plot(PCA_df['PCA_1'][df.diagnosis == 'M'],PCA_df['PCA_2'][df.diagnosis == 'M'],'o', alpha = 0.7)
plt.plot(PCA_df['PCA_1'][df.diagnosis == 'B'],PCA_df['PCA_2'][df.diagnosis == 'B'],'o', alpha = 0.7)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(['Maligno','Beningno'])
#plt.autoscale(enable=False)
plt.Figure( figsize=(800, 600), dpi=80, facecolor='w', edgecolor='k')


# Regresión lineal

# In[188]:

clf = LogisticRegression(penalty='l2',C=0.5)
clf.fit(X,y)
print('Training Accuracy.....',clf.score(X,y))
prediction = clf.predict(testdf[['PCA_1','PCA_2']])
print('Validation Accuracy....',clf.score(testdf[['PCA_1','PCA_2']],testdf['target']))
loss = prediction - testdf['target']
accuracy = 1 - np.true_divide(sum(np.abs(loss)),len(loss))

radius = np.linspace(min(X.PCA_1), max(X.PCA_2), 100)
line = (-clf.coef_[0][0]/clf.coef_[0][1])*radius + np.ones(len(radius))*(-clf.intercept_/clf.coef_[0][1])
plt.plot(radius,line)
plt.plot(PCA_df['PCA_1'][df.diagnosis == 'M'],PCA_df['PCA_2'][df.diagnosis == 'M'],'o', alpha = 0.7)
plt.plot(PCA_df['PCA_1'][df.diagnosis == 'B'],PCA_df['PCA_2'][df.diagnosis == 'B'],'o', color = 'b', alpha = 0.7)
plt.legend(['Decision Line','Malignant','Benign'])
plt.title('Logistic Regression. Accuracy:' + str(accuracy)[0:4])
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')


# Máquina de vector de soporte lineal

# In[190]:

def model(x):
    return 1 / (1 + np.exp(-x))

PCA_df['target'] = 0
PCA_df['target'][df.diagnosis == 'M'] = 1

traindf, testdf = train_test_split(PCA_df, test_size = 0.3)

X = traindf[['PCA_1','PCA_2']]
y = traindf['target']
Reg = np.linspace(0.1,10,100)
accuracy = []
for C in Reg:
    clf = LogisticRegression(penalty='l2',C=C)
    clf.fit(X,y)
    prediction = clf.predict(testdf[['PCA_1','PCA_2']])
    loss = prediction - testdf['target']
    accuracy.append(1 - np.true_divide(sum(np.abs(loss)),len(loss)))
#loss = model(clf.coef_*X + clf.intercept_)


plt.plot(Reg,accuracy,'o')
plt.xlabel('Regularization')
plt.ylabel('Validation Score')


# In[227]:

C = 1
clf2 = SVC(kernel = 'linear',C =C)
clf2.fit(X, y)
print('training accuracy...',clf2.score(X, y, sample_weight=None))
print('validation accuracy...',clf2.score(testdf[['PCA_1','PCA_2']],testdf['target']))

w = clf2.coef_[0]
a = -w[0] / w[1]
xx =  np.linspace(min(X.PCA_1), max(X.PCA_2), 100)
yy = a * xx - (clf2.intercept_[0]) / w[1]
plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8)
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8)
plt.scatter(clf2.support_vectors_[:, 0], clf2.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, color = 'g')
plt.plot(xx, yy)
plt.title('SVM.' + ' Reg =' + str(C) + 'Accuracy:' + str(clf2.score(testdf[['PCA_1','PCA_2']],testdf['target']))[0:4], fontsize = 10)


mu_vec1 = np.array([0,0])
cov_mat1 = np.array([[2,0],[0,2]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([1,2])
cov_mat2 = np.array([[1,0],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1,2).T
print('number of supporting points...',clf2.n_support_ )

plt.legend(['Decision Line','Malignant','Benign'])


# In[232]:

plt.plot(xx, yy,'y')
plt.plot(radius,line,'m')
plt.title('Comparison of Decision Boundaries')
plt.legend(['SVM','Logistic Regression'])
plt.ylim([-10,15])
plt.xlim([-5,15])
plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8)
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8)


# In[ ]:




# In[194]:

pca.explained_variance_


# In[195]:

pca.explained_variance_ratio_


# In[196]:

.44272026+.18971182


# In[197]:

prediction_var = ['radius_mean','texture_mean','smoothness_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']


# In[198]:

[data1.columns[1:7]]


# In[199]:

prediction_var
type(prediction_var)


# Las variables predictoras serán usadas para el modelo de predicción, para ello se divide el dataframe en 2, uno de entrenamiento y otro de predicción:
# 

# ### Random Forest

# In[200]:

train, test = train_test_split(data1, test_size = 0.3)# in this our main data is splitted into train and test
# we can check their dimension
print(train.shape)
print(test.shape)


# In[201]:

prediction_var


# In[ ]:




# In[ ]:




# In[202]:

train_X=train[prediction_var]
# taking the training data input 
train_y=train.diagnosis# This is output of our training data
# same we have to do for test
test_X= test[prediction_var] # taking test data inputs
test_y =test.diagnosis   #output value of test dat


# In[203]:

model=RandomForestClassifier(n_estimators=100)# a simple random forest model


# In[204]:

model.fit(train_X,train_y)# now fit our model for traiing data


# In[205]:

prediction=model.predict(test_X)# predict for the test data
# prediction will contain the predicted value by our model predicted values of dignosis column for test inputs


# In[206]:

metrics.accuracy_score(prediction,test_y) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values


# In[ ]:




# ### SMV

# In[207]:

model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# # Análisis cruzado
# 

# In[208]:

def model(model,data,prediction,outcome):
    # This function will be used for to check accuracy of different model
    # model is the m
    kf = KFold(data.shape[0], n_folds=10) # if you have refer the link then you must understand what is n_folds


# In[209]:

prediction_var


# Construcción de una función para evaluar el ajuste de los modelos mediante validación cruzada.

# In[210]:

# As we are going to use many models lets make a function
# Which we can use with different models
def classification_model(model,data,prediction_input,output):
    # here the model means the model 
    # data is used for the data 
    #prediction_input means the inputs used for prediction
    # output mean the value which are to be predicted
    # here we will try to find out the Accuarcy of model by using same data for fiiting and 
    #comparison for same data
    #Fit the model:
    model.fit(data[prediction_input],data[output]) #Here we fit the model using training set
  
    #Make predictions on training set:
    predictions = model.predict(data[prediction_input])
  
    #Print accuracy
    # now checkin accuracy for same data
    accuracy = metrics.accuracy_score(predictions,data[output])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
 
    
    kf = KFold(data.shape[0], n_folds=5)
    # About cross validitaion please follow this link
    #https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/
    #let me explain a little bit data.shape[0] means number of rows in data
    #n_folds is for number of folds
    error = []
    for train, test in kf:
        # as the data is divided into train and test using KFold
        # now as explained above we have fit many models 
        # so here also we are going to fit model
        #in the cross validation the data in train and test will change for evry iteration
        train_X = (data[prediction_input].iloc[train,:])# in this iloc is used for index of trainig data
        # here iloc[train,:] means all row in train in kf amd the all columns
        train_y = data[output].iloc[train]# here is only column so it repersenting only row in train
        # Training the algorithm using the predictors and target.
        model.fit(train_X, train_y)
    
        # now do this for test data also
        test_X=data[prediction_input].iloc[test,:]
        test_y=data[output].iloc[test]
        error.append(model.score(test_X,test_y))
        # printing the score 
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
    


# In[217]:

#Clasificador de árboles de decisiones
model = DecisionTreeClassifier()
prediction_var 
outcome_var= "diagnosis"
classification_model(model,data,prediction_var,outcome_var)


# In[218]:

#Clasificador de vector de soportes
model = svm.SVC()

classification_model(model,data,prediction_var,outcome_var)


# In[219]:

#Vecino más cercano
model = KNeighborsClassifier()
classification_model(model,data,prediction_var,outcome_var)


# In[220]:

# same here cross validation scores are not good
# now move to RandomForestclassifier
#Clasificador de bosque aleatorio
model = RandomForestClassifier(n_estimators=100)
classification_model(model,data,prediction_var,outcome_var)


# In[221]:

#Regresión logística
model=LogisticRegression()
classification_model(model,data,prediction_var,outcome_var)


# In[222]:

#Naive Bayes Bayesianoingenuo
model = GaussianNB()
classification_model(model,data,prediction_var,outcome_var)


# In[226]:

#Adaboost
# Using decision stumps due to size of sample.
# Attempting to prevent over-fitting
stump_clf =  DecisionTreeClassifier(random_state=42, max_depth=1)
model = AdaBoostClassifier(base_estimator = stump_clf)
classification_model(model,data,prediction_var,outcome_var)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



