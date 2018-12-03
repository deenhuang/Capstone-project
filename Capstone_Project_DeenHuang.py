
# coding: utf-8

# <h1 align="center"> 
# Capstone Project - Default of Credit Card Clients Dataset
# </h1> 
# 
# <h2 align="center"> 
# By: Deen Huang
# </h2>
# <h3 align="center"> 
# December 2, 2018
# </h3>

# ## Import Libraries

# In[778]:


#import packages
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve

import warnings
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# ## Loading Data

# In[779]:


#loading data
data = pd.read_csv('UCI_Credit_Card.csv', header=0)
data.head()


# ## Exploratory Data Analysis

# In[780]:


data.rename(columns={'default.payment.next.month':'default'}, inplace=True)
data.rename(columns={'PAY_0':'PAY_1'}, inplace=True)
data = data.drop('ID', axis=1)
data.head()


# In[781]:


data.shape


# In[782]:


data.describe()


# In[783]:


data.isnull().sum()


# In[784]:


corr=data.corr()
corr = (corr)
#sns.set(font_scale=1.5)
plt.figure(figsize=(15,15))
hm = sns.heatmap(corr,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 10},
                 yticklabels=corr.columns.values,
                 xticklabels=corr.columns.values)

plt.title('Heatmap of Correlation')
plt.show()


# In[785]:


#check balance
plt.figure()
pd.Series(data['default']).value_counts().sort_index().plot(kind = 'bar')
plt.ylabel("value")
plt.xlabel("target")
plt.title('Number of default/non-default observations(0=non-default, 1=default)');


# In[786]:


# find the ratio for default and non-default target
default, non_default = data.default.value_counts()
print(data.default.value_counts())
print()

default = len(data[data['default']==1])
non_default = len(data[data['default']==0])

ratio = float(default/(default+non_default))
print('default Ratio :',ratio)


# In[787]:


#compare education to defaultable observations
new_var=data[['default', 'EDUCATION']]
new_var=new_var[new_var['EDUCATION']>=1]
sns.factorplot('default', col='EDUCATION', data=new_var, kind='count', size=3, aspect=.8, col_wrap=4);


# In[788]:


#compare marriage to defaultable observations
new_var=data[['default', 'MARRIAGE']]
new_var=new_var[new_var['MARRIAGE']>=1]
sns.factorplot('default', col='MARRIAGE', data=new_var, kind='count', size=3, aspect=.8, col_wrap=4);


# In[789]:


#compare sex to defaultable observations
new_var=data[['default', 'SEX']]
new_var=new_var[new_var['SEX']>=1]
sns.factorplot('default', col='SEX', data=new_var, kind='count', size=3, aspect=.8, col_wrap=4);


# ## Modeling

# ### Classical Machine Learning - Gaussian Naive Bayes

# In[790]:


# set the dataframe for features and target
X = data.drop(['default'], axis=1)
y = data['default']


# In[791]:


# split dataset into 20% of test data and 80% training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[792]:


# Normalize the data 

# rescaling the features to a standard normal distribution with a mean of 0 and standard deviation of 1
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train


# In[793]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[794]:


from sklearn.naive_bayes import GaussianNB

clf_GNB = GaussianNB()
clf_GNB.fit(X_train_scaled, y_train)


# In[795]:


y_pred = clf_GNB.predict(X_test_scaled)
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['GNB', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results


# In[796]:


cfm = confusion_matrix(y_test, y_pred)


# In[797]:


sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None)
plt.title('Gaussian Naive Bayes confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[798]:


print("Test data- Gaussian Naive Bayes report \n", classification_report(y_test, y_pred))


# In[799]:


# graph the ROC
def roc_curve_acc(Y_test, Y_pred, method):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, Y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',label='%s AUC = %0.3f'%(method, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'b--')
    plt.ylim([-0.1, 1.1])
    plt.xlim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

roc_curve_acc(Y_test, y_pred_GNB, "Gaussian Naive Bayes")


# ### Deep Learning - Neural Network

# In[800]:


#Process the imbalanced data set for neural network

#Create a new Class for non default
data.loc[data.default == 0, 'non_default'] = 1
data.loc[data.default == 1, 'non_default'] = 0

#Create dataframes of only default and non-default
default = data[data.default == 1]
non_default = data[data.non_default == 1]

# set the dataframe for features and target
X = data.drop(['default'], axis=1)
y = data['default']

# split dataset into 20% of test data and 80% training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Set X_train equal to 80% of default observations
X_train = default.sample(frac=0.8)

# Add 80% of the non-default observations to X_train
X_train = pd.concat([X_train, non_default.sample(frac = 0.8)], axis = 0)

# X_test contains all the observations which are not in X_train.
X_test = data.loc[~data.index.isin(X_train.index)]

#Shuffle the train data in a random order.
X_train = shuffle(X_train)
X_test = shuffle(X_test)

#Add target to y_train and y_test.
y_train = X_train.default
y_train = pd.concat([y_train, X_train.non_default], axis=1)

y_test = X_test.default
y_test = pd.concat([y_test, X_test.non_default], axis=1)

#Drop target from X_train and X_test.
X_train = X_train.drop(['default','non_default'], axis = 1)
X_test = X_test.drop(['default','non_default'], axis = 1)

#Names the features in X_train.
features = X_train.columns.values

# rescaling the features to a standard normal distribution with a mean of 0 and standard deviation of 1
for feature in features:
    mean, std = data[feature].mean(), data[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std
    
# Split the data set
split = int(len(y_test)/2)

X_train = X_train.as_matrix()
y_train = y_train.as_matrix()
X_valid = X_test.as_matrix()[:split]
y_valid = y_test.as_matrix()[:split]
X_test = X_test.as_matrix()[split:]
y_test = y_test.as_matrix()[split:]


# In[801]:


# build 5-hidden-layer NN and save the entire model

#set layer neurons and learning rate
H1_N = 23
H2_N = 23*2
H3_N = 23*3
H4_N = 23*4
H5_N = 23*5

learning_rate = 0.02

# declare the basic structure of the data
x = tf.placeholder("float", shape=[None, 23])
y_ = tf.placeholder("float", shape=[None, 2])

def weight_variable(shape, index):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, index)

def bias_variable(shape, index):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, index)

# layer 1
W_N1 = weight_variable(shape = [23,H1_N], index = 'W_N1')
b_N1 = bias_variable(shape = [H1_N], index = 'b_N1')
h_fc1 = tf.nn.sigmoid(tf.matmul(x, W_N1) + b_N1)

# layer 2
W_N2 = weight_variable(shape = [H1_N,H2_N], index = 'W_N2')
b_N2 = bias_variable(shape = [H2_N], index = 'b_N2')
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_N2) + b_N2)

# layer 3
W_N3 = weight_variable(shape = [H2_N,H3_N], index = 'W_N3')
b_N3 = bias_variable(shape = [H3_N], index = 'b_N3')
h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2, W_N3) + b_N3)

# layer 4
W_N4 = weight_variable(shape = [H3_N,H4_N], index = 'W_N4')
b_N4 = bias_variable(shape = [H4_N], index = 'b_N4')
h_fc4 = tf.nn.sigmoid(tf.matmul(h_fc3, W_N4) + b_N4)

# layer 5
W_N5 = weight_variable(shape = [H4_N,H5_N], index = 'W_N5')
b_N5 = bias_variable(shape = [H5_N], index = 'b_N5')
h_fc5 = tf.nn.sigmoid(tf.matmul(h_fc4, W_N5) + b_N5)

#output layer
W_N6 = weight_variable(shape = [H5_N,2], index = 'W_N6')
b_N6 = bias_variable(shape = [2], index = 'b_N6')
y = tf.nn.softmax(tf.matmul(h_fc5, W_N6) + b_N6)

# Parameters
training_epochs = 110
training_dropout = 0.8
n_samples = y_train.shape[0]
batch_size = 256
pkeep = tf.placeholder(tf.float32) 

# Cost function: Cross_entropy
# Loss
y_clipped = tf.clip_by_value(y, 1e-10, 0.9999999)
cost = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y_clipped)
                         + (1 - y_) * tf.log(1 - y_clipped), axis=1))

correct_prediction = tf.equal(tf.argmax(y,axis=1), tf.argmax(y_,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

###### Save the model
accuracyNew = []
costNew = []
validAccuracyNew = [] 
validCostNew = []
testAccuracyNew = []
testCostNew = []
    
# training cycles
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
    
for epoch in range(training_epochs): 
    for batch in range(int(n_samples/batch_size)):
        batch_x = inputX[batch*batch_size : (1+batch)*batch_size]
        batch_y = inputY[batch*batch_size : (1+batch)*batch_size]

        sess.run([optimizer], feed_dict={x: batch_x, 
                                         y_: batch_y,
                                         pkeep: training_dropout})

    # Display every 10 epochs
    if (epoch) % 10 == 0:
        train_accuracy, train_cost = sess.run([accuracy, cost], feed_dict={x: X_train, 
                                                                           y_: y_train,
                                                                           pkeep: training_dropout})

        valid_accuracy, valid_cost = sess.run([accuracy, cost], feed_dict={x: X_valid, 
                                                                           y_: y_valid,
                                                                           pkeep: training_dropout})
        
        test_accuracy, test_cost = sess.run([accuracy, cost], feed_dict={x: X_test, 
                                                                         y_: y_test,
                                                                         pkeep: training_dropout})

        print ("Epoch:", epoch,
               "Acc =", "{:.3f}".format(train_accuracy), 
               "Cost =", "{:.3f}".format(train_cost),
               "Valid_Acc =", "{:.3f}".format(valid_accuracy), 
               "Valid_Cost = ", "{:.3f}".format(valid_cost),
               "test_Acc =", "{:.3f}".format(test_accuracy), 
               "test_Cost = ", "{:.3f}".format(test_cost))
            
        # Store the results of the model
        accuracyNew.append(train_accuracy)
        costNew.append(train_cost)
        validAccuracyNew.append(valid_accuracy)
        validCostNew.append(valid_cost)
        testAccuracyNew.append(test_accuracy)
        testCostNew.append(test_cost)
            
print()
print("Optimization Finished!")
print()
    
#saver = tf.train.Saver()
#saver.save(sess, save_path='./DNN_well_trained.ckpt')


# In[802]:


# Plot the accuracy and cost function
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))

ax1.plot(accuracyNew) # blue
ax1.plot(validAccuracyNew) # red
ax1.plot(testAccuracyNew) # purple
ax1.set_title('Accuracy')

ax2.plot(costNew)
ax2.plot(validCostNew)
ax2.plot(testCostNew)
ax2.set_title('Cost')

plt.xlabel('Epoches')
plt.ylabel('Value')
plt.show()

