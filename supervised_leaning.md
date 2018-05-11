
# K-NN
```
import mglearn
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_text,y_train, y_test=train_test_split(X,y,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)
clf=clf.fit(X_train,y_train)
```
For each data point in the test set, this computes its nearest neighbors in the training set and finds the most common class among these:
```
print('Test set predictions:{}'.format(clf.predict(X_test)))
print("Test set accuracy:{:.2f}".format(clf.score(X_test,y_test)))
```

Considering more and more neighbors leads to a smoother decision boundary. 

A smoother boundary corresponds to a simpler model. In other words, using few neighbors corresponds to high model complexity (as shown on the left side of Figure 2-1), 

and using many neighbors corresponds to low model complexity (as shown on the right side of Figure 2-1).
```
%matplotlib inline
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(10,3))
for n_neighbors, ax in zip([1,3,9],axes):
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title('{}neighbor(s)'.format(n_neighbors))
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
axes[0].legend(loc=3)
```
```
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)
training_accuracy=[]
test_accuracy=[]
neighbor_settings=range(1,11)
for n_neighbors in neighbor_settings:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))
plt.plot(neighbor_settings,  training_accuracy,label='training accuracy')
plt.plot(neighbor_settings,test_accuracy,label='test accuracy')
plt.ylabel('accuracy')
plt.xlabel('neighbors')
plt.legend()

```
```
from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
reg=KNeighborsRegressor(n_neighbors=3)
reg=reg.fit(X_train,y_train)

print('Test set predictions:\n {}'.format(reg.predict(X_test)))

print ('Test set R^2:{:.2f}'.format(reg.score(X_test,y_test))
```
```
import numpy as np
fig,axes=plt.subplots(1,3,figsize=(15,4))

line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors,ax in zip([1,3,9],axes):
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(X_train,y_train,'^',c='blue',markersize=8)
    ax.plot(X_test,y_test,'o',c='red',markersize=8)
    ax.set_title('{}neighbors:\n train score: {:.2f} test score:{:.2f}'.format(n_neighbors, 
                                                                               reg.score(X_train,y_train),
                                                                               reg.score(X_test,y_test)))

    ax.set_xlabel('feature')
    ax.set_ylabel('target')
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"],loc='best')
             
```
In principle, there are two important parameters to the KNeighbors classifier: 

** the number of neighbors** and how you **measure distance between data points**. 

In practice, using a small number of neighbors like three or five often works well, but you should certainly adjust this parameter. 

Choosing the right distance measure is somewhat beyond the scope of this book. By default, Euclidean distance is used, which works well in many settings.

One of the strengths of k-NN is that the model is very easy to understand, and often gives reasonable performance without a lot of adjustments. Using this algorithm is a good baseline method to try before considering more advanced techniques. 

Building the nearest neighbors model is usually very fast, but when your training set is **very large** (either in number of features or in number of samples) prediction can be slow. 

When using the k-NN algorithm, it’s important to preprocess your data (see Chapter 3). This approach often **does not perform well on datasets with many features** (hundreds or more), 

and it does particularly badly with datasets where most features are 0 most of the time (so-called sparse datasets).

So, while the nearest k-neighbors algorithm is easy to understand, it is not often used in practice, 

due to prediction **being slow and its inability to handle many features**. The method we discuss next has neither of these drawbacks.
