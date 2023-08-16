import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection

import mytorch

iris = sns.load_dataset("iris")
#g = sns.pairplot(iris, hue="species")
df = iris[iris.species != "setosa"]
g = sns.pairplot(df, hue="species")
df['species_n'] = iris.species.map({'versicolor':1, 'virginica':2})

# Y = 'petal_length', 'petal_width'; X = 'sepal_length', 'sepal_width')
X_iris = np.asarray(df.loc[:, ['sepal_length', 'sepal_width']], dtype=np.float32)
Y_iris = np.asarray(df.loc[:, ['petal_length', 'petal_width']], dtype=np.float32)
label_iris = np.asarray(df.species_n, dtype=int)

# Scale
from sklearn.preprocessing import StandardScaler
scalerx, scalery = StandardScaler(), StandardScaler()
X_iris = scalerx.fit_transform(X_iris)
Y_iris = StandardScaler().fit_transform(Y_iris)

# Split train test
X_iris_tr, X_iris_val, Y_iris_tr, Y_iris_val, label_iris_tr, label_iris_val = \
    sklearn.model_selection.train_test_split(X_iris, Y_iris, label_iris, train_size=0.5, stratify=label_iris)
lr = 1e-4
X = mytorch.MyArray(X_iris)
Y = mytorch.MyArray(Y_iris)
l1 = mytorch.LinearLayer(in_features=2, out_features=100)
l3 = mytorch.LinearLayer(in_features=100, out_features=Y.shape[1])
# w1 = mytorch.MyArray(np.random.randn(2, 100), requires_grad=True)
l2 = mytorch.ReLU()
w2 = mytorch.MyArray(np.random.randn(100, Y.shape[1]), requires_grad=True)
for _ in range(40):
    o1 = l1(X)
    o2 = l2(o1)
    y_pred = l3(o2)
    loss = y_pred-Y
    loss = loss.square()
    ls = loss.sum()
    l1.w.zero_grad()
    l3.w.zero_grad()
    loss.backward()
    if _ == 0 or _ == 39:
        print("loss for epoch ",_,ls)
    # w2 -= lr * w2.grad
    l1.w -= lr * l1.w.grad
    l3.w -= lr * l3.w.grad