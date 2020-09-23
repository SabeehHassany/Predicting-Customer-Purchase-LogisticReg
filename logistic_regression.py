"""
Logistic Regression takes advantage of logistic functions like the sigmoid to create almost like a threshold for points to be classified as an outcome. The predictors are projected onto an n-dimensional plane to do so. This dataset is another widely used dataset from Kaggle that represents certain metrics for users who saw an e-commerce/social media ad for a product and whether they bought it or not.

### Importing the libraries
These are the three go to libraries for most ML.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""### Importing the dataset
I imported the dataset through Pandas dataframe then using iloc assign everything besides the last column as our independent variable(s) or X and the last column as our dependent variable or Y. The name of the dataset has to be updated and it must be in the same folder as your .py file or uploaded on Jupyter Notebooks or Google Collab.
"""

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""### Splitting the dataset into the Training set and Test set
Here I used a normal 80/20 test size and also assign 'random_state' to 0 for consistency.
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""### Feature Scaling
We can feature scale our data to make it easier for our model to train on our data and give us accurate results that aren't shifted by the presence of extreme outliers. It's not necessary, but helpful for getting more accurate results.
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""### Training the Logistic Regression model on the Training set
Simiple fitting/training our model on train data.
"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

"""### Confusion Matrix
The confusion matrix is a useful metric for classification models to allow us to visualize the correct positive, false positive, false negative, and correct negative predictions as well as a ultimate accuracy score on the bottom. Here we acheive a whopping 92.5% accuracy!
"""

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accs = accuracy_score(y_test, y_pred)
print(accs)


"""### Visualising the Training set results
It's useful to first visualize our training set results so we can see how our model made wrong/correct predictions and in what way the data is split. Here we use a meshgrid to arrange our predictions in almost a scatter plot formation and use the enumerate function with a contour line to show where our model made a positive prediction (green area) and where it made a negative prediction(red area).
"""

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25), np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Training set Results')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

"""## Visualising the Test set results
Same thing as above but visualizing our test results.
"""

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25), np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Test set Results')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()