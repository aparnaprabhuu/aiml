from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

iris_datasets = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris_datasets["data"], iris_datasets["target"], random_state=0)
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, Y_train)

for i in range(len(X_test)):
    X = X_test[i]
    X_new = np.array([X])
    prediction = kn.predict(X_new)
    actual_class = iris_datasets["target_names"][Y_test[i]]
    predicted_class = iris_datasets["target_names"][prediction[0]]
    print("\nActual: {}, Predicted: {}".format(actual_class, predicted_class))

print("\nTest Score [Accuracy]: {:.2f}".format(kn.score(X_test, Y_test)))
