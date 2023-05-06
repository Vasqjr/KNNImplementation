#This is a sample KNN Implementation Script

#Import All Of The Required Packages For KNN Implementation Into The Program
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


#Loads The Iris Dataset Which Returns A Bunch Object Containing The Dataset's Attributes. Then Creates a Pandas DataFrame from the initial data.
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

#Extracts The feature values(X) and target values(y) from the DataFrame. It then splits the data into training and testing sets using the train_test_split.
#Test Size Parameter is 0.3 = 30% of data is used for testing, and 70% is used for training.
#Random_state parameter set to 42 ensures reproducibility of the results.
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Normalizes the feature values using scikit-learn's StandardScaler class.
#First creates an instance of the class, then uses fit_transform to fit the scaler on the training set.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Creates an instance of the KNeighborsClassifier class with k = 3 neighbors.
#Fits the model on the normalized training data.
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

#Use the trained model to predict the target values of the testing set using the predict method
#Uses scikit-learn's accuracy_score function, computes the accuracy of the model.
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#Finally, uses Matplotlib.pyplot to visualize the data as scatter plots.
#Creates a figure with four subplots(nrows = 2, ncols = 2), sets their size to 10 * 10, flatten the axes array so we can iterate over it easily.
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 20))
axs = axs.flatten()

plot_num = 0
for i in range(df.shape[1]-1):
    for j in range(i+1, df.shape[1]-1):
        if (df.iloc[:, i].nunique() > 1) and (df.iloc[:, j].nunique() > 1):
            axs[plot_num].scatter(df.iloc[:, i], df.iloc[:, j], c=df['target'], cmap='viridis')
            axs[plot_num].set_xlabel(iris.feature_names[i])
            axs[plot_num].set_ylabel(iris.feature_names[j])
            axs[plot_num].set_title(f"{iris.feature_names[i]} vs. {iris.feature_names[j]}")
            plot_num += 1

plt.show()


