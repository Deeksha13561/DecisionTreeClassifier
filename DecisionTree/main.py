import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree

data = pd.read_csv(r'C:\Users\LENOVO\PycharmProjects\DecisionTreeClassifier\Iris.csv')
print(data.head(15))

#check if our data has any values that require cleaning
print(data.isnull().sum())
#set the ID column as index
data = data.set_index("Id")

#Encoding the target data column which is in text format
label_encode = LabelEncoder()
Encoded_Species = label_encode.fit_transform(data['Species'])
#Iris-setosa - 0,Iris-versicolor - 1, Iris-virginica - 2
print(data.head(5))

#Set the data for X and Y
#X has all rows, and all columns except the last
X = data.iloc[:,:-1].values
print(X.shape)

#Y is our target columns, which has sample outputs to train our model
Y = Encoded_Species
print(Y.shape)

#splitting data into 70% training and 30% test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

#Building the decison tree on X and Y
Decison_tree = DecisionTreeClassifier()
Decison_tree.fit(X_train, y_train)

#Genertare predictions
predictions = Decison_tree.predict(X_test)

print(confusion_matrix(y_test,predictions))

#Testing our model on a random set of values
random_test = np.array([[7.3,2.9,6.3,1.8]])
# Predicted label should be Iris-virginica, label 2
pred_test = Decison_tree.predict(random_test)
print(pred_test)

#Visualizing Decision Trees
tree.plot_tree(Decison_tree);
#Adding features and class names to make the tree more attractive
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(Decison_tree,
               feature_names = fn,
               class_names=cn,
               filled = True);

#The decision tree is generated as an image which gets store in the project directory.
fig.savefig('img.png')