import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
import six
from six import StringIO
import pydotplus
from matplotlib import image as mpimg
from sklearn import tree

my_data = pd.read_csv("drug200.csv", delimiter=",")
print(my_data[0:5])
print(my_data.size)

X = my_data[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]].values
print(X[0:5])

# pandas.get_dummies() converts categorical variables into dummy /
# indicator variables.

le_sex = preprocessing.LabelEncoder()
le_sex.fit(["F","M"])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(["LOW","NORMAL","HIGH"])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(["NORMAL","HIGH"])
X[:,3] = le_Chol.transform(X[:,3])

print(X[0:5])

y = my_data["Drug"]
print(y[0:5])

# Setting up the Decision Tree

X_trainset, X_testset, y_trainset, y_testset = train_test_split(
    X, y, test_size=0.3, random_state=3)

print("X_trainset shape:", X_trainset.shape)
print("y_trainset shape:", y_trainset.shape)
print("X_testset shape:", X_testset.shape)
print("y_testset shape:", y_testset.shape)

# Modeling

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree)

drugTree.fit(X_trainset, y_trainset)

# Prediction

predTree = drugTree.predict(X_testset)
print("predTree:", predTree[0:5])
print("y_testset:", y_testset[0:5])

# Evaluation

print("DecisionTree's Accuracy: ", metrics.accuracy_score(
    y_testset, predTree))

# Visualization

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out = tree.export_graphviz(drugTree, feature_names=featureNames,
                           out_file=dot_data,
                           class_names=np.unique(y_trainset),
                           filled=True,
                           special_characters=True,
                           rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100,200))
plt.imshow(img,interpolation="nearest")
