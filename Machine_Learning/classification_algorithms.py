"""
Classification Algorithms
"""

import itertools
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn import metrics
import six
from six import StringIO
import pydotplus
from matplotlib import image as mpimg
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

df = pd.read_csv("basketball.csv")
print(df.head())
print(df.shape)

# Add a column that contains "true" if the wins above bubble are over 7,
# and "false" if not. We'll call this column Win Index, or
# windex for short.
df["windex"] = np.where(df.WAB > 7, "True", "False")

# Add a column that contains "true" if the team made it into the final four,
# and "false" otherwise.
df["F4"] = np.where(df.POSTSEASON == "F4", "True", "False")
# Data Visualization and Pre-Processing
# Filter the data set to the teams that made the Sweet Sixteen,
# the Elite Eight, and the Final Four in the post season.
# Create a new dataframe that will hold the values with a new column.
df1 = df.loc[df["POSTSEASON"].str.contains("F4|S16|E8", na=False)]
print(df1.head())
df1["POSTSEASON"].value_counts()

"""
# Plot columns to understand data better
bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1",
                  col_wrap=6)
g.map(plt.hist, "BARTHAG", bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1",
                  col_wrap=2)
g.map(plt.hist, "ADJOE", bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# Pre-processing: Feature Selection / Extraction

# Adjusted Defense Efficiency plot
bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1",
                  col_wrap=2)
g.map(plt.hist, "ADJDE", bins=bins, ec="k")
g.axes[-1].legend()
plt.show()
# This data point doesn't impact the ability of a team to get
# into the Final Four.
"""
# Convert Categorical Features to Numerical Values

# postseason
print(df1.groupby(["windex"])["POSTSEASON"].value_counts(normalize=True))
# 13% of teams with 6 or less wins above bubble make it into the final four,
# while 17% of teams with 7 or more do.

# Convert wins above bubble (winindex) under 7 to 0, and over 7 to 1:
df1["windex"].replace(to_replace=["False","True"], value=[0,1],
                      inplace=True)
# Convert F4 to 1, and otherwise to 0:
df1["F4"].replace(to_replace=["False","True"], value=[0,1],
                  inplace=True)
print(df1.head())

# Feature Selection

# Define feature sets, X:
X = df1[["G", "W", "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
         "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD", "2P_O", "2P_D",
         "3P_O", "3P_D", "ADJ_T", "WAB", "SEED", "windex"]]

# Labels
# Round where the given team was eliminated or where their season ended
# (R68 = First Four, R64 = Round of 64, R32 = Round of 32,
# S16 = Sweet Sixteen, E8 = Elite Eight, F4 = Final Four,
# 2ND = Runner-up, Champion = Winner of the NCAA March Madness Tournament
# for that given year).
y = df1["POSTSEASON"].values
print(y[0:5])

# Normalize Data
# Data standardization gives data zero mean and unit variance
# (technically should be done after train-test split).
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

# Training and Validation
# Split the data into training and validation data to find best k
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=4)
print("Train set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)

#==================
"""
K Nearest Neighbor
"""
#==================

# Build a KNN model using a value of k = 5, and find the accuracy on
# the validation data (X_val and y_val)
print("Train set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)



# Training
k = 5
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
print(neigh)

# Predicting
yhat = neigh.predict(X_val)
print(yhat[0:5])

# Accuracy evaluation
val_accuracy = accuracy_score(y_val, yhat)
print("Validation set Accuracy: ", val_accuracy)

"""
Calculate accuracy of other K values
"""
Ks = 15
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = []
for n in range(1, Ks):

    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = neigh.predict(X_val)
    mean_acc[n-1] = accuracy_score(y_val, yhat)

    std_acc[n-1] = np.std(yhat==y_val) / np.sqrt(yhat.shape[0])

print(mean_acc)

# Set model with k = 5
neigh = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)

#=============
"""
Decision Tree
"""
#=============
for depth in range(1,11):
    # Modeling

    bball_tree = DecisionTreeClassifier(criterion="entropy", max_depth = depth)
    print(bball_tree)

    bball_tree.fit(X_train, y_train)

    # Prediction

    predTree = bball_tree.predict(X_val)
    print("predTree:", predTree[0:5])
    print("y_val:", y_val[0:5])

    # Evaluation

    print("DecisionTree's Accuracy: ", accuracy_score(y_val, predTree))

# Best depth: 5 with accuracy 0.5
# Set model with best depth
bball_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 1)
bball_tree.fit(X_train, y_train)
predTree = bball_tree.predict(X_val)

# Visualization
print(X)
print(type(X))
dot_data = StringIO()
filename = "basketball_tree.png"
featureNames = [
"G", "W", "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
"TOR", "TORD", "ORB", "DRB", "FTR", "FTRD", "2P_O", "2P_D",
"3P_O", "3P_D", "ADJ_T", "WAB", "SEED", "windex"]

targetNames = df1["POSTSEASON"].unique().tolist()
out = tree.export_graphviz(bball_tree, feature_names=featureNames,
                           out_file=dot_data,
                           class_names=np.unique(y_train),
                           filled=True,
                           special_characters=True,
                           rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100,200))
plt.imshow(img,interpolation="nearest")



#=============
"""
SVM
"""
#=============

feature_df = df1[["G", "W", "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
         "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD", "2P_O", "2P_D",
         "3P_O", "3P_D", "ADJ_T", "WAB", "SEED"]]


X = np.asarray(feature_df)
print(X[0:5])


# We want the model to predict the value of 'Windex'
# As this field can have only one of two possible
# values, we need to change its measurement level to reflect this.
df1["windex"] = df1["windex"].astype("int")
y = np.asarray(df1["windex"])
print(y[0:5])


# Train / Test Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)
"""
# Available kernel functions: Linear; Polynomial:
# Radial basis function (RBF); Sigmoid
for ker in {"linear", "poly", "rbf", "sigmoid"}:
    clf = svm.SVC(kernel=ker)
    clf.fit(X_train, y_train)

    yhat = clf.predict(X_test)
    print(yhat[0:5])

    # Evaluation
    def plot_confusion_matrix(cm, classes, normalize=False,
                              title="Confusion matrix: ker=" + ker ,
                              cmap=plt.cm.Blues):
        ""
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting 'normalize=True'
        ""
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        print(cm)

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i,j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")


    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
    np.set_printoptions(precision=2)

    print(classification_report(y_test, yhat))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,
                          classes=["Lose(0)", "Win(1)"],
                          normalize=False,
                          title="Confusion Matrix: ker=" + ker)
    plt.show()
    print("F1 Score = ", f1_score(y_test, yhat, average="micro"))
    print("Jaccard = ", jaccard_score(y_test, yhat, pos_label=0))
"""

clf = svm.SVC(kernel="sigmoid")
clf.fit(X_train, y_train)



#===================
"""
Logistic Regression
"""
#===================

"""
Logistic Regression

Logistic regression differs from linear regression in the fact that linear regression deals with estimating continuous values, whereas logistic regression deals with discrete categories.
"""

#solver: newton-cg, lbfgs, liblinear, sag, saga
regulizer = [0.01, 0.02, 0.05, 0.10, 0.25]
solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]

"""
for s in solver:
    for r in regulizer:
        title="Confusion Matrix \n regulizer = " + str(r) + "\n solver = " + s
        # Modeling (Logistic Regression with Scikit-learn)
        # C parameter indicates inverse of regularization strength.
        LR = LogisticRegression(C=r, solver=s).fit(
            X_train,y_train)
        print(LR)

        yhat = LR.predict(X_test)
        print("yhat = ", yhat)

        # predict_proba returns estimates for all classes, ordered by the label
        # of classes. So the first column is the probability of class 1,
        # P(Y=1|X)
        yhat_prob = LR.predict_proba(X_test)
        print("yhat_prob - ", yhat_prob)

        ""
        Evaluation
        ""

        # Jaccard index - size of intersection divided by size of union
        # of two label sets.
        print("F1 Score = ", f1_score(y_test, yhat, average="micro"))
        print("Jaccard = ", jaccard_score(y_test, yhat, pos_label=0))

        # Another way of looking at accuracy of classifier is
        # the confusion matrix
        def plot_confusion_matrix(cm, classes, normalize=False,
                                  title=title,
                                  cmap=plt.cm.Blues):
            ""
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting 'normalize=True'.
            ""
            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print("Confusion matrix, without normalization")
            print(cm)

            plt.imshow(cm, interpolation="nearest", cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = ".2f" if normalize else "d"
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]),
                                          range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel("True label")
            plt.xlabel("Predicted label")

        print(confusion_matrix(y_test, yhat, labels=[0,1]))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=["lost=0","win=1"],
                              normalize=False,
                              title=title)
        plt.show()

        # Precision is a measure of the accuracy provided that a class label
        # has been predicted. It is defined by:
        # precision = TP / (TP + FP)

        # Recall is true positive rate. It is defined as:
        # recall = TP / (TP + FN)

        # The F1 score is the harmonic average of the precision and recall,
        # where an F1 score reaches its best value at 1 (perfect precision
        # and recall) and worst at 0. It is a good way to show that a
        # classifier has a good value for both recall and precision.

        print(classification_report(y_test, yhat))

        # Log loss - measures the performance of a classifier where the
        # predicted output is a probability value between 0 and 1.
        log_loss_LR = log_loss(y_test, yhat_prob)
        print("log loss = ", log_loss(y_test, yhat_prob))
        print("-----------------------------------------------------")

# For logistic regression, liblinear with c=0.25 has best
# log loss at 0.288
# Jaccard Index: 0.5
# f1-scores: 0.95 and 0.67
# Accuracy: 0.92
"""
LR = LogisticRegression(C=0.01, solver="liblinear").fit(
    X_train,y_train)


print(
"""
#==============================
Model Evaluation Using Test Set
#==============================
"""
)

test_df = pd.read_csv("basketball_test_set.csv", error_bad_lines=False)
print(test_df.head())

test_df["windex"] = np.where(test_df.WAB > 7, "True", "False")
test_df["F4"] = np.where(test_df.POSTSEASON == "F4", "True", "False")
test_df1 = test_df.loc[test_df["POSTSEASON"].str.contains(
    "F4|S16|E8", na=False)]
test_df1["windex"].replace(to_replace=["False","True"],
                               value=[0,1], inplace=True)
test_df1["F4"].replace(to_replace=["False","True"],
                       value=[0,1], inplace=True)
test_df1["YEAR"].replace(to_replace=["2019TEAM"], value=[2019], inplace=True)
test_X = test_df1[["G", "W", "ADJOE", "ADJDE", "BARTHAG", "EFG_O",
                         "EFG_D", "TOR", "TORD", "ORB", "DRB", "FTR",
                         "FTRD", "2P_O", "2P_D", "3P_O", "3P_D",
                         "ADJ_T", "WAB", "SEED", "windex"]]
test_X = preprocessing.StandardScaler().fit(test_X).transform(test_X)
print(test_X[0:5])

test_y = test_df1["POSTSEASON"].values
print(test_y[0:5]
      )

print("X_train " ,X_train.shape)
print("test_X ", test_X.shape)

def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1


# KNN
yhat = neigh.predict(test_X)
accuracy_knn = accuracy_score(test_y, yhat)
jaccard_knn = jaccard_index(test_y, yhat)
f1Score_knn = f1_score(test_y, yhat, average="micro")

# Decision Tree
predTree = bball_tree.predict(test_X)
accuracy_tree = accuracy_score(test_y, predTree)
jaccard_tree = jaccard_index(test_y, predTree)
f1Score_tree = f1_score(test_y, predTree, average="micro")

# SVM

test_df1["windex"] = df1["windex"].astype("int")
test_df1 = test_df1[test_df1["windex"].isin([0,1])]
test_X = test_df1[["G", "W", "ADJOE", "ADJDE", "BARTHAG", "EFG_O",
                         "EFG_D", "TOR", "TORD", "ORB", "DRB", "FTR",
                         "FTRD", "2P_O", "2P_D", "3P_O", "3P_D",
                         "ADJ_T", "WAB", "SEED"]]
X = np.asarray(feature_df)
y = np.asarray(test_df1["windex"])
yhat = clf.predict(X)
print("test_df1[windex]", test_df1["F4"])
print("y \n", y.shape, "\n",y)
print("yhat \n", yhat.shape, "\n", yhat)
accuracy_svm = accuracy_score(y, yhat)
f1Score_svm = f1_score(y, yhat)
jaccard_svm = jaccard_index(y, yhat)

# Logistic Regression
yhat = LR.predict(X)
yhat_prob = LR.predict_proba(X)
accuracy_LR = accuracy_score(y, yhat)
f1Score_LR = f1_score(y, yhat, average="micro")
jaccard_LR = jaccard_index(y, yhat)
log_loss_LR = log_loss(y, yhat_prob)

results_df = pd.DataFrame({
    "Algorithm": ["KNN", "Decision Tree", "SVM", "Logistic Regression"],
    "Accuracy": [accuracy_knn, accuracy_tree, accuracy_svm, accuracy_LR],
    "Jaccard": [jaccard_knn, jaccard_tree, jaccard_svm, jaccard_LR],
    "F1-score": [f1Score_knn, f1Score_tree, f1Score_svm, f1Score_LR],
    "LogLoss": ["N/A","N/A","N/A",log_loss_LR]
})

print(results_df)
