import pandas as pd
import pylab as pl
import numpy as np
import itertools
from scipy import optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head())

ax = cell_df[cell_df["Class"] == 4][0:50].plot(
    kind="scatter", x="Clump", y="UnifSize", color="DarkBlue",
    label="malignant")
cell_df[cell_df["Class"] == 2][0:50].plot(
    kind="scatter", x="Clump", y="UnifSize", color="Yellow",
    label="benign", ax=ax)
plt.show()

# Data pre-processing and selection
print(cell_df.dtypes)

# BareNuc column includes some values that are not numeric,
# so drop those rows.
cell_df = cell_df[pd.to_numeric(cell_df["BareNuc"], errors="coerce").
                  notnull()]
cell_df["BareNuc"] = cell_df["BareNuc"].astype("int")
print(cell_df.dtypes)

feature_df = cell_df[["Clump", "UnifSize", "UnifShape", "MargAdh",
                      "SingEpiSize", "BareNuc", "BlandChrom",
                      "NormNucl", "Mit"]]
X = np.asarray(feature_df)
print(X[0:5])


# We want the model to predict the value of Class (benign (=2)
# or malignant (=4). AS this field can have only one of two possible
# values, we need to change its measurement level to reflect this.
cell_df["Class"] = cell_df["Class"].astype("int")
y = np.asarray(cell_df["Class"])
print(y[0:5])


# Train / Test Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)

# Available kernel functions: Linear; Polynomial:
# Radial basis function (RBF); Sigmoid
clf = svm.SVC(kernel="rbf")
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print(yhat[0:5])

# Evaluation
def plot_confusion_matrix(cm, classes, normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'
    """
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
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=["Benign(2)", "Malignant(4)"],
                      normalize=False,
                      title="Confusion matrix")
plt.show()
print(f1_score(y_test, yhat, average="weighted"))
print(jaccard_score(y_test, yhat, pos_label=2))
