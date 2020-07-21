import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
import itertools

"""
Logistic Regression

Logistic regression differs from linear regression in the fact that linear regression deals with estimating continuous values, whereas logistic regression deals with discrete categories.
"""

#solver: newton-cg, lbfgs, liblinear, sag, saga
regulizer = [0.01, 0.02, 0.05, 0.10, 0.25]
solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]

churn_df = pd.read_csv("ChurnData.csv")

X = np.asarray(churn_df[["tenure","age","address","income","ed",
                         "employ","equip"]])
y = np.asarray(churn_df["churn"])

# Normalize dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Train / Test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                   random_state=4)
#print("Train set:", X_train.shape, y_train.shape)
#print("Test set:", X_test.shape, y_test.shape)

for s in solver:
    for r in regulizer:
        # Modeling (Logistic Regression with Scikit-learn)
        # C parameter indicates inverse of regularization strength.
        LR = LogisticRegression(C=r, solver=s).fit(
            X_train,y_train)
        print(LR)

        yhat = LR.predict(X_test)
        #print(yhat)

        # predict_proba returns estimates for all classes, ordered by the label
        # of classes. So the first column is the probability of class 1,
        # P(Y=1|X)
        yhat_prob = LR.predict_proba(X_test)
        #print(yhat_prob)

        """
        Evaluation
        """

        # Jaccard index - size of intersection divided by size of union
        # of two label sets.
        print("Jaccard Score:", jaccard_score(y_test, yhat))


        # Another way of looking at accuracy of classifier is
        # the confusion matrix
        def plot_confusion_matrix(cm, classes, normalize=False,
                                  title="Confusion Matrix",
                                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting 'normalize=True'.
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
            for i, j in itertools.product(range(cm.shape[0]),
                                          range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel("True label")
            plt.xlabel("Predicted label")

        print(confusion_matrix(y_test, yhat, labels=[1,0]))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=["churn=1","churn=0"],
                              normalize=False,
                              title="Confusion matrix")
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
        print(log_loss(y_test, yhat_prob))
