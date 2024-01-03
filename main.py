import pandas as pd             #import pandas with the alias pd
import numpy as np              #import numpy with the alias np
import seaborn as sns           #import seaborn with the alias sns
import scipy.stats as ss
import matplotlib.pyplot as plt #import matplotlib.pyplot with the alias plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.neural_network
import sklearn.model_selection
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

import sklearn.tree
import sklearn.ensemble


from numpy.random import default_rng

file = pd.read_csv("./dataset_train.csv", header = None)



##PREPARE DATA
X = np.array(file.iloc[:,1:258])

y = np.array(file.iloc[:,258])


Xm = X.mean(axis=0)
X = X - Xm
nclass = 4
nx = len(X)
n_feat = len(X[0])

column = []
for i in range(0,len(X[0])):
    column.append("Feat " + str(i)) 
xd = pd.DataFrame(X,columns=column)
xl = pd.DataFrame(y,columns=['class'])
xdl = xd.join(xl)
#check dataframe
print(xdl.head())

xdl_small = xdl[['Feat 1', "Feat 2", "Feat 3", "Feat 4", "class"]]
sns.pairplot(xdl_small, hue="class")
plt.draw()

trainsize= 0.7
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= trainsize, random_state=5, shuffle=True, stratify = y)

# Initialize lists to store errors
linear_classifier_train_errors = []
linear_classifier_test_errors = []
quadratic_classifier_train_errors = []
quadratic_classifier_test_errors = []
for nfeat in range (1,100):
  pca = PCA(n_components=nfeat)
  pca.fit(X_train, y_train)
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)
  # Train linear classifier
  linear_classifier = LinearDiscriminantAnalysis()
  linear_classifier.fit(X_train_pca, y_train)

  # Train quadratic classifier
  quadratic_classifier = QuadraticDiscriminantAnalysis()
  quadratic_classifier.fit(X_train_pca, y_train)

  # Compute training and test errors for both classifiers
  linear_train_error = 1 - linear_classifier.score(X_train_pca, y_train)
  linear_test_error = 1 - linear_classifier.score(X_test_pca, y_test)
  quadratic_train_error = 1 - quadratic_classifier.score(X_train_pca, y_train)
  quadratic_test_error = 1 - quadratic_classifier.score(X_test_pca, y_test)

  # Append errors to lists
  linear_classifier_train_errors.append(linear_train_error)
  linear_classifier_test_errors.append(linear_test_error)
  quadratic_classifier_train_errors.append(quadratic_train_error)
  quadratic_classifier_test_errors.append(quadratic_test_error)
  


min_quadratic_train_error = min(quadratic_classifier_train_errors)
min_quadratic_test_error = min(quadratic_classifier_test_errors)

# Troba el valor òptim de n_features segons els errors del discriminador quadràtic
optimal_nfeat = quadratic_classifier_test_errors.index(min_quadratic_test_error) + 1  # +1 per ajustar l'índex

plt.figure(figsize=(10, 6))
plt.plot(range(1, 100), linear_classifier_train_errors, label='Linear Classifier (Train)', color='blue')
plt.plot(range(1, 100), linear_classifier_test_errors, label='Linear Classifier (Test)', color='red')
plt.plot(range(1, 100), quadratic_classifier_train_errors, label='Quadratic Classifier (Train)', color='green')
plt.plot(range(1, 100), quadratic_classifier_test_errors, label='Quadratic Classifier (Test)', color='purple')
plt.axvline(x=optimal_nfeat, color='gray', linestyle='--', label=f'Optimal n_features = {optimal_nfeat}')

plt.xlabel('Number of Features (nfeat)')
plt.ylabel('Classification Error')
plt.legend()
plt.title('Training and Test Errors vs. Number of Features')
plt.grid(True)
plt.show()

pca = PCA(n_components=optimal_nfeat)
pca.fit(X_train, y_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# # Loop over different numbers of features
# for nfeat in range(1, 4):
#     # Apply MDA for feature selection
#     mda = LinearDiscriminantAnalysis(n_components=nfeat)
#     mda.fit(X_train, y_train)
#     X_train_mda2 = mda.transform(X_train)
#     X_test_mda2  = mda.transform(X_test)

#     # Train linear classifier
#     linear_classifier = LinearDiscriminantAnalysis()
#     linear_classifier.fit(X_train_mda2, y_train)

#     # Train quadratic classifier
#     quadratic_classifier = QuadraticDiscriminantAnalysis()
#     quadratic_classifier.fit(X_train_mda2, y_train)

#     # Compute training and test errors for both classifiers
#     linear_train_error = 1 - linear_classifier.score(X_train_mda2, y_train)
#     linear_test_error = 1 - linear_classifier.score(X_test_mda2, y_test)
#     quadratic_train_error = 1 - quadratic_classifier.score(X_train_mda2, y_train)
#     quadratic_test_error = 1 - quadratic_classifier.score(X_test_mda2, y_test)

#     # Append errors to lists
#     linear_classifier_train_errors.append(linear_train_error)
#     linear_classifier_test_errors.append(linear_test_error)
#     quadratic_classifier_train_errors.append(quadratic_train_error)
#     quadratic_classifier_test_errors.append(quadratic_test_error)

# # Create a plot to visualize the error curves
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 4), linear_classifier_train_errors, label='Linear Classifier (Train)', color='blue')
# plt.plot(range(1, 4), linear_classifier_test_errors, label='Linear Classifier (Test)', color='red')
# plt.plot(range(1, 4), quadratic_classifier_train_errors, label='Quadratic Classifier (Train)', color='green')
# plt.plot(range(1, 4), quadratic_classifier_test_errors, label='Quadratic Classifier (Test)', color='purple')
# plt.xlabel('Number of Features (nfeat)')
# plt.ylabel('Classification Error')
# plt.legend()
# plt.title('Training and Test Errors vs. Number of Features')
# plt.grid(True)
# plt.show()

# mda = LinearDiscriminantAnalysis(n_components=3)
# mda.fit(X_train, y_train)
# X_train_mda2 = mda.transform(X_train)
# X_test_mda2  = mda.transform(X_test)

# # Percentage of variance explained for each components
# print('MDA PROJECTION TO 2D')
# print('explained variance ratio (first two components): %s'
#       % str(mda.explained_variance_ratio_))

# # Train and test LDA and QDA classifiers
# # LDA 2D

#def lda_classifier(X_train, y_train, X_test, y_test):

  # Linear Discriminant Analysis
#   lda = LinearDiscriminantAnalysis(solver="svd",store_covariance=True)
#   ldamodel = lda.fit(X_train, y_train)
#   y_tpred_lda = ldamodel.predict(X_train)
#   y_testpred_lda = ldamodel.predict(X_test)

#   lda_train_error = 1. - accuracy_score(y_train,y_tpred_lda)
#   lda_train_cmat = confusion_matrix(y_train,y_tpred_lda)

#   lda_test_error = 1. - accuracy_score(y_test,y_testpred_lda)
#   lda_test_cmat = confusion_matrix(y_test,y_testpred_lda)

#   lda_error = np.array([lda_train_error, lda_test_error])
#   lda_cmat  = np.array([lda_train_cmat, lda_test_cmat])

#   return lda, lda_error, lda_cmat

# def qda_classifier(X_train, y_train, X_test, y_test):
#   # Quadratic Discriminant Analysis
#   qda = QuadraticDiscriminantAnalysis(store_covariance=True)
#   qdamodel = qda.fit(X_train, y_train)
#   y_tpred_qda = qdamodel.predict(X_train)
#   y_testpred_qda = qdamodel.predict(X_test)

#   qda_train_error = 1. - accuracy_score(y_train,y_tpred_qda)
#   qda_train_cmat = confusion_matrix(y_train,y_tpred_qda)

#   qda_test_error = 1. - accuracy_score(y_test,y_testpred_qda)
#   qda_test_cmat = confusion_matrix(y_test,y_testpred_qda)

#   qda_error = np.array([qda_train_error, qda_test_error])
#   qda_cmat  = np.array([qda_train_cmat, qda_test_cmat])

#   return qda, qda_error, qda_cmat
# lda, lda_error, lda_cmat = lda_classifier(X_train_mda2,y_train,X_test_mda2,y_test)
# print('LDA train error: %f ' %lda_error[0])
# print('LDA train confusion matrix:')
# print(lda_cmat[0])
# print('LDA test error: %f ' %lda_error[1] )
# print('LDA test confusion matrix:')
# print(lda_cmat[1])

# # QDA 2D
# qda, qda_error, qda_cmat = qda_classifier(X_train_mda2,y_train,X_test_mda2,y_test)
# print('QDA train error: %f ' %qda_error[0])
# print('QDA train confusion matrix:')
# print(qda_cmat[0])
# print('QDA test error: %f ' %qda_error[1] )
# print('QDA test confusion matrix:')
# print(qda_cmat[1])



# model = sklearn.pipeline.Pipeline([
#     ("scaling", sklearn.preprocessing.MinMaxScaler()),
#     ("clf", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(10,),solver='sgd',momentum=0))
# ])
# model.fit(X_train_mda2, y_train)
# pred_train = model.predict(X_train_mda2)
# pred_test = model.predict(X_test_mda2)

# print("TRAINING\n" + classification_report(y_train, pred_train))
# print("\nTESTING\n" + classification_report(y_test, pred_test))

# train_acc = accuracy_score(y_train, pred_train)
# train_cmat = confusion_matrix(y_train, pred_train)
# test_acc = accuracy_score(y_test, pred_test)
# test_cmat = confusion_matrix(y_test, pred_test)
# print('train accuracy: ', train_acc)
# print('train error:    ', 1. - train_acc)
# print('test accuracy:  ', test_acc)
# print('test error:     ', 1. - test_acc)

# plt.subplots(figsize=(6,6))
# sns.heatmap(train_cmat, square=True, annot=True, cbar=False, fmt="d")
# plt.title('Confusion Matrix - Training')
# plt.xlabel('predicted value')
# plt.ylabel('true value');
# plt.show()

# plt.subplots(figsize=(6,6))
# sns.heatmap(test_cmat, square=True, annot=True, cbar=False, fmt="d")
# plt.title('Confusion Matrix - Testing')
# plt.xlabel('predicted value')
# plt.ylabel('true value');
# plt.show()



# params = [{'solver' : 'sgd', 'learning_rate' : 'constant', 'momentum' : 0, 'learning_rate_init' : 0.2},
#           {'solver' : 'sgd', 'learning_rate' : 'constant', 'momentum' : .9, 'nesterovs_momentum' : False, 'learning_rate_init' : 0.2},
#           {'solver' : 'sgd', 'learning_rate' : 'constant', 'momentum' : .9, 'nesterovs_momentum' : True, 'learning_rate_init' : 0.2},
#           {'solver' : 'sgd', 'learning_rate' : 'invscaling', 'momentum' : 0, 'learning_rate_init' :0.2},
#           {'solver' : 'sgd', 'learning_rate' : 'invscaling', 'momentum' : .9, 'nesterovs_momentum' : False, 'learning_rate_init' : 0.2},
#           {'solver' : 'sgd', 'learning_rate' : 'invscaling', 'momentum' : .9, 'nesterovs_momentum': True, 'learning_rate_init': 0.2},
#           {'solver' : 'adam', 'learning_rate_init' : 0.01,}]

# labels = ["constant learning-rate", "constant with momentum",
#           "constant with Nesterov's momentum",
#           "inv-scaling learning-rate", "inv-scaling with momentum",
#           "inv-scaling with Nesterov's momentum", "adam"]

# def plot_on_dataset(X,y):
#   print("/nLearning on MNIST dataset/n")

#   X = MinMaxScaler().fit_transform(X)
#   mlps = []
#   max_iter = 500

#   for label, param in zip(labels, params):
#     print("Training: %s" % label)
#     mlp = MLPClassifier(random_state = 0, max_iter = max_iter, **param)

#     #some parameter combinations will not converge so they will be ignored:
#     with warnings.catch_warnings():
#       warnings.filterwarnings("ignore", category = ConvergenceWarning, module = 'sklearn')
#       mlp.fit(X,y)

#     mlps.append(mlp)
#     print("Training set score: %f" % mlp.score(X,y))
#     print("Training set loss: %f" % mlp.loss_)

#   return mlps

# mlps = []
# mlps = plot_on_dataset(*(X_train, y_train))


# plot_args = [{'c' : 'red', 'linestyle' : '-'},
#              {'c' : 'green', 'linestyle' : '-'},
#              {'c' : 'blue', 'linestyle' : '-'},
#              {'c' : 'red', 'linestyle' : '--'},
#              {'c' : 'green', 'linestyle' : '--'},
#              {'c' : 'blue', 'linestyle' : '--'},
#              {'c' : 'black', 'linestyle' : '-'}]

# fig = plt.plot(figsize = (15,10))
# axes = plt.gca()

# for mlp, label, args in zip(mlps, labels, plot_args):
#   axes.plot(mlp.loss_curve_, label = label, **args)

# axes.set_title("MNIST Dataset")
# axes.legend(axes.get_lines(), labels)
# plt.show()





model2 = sklearn.pipeline.Pipeline([
    ("scaling", sklearn.preprocessing.MinMaxScaler()),
    ("clf", sklearn.neural_network.MLPClassifier(max_iter = 100, solver="adam")) #100 epochs
])

hidden_layer_sizes = [(100,), (100, 100), (100,100,100)]
activation = ["logistic", "relu"]
alpha = [0.0001, 0.001, 0.01]
batch_size = ["auto", 16]
learning_rate = ["constant", "adaptive"]
learning_rate_init = [0.001, 0.01]

grid_search = sklearn.model_selection.GridSearchCV(estimator= model2,
                                                   param_grid={"clf__hidden_layer_sizes" : hidden_layer_sizes,
                                                               "clf__alpha" : alpha,
                                                               "clf__learning_rate_init" : learning_rate_init,
                                                               "clf__activation" : activation,
                                                               "clf__batch_size" : batch_size,
                                                               "clf__learning_rate" : learning_rate,
                                                              
                                                               },
                                                   cv=sklearn.model_selection.ShuffleSplit(n_splits=1, train_size=0.75, random_state=1),
                                                   return_train_score = True,
                                                   n_jobs = 4,
                                                   verbose = 1)
grid_search.fit(X_train_pca, y_train)

pred_train = grid_search.predict(X_train_pca)
pred_test = grid_search.predict(X_test_pca)

print("TRAINING\n" + classification_report(y_train, pred_train))
print("\nTESTING\n" + classification_report(y_test, pred_test))

train_acc = accuracy_score(y_train, pred_train)
train_cmat = confusion_matrix(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
test_cmat = confusion_matrix(y_test, pred_test)
print('train accuracy: ', train_acc)
print('train error:    ', 1. - train_acc)
print('test accuracy:  ', test_acc)
print('test error:     ', 1. - test_acc)

plt.subplots(figsize=(6,6))
sns.heatmap(train_cmat, square=True, annot=True, cbar=False, fmt="d")
plt.title('Confusion Matrix - Training')
plt.xlabel('predicted value')
plt.ylabel('true value');
plt.show()

plt.subplots(figsize=(6,6))
sns.heatmap(test_cmat, square=True, annot=True, cbar=False, fmt="d")
plt.title('Confusion Matrix - Testing')
plt.xlabel('predicted value')
plt.ylabel('true value');
plt.show()




clf = sklearn.tree.DecisionTreeClassifier(random_state=1)
clf.fit(X_train_pca, y_train)
pred_train = clf.predict(X_train_pca)
pred_test = clf.predict(X_test_pca)
train_error = 1. - accuracy_score(y_train, pred_train)
train_cmat = confusion_matrix(y_train, pred_train)
test_error = 1. - accuracy_score(y_test, pred_test)
test_cmat = confusion_matrix(y_test, pred_test)
print("TRAINING\n" + classification_report(y_train, pred_train))
print("\nTESTING\n" + classification_report(y_test, pred_test))
print('train error: %f ' % train_error)
print('train confusion matrix:')
print(train_cmat)
print('test error: %f ' % test_error)
print('test confusion matrix:')
print(test_cmat)