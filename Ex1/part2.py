# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:50:33 2020

This scipt does active learning with multi-layer perceptron and nural network

@author: puyuan
"""
import random
from sklearn.svm import SVC, LinearSVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imageio as io
import os
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

origdata = pd.read_csv("Iris.csv")
k1, k2 = 'PetalLengthCm', 'PetalWidthCm'
data = origdata[[k1, k2, 'Species']].copy()
X = data[[k1, k2]]
y = data['Species']
y[y=='Iris-setosa'] = 0
y[y=='Iris-versicolor'] = 1
y[y=='Iris-virginica'] = 2

setosa = y == 0
versicolor = y == 1
virginica = y == 2

# We only want the second and third kinds of the flowers now.
X1 = X[y != 0]
y1 = y[y != 0]
X1 = X1.reset_index(drop=True)
y1 = y1.reset_index(drop=True)
y1 -= 1

# Get the ideal linear SVM
y1 = y1.astype(dtype=np.uint8)
clf0 = MLPClassifier(random_state=2, max_iter=300).fit(X1, y1)

xmin, xmax = X1[k1].min(), X1[k1].max()
ymin, ymax = X1[k2].min(), X1[k2].max()
stepx = (xmax - xmin)/99
stepy = (ymax - ymin)/99
# a0, b0, c0 = clf0.coef_[0, 0], clf0.coef_[0, 1], clf0.intercept_

# lx0 = [xmin + stepx * i for i in range(100)]
# ly0 = [-(a0*lx0[i] + c0)/b0 for i in range(100)]

X_pool, X_test, y_pool, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1)
X_pool, X_test, y_pool, y_test = X_pool.reset_index(drop=True), X_test.reset_index(drop=True), y_pool.reset_index(drop=True), y_test.reset_index(drop=True)

def find_most_ambiguous(clf, unknown_indexes):
    temp_1 = X_pool.iloc[unknown_indexes][k1].tolist()
    temp_2 = X_pool.iloc[unknown_indexes][k2].tolist()
    result = []
    for i in range(0,len(temp_1)):
        #print(clf.predict_proba([[temp_1[i], temp_2[i]]])[0][0])
        result.append(clf.predict_proba([[temp_1[i], temp_2[i]]])[0][0]-0.5)
    ind = np.argmin(abs(np.array(result)))
    return unknown_indexes[ind]

def plot_svm(clf, train_indexes, unknown_indexes, new_index = False, title = False, name = False):
    X_train = X_pool.iloc[train_indexes]
    y_train = y_pool.iloc[train_indexes]

    X_unk = X_pool.iloc[unknown_indexes]
    y_unk = y_pool.iloc[unknown_indexes]

    if new_index:
        X_new = X_pool.iloc[new_index]
        
    # a, b, c = clf.coef_[0, 0], clf.coef_[0, 1], clf.intercept_
    # Straight Line Formula
    # a*x + b*y + c = 0
    # y = -(a*x + c)/b

    # lx = [xmin + stepx * i for i in range(100)]
    # ly = [-(a*lx[i] + c)/b for i in range(100)]

    # fig = plt.figure(figsize=(9,6))

    # # plt.scatter(x[k1][setosa], x[k2][setosa], c='r')
    # plt.scatter(X_unk[k1], X_unk[k2], c='k', marker = '.')
    # plt.scatter(X_train[k1][y_train==0], X_train[k2][y_train==0], c='r', marker = 'o')
    # plt.scatter(X_train[k1][y_train==1], X_train[k2][y_train==1], c='c', marker = 'o')
    

    # plt.plot(lx, ly, c='m')
    # plt.plot(lx0, ly0, '--', c='g')
    
    # if new_index:
    #     plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
    #     plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
    #     plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
    #     plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
    #     plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)

    # if title:
    #     plt.title(title)
    
    # plt.xlabel(k1)
    # plt.ylabel(k2)

    # if name:
    #     fig.set_size_inches((9,6))
    #     plt.savefig(name, dpi=100)

    # plt.show()
    if new_index:
        total = len(unknown_indexes) +1
    else:
        total = len(unknown_indexes)
    correct = 0
    for i in unknown_indexes:
        temp_x = X_unk[k1][i]
        temp_y = X_unk[k2][i]
        #print(clf.predict_proba([[temp_x, temp_y]]))
        #print("I reach here")
        prediction = clf.predict_proba([[temp_x, temp_y]])
        if (y_unk[i] == 1):
             correct += prediction[0][1]
        if (y_unk[i] == 0):
             correct += prediction[0][0]
    if new_index:
        temp_x = X_pool.iloc[new_index][k1]
        temp_y = X_pool.iloc[new_index][k2]
        prediction = clf.predict_proba([[temp_x, temp_y]])
        if (y_pool.iloc[new_index] == 1):
             correct += prediction[0][1]
        if (y_pool.iloc[new_index] == 0):
             correct += prediction[0][0]
    return correct/total
    
ave_accuracy = []
for i in range(0,100):   
    train_indexes = list(range(10))
    unknown_indexes = list(range(10, 80))
    X_train = X_pool.iloc[train_indexes]
    y_train = y_pool.iloc[train_indexes]
    # mlp = MLPClassifier(max_iter=300)
    # clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    # clf.fit(X_train, y_train)
    # print('Best parameters found:\n', clf.best_params_)
    clf = MLPClassifier(activation='tanh', alpha= 0.0001, hidden_layer_sizes= (50,50,50), learning_rate= 'constant', solver= 'adam', random_state=1, max_iter=5000)  
    clf.fit(X_train, y_train)
    #folder = "rs1it5/"
    folder = "rs2it202/"
    # folder = "rs1it20/"
    
    try:
        os.mkdir(folder)
    except:
        pass
    
    filenames = []
    title = "Beginning"
    #name = folder + ("rs1it5_0a.jpg")
    name = folder + ("rs2it202_0a.jpg")
    plot_svm(clf, train_indexes, unknown_indexes, False, title, name)
    
    filenames.append(name)
    
    #n = find_most_ambiguous(clf, unknown_indexes)
    n = random.choice(unknown_indexes)
    unknown_indexes.remove(n)
    
    title = "Iteration 0"
    #name = folder + ("rs1it5_0b.jpg")
    name = folder + ("rs2it202_0b.jpg")
    filenames.append(name)
    plot_svm(clf, train_indexes, unknown_indexes, n, title, name)
    
    #num = 5
    num = 20
    t = []
    accuracy = [];
    for i in range(num):
        train_indexes.append(n)
        X_train = X_pool.iloc[train_indexes]
        y_train = y_pool.iloc[train_indexes]
        clf = MLPClassifier(activation='tanh', alpha= 0.0001, hidden_layer_sizes= (50,50,50), learning_rate= 'constant', solver= 'adam', random_state=1, max_iter=5000)
        clf.fit(X_train, y_train)
        #title, name = "Iteration "+str(i+1), folder + ("rs1it5_%d.jpg" % (i+1))
        title, name = "Iteration "+str(i+1), folder + ("rs2it202_%d.jpg" % (i+1))
        #n = find_most_ambiguous(clf, unknown_indexes)
        n = random.choice(unknown_indexes)
        unknown_indexes.remove(n)
        temp_accuracy= plot_svm(clf, train_indexes, unknown_indexes, n, title, name)
        accuracy.append(temp_accuracy)
        filenames.append(name)
    ave_accuracy.append(accuracy)

accuracy = np.average(ave_accuracy,axis = 0)
images = []
print(filenames)

# for filename in filenames:
#     images.append(io.imread(filename))
# #io.mimsave('rs1it5.gif', images, duration = 1)
# io.mimsave('rs2it202.gif', images, duration = 1)
# # io.mimsave('rs1it20.gif', images, duration = 1)
# try:
#     os.mkdir('rs1it5')
# #    os.mkdir('rt2it20')
# except:
#     pass
# os.listdir('rs1it5')
# fig = plt.figure(figsize=(9,6))
plt.title("MLP Accuracy Test with Active Learning by Random Sampling", fontsize = 20)
plt.xlabel("Number of Unlabeled Data Points Added to the Training Set", fontsize = 20)
plt.ylabel("Accuracy", fontsize = 20)
plt.plot(np.linspace(1,20, 20), accuracy)
plt.xticks(np.linspace(1,20, 20))
# SVM random accuracy = [0.8550724637681208, 0.8680294117647049, 0.8797014925373131, 0.8837575757575779, 0.8860307692307687, 0.8928125, 0.895428571428571, 0.8991290322580603, 0.9041967213114765, 0.9083999999999973, 0.912000000000001, 0.9148620689655153, 0.9191929824561449, 0.9232142857142843, 0.9239272727272705, 0.9256666666666722, 0.9292452830188628, 0.931538461538457, 0.9351764705882305, 0.9381199999999963]
# SVM confidence-based labeling = [0.8550724637681208, 0.9545588235294071, 0.9697014925373091, 0.9692121212121333, 0.9692307692307632, 0.98428125, 0.9836507936507966, 0.983870967741939, 0.9836065573770585, 0.9995000000000002, 0.9994915254237287, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# MLP random accuracy = 
# MLP confidence-based labeling = [0.9033202830246274, 0.9544219980395369, 0.9535954997142622, 0.9538158359733959, 0.9705478325964527, 0.9728021307990611, 0.9816443501903069, 0.9838533487039764, 0.9835987103930235, 0.9833142633575432, 0.9898514091303743, 0.9964223402634573, 0.9982654729470212, 0.9986867061804886, 0.999036386328053, 0.9992160154640822, 0.9993596810951472, 0.9993968726948064, 0.9993572114973281, 0.999299522419535]
#with open('rs1it5.gif','rb') as f:
#    display(Image(data=f.read(), format='gif'))