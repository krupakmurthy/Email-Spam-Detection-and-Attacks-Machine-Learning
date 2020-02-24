import sys
import json
import os
import re
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def get_ROC(result, test_labels):
    
    #Calculate TP,FP,TN,FN
    TP = FP = TN = FN = TPR = FPR = 0
    for i in range(len(test_labels)): 
        if test_labels[i]==result[i]==1:
           TP = TP + 1
        if result[i]==1 and test_labels[i]!=result[i]:
           FP = FP + 1
        if test_labels[i]==result[i]==0:
           TN = TN + 1
        if result[i]==0 and test_labels[i]!=result[i]:
           FN = FN + 1

    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    #auc = np.trapz(TPR,FPR)
    
    print(TP, TN, FN, FP)
    FPR, TPR, thresholds = metrics.roc_curve(test_labels, result, pos_label=0)
    plt.plot(TPR,FPR)
    plt.show()
    auc = np.trapz(FPR,TPR)
    print('AUC:', auc)
      



if __name__ == "__main__":
    tweets_folder_path = sys.argv[1]+'/'
    r=[]
    
    all_words = []
    
    for root, dirs, files in os.walk(tweets_folder_path):
        for name in files:
            r.append(os.path.join(root, name))

    for i in r:
        #print(i)
        test_file = open(i, 'r', encoding = "utf")
        for lines in test_file:
            words = lines.split()
            for word in words:
                all_words.append(word)


    #print(len(all_words))

    #Create Dictionary
    dictionary = Counter(all_words)
    #print(dictionary)
    #print("\n")
    dictionary = dictionary.most_common(30)
    #print(dictionary)
    dict_len = len(dictionary)
    #print(dict_len)


    #Extract Features
    #print(len(r))
    features_matrix = np.zeros((len(r),dict_len))
    #print(features_matrix)
    docID = 0;
    for i in r:
        #print(i)
        test_file = open(i, 'r', encoding = "utf")
        for i,line in enumerate(test_file):
            words = line.split()
            #print(words)
            for word in words:
                wordID = 0
                for i,d in enumerate(dictionary):
                    if d[0] == word:
                        wordID = i
                        features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1

        #print(features_matrix)
        #train classifiers

        #labeling
        train_labels = np.zeros((len(r)))
        #print(train_labels)

    for i in r:
        split = i.split("/")[-1]
        if split.startswith("spm"):
            #print(r.index(i))
            index = r.index(i)
            #print(index)
            train_labels[index] = 1
    #print(train_labels)
    train_matrix = features_matrix
    #print(train_matrix)


    #train classifiers
    classifier_NB = MultinomialNB()
    classifier_MT = LogisticRegression()
    classifier_SVC = LinearSVC()

    classifier_NB.fit(train_matrix,train_labels)
    classifier_MT.fit(train_matrix,train_labels)
    classifier_SVC.fit(train_matrix,train_labels)

    #test, follow same process as train
    test_dir = 'C:/Users/krupa krishnamurthy/NB/NB_MT/dataset/bare/TRIAL2/'

    all_words_test = []
    direc = []
    
    for root, dirs, files in os.walk(test_dir):
        for name in files:
            direc.append(os.path.join(root, name))

    #print(len(all_words))
    #print(len(direc))
    
    #Extract Features
    #print(len(direc))
    test_features_matrix = np.zeros((len(direc),dict_len))
    #print(features_matrix)
    docID = 0;
    for s in direc:
        print(s)
        test_file = open(s, 'r', encoding = "utf")
        for i,line in enumerate(test_file):
            words = line.split()
            #print(words)
            for word in words:
                #print(word)
                wordID = 0
                for i,d in enumerate(dictionary):
                    if d[0] == word:
                        wordID = i
                        test_features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1

    test_matrix = test_features_matrix

    #print(test_matrix)
    test_labels = np.zeros(len(direc))
    #print(test_labels)
    #print(len(direc))

    for i in direc:
        split = i.split("/")[-1]
        #print(split)
        if split.startswith("spm"):
            #print(r.index(i))
            index = direc.index(i)
            #print(index)
            test_labels[index] = 1
    print(test_labels)

    result1 = classifier_NB.predict(test_matrix)
    print(result1)
    get_ROC(result1, test_labels)

    result2 = classifier_MT.predict(test_matrix)
    print(result2)
    get_ROC(result2, test_labels)

    result3 = classifier_SVC.predict(test_matrix)
    print(result3)
    get_ROC(result3, test_labels)

    #print (confusion_matrix(test_labels,result1))
    #print (confusion_matrix(test_labels,result2))
    #print (confusion_matrix(test_labels,result3))

