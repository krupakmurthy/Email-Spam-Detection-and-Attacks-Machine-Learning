import sys
import json
import os
import re
import time
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc



if __name__ == "__main__":
    
    folder_path = 'train/'
    r=[]
    dictionary_temp = []
    dictionary = []
    all_words = []
    all_words_dict = []
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            r.append(os.path.join(root, name))

    for i in r:
        #print(i)
        test_file = open(i, 'r', encoding = "utf")
        for lines in test_file:
            words = lines.split()
            for word in words:
                all_words.append(word)

    sum_tl = 0
    with open ("dictionary.txt", mode="r") as file:
        for lines in file:
            tl = eval(lines)
            tl0 = tl[0].strip()
            tl1 = tl[1]
            sum_tl = sum_tl + tl1
            dictionary_temp.append(tl)
    file.close()
            
    average = sum_tl / len(dictionary_temp)
    #print(int(average))
    with open ("dictionary.txt", mode="r") as file:
        
        for lines in file:
        
            #print(lines)
            lines = lines.split(",")
            lines_0 = lines[0][1:]
            lines_0 = lines_0[1:-1]
            #print(lines_0)
            lines_1 = int(average)
            #print(lines_1)
            data = lines_0,lines_1
            #print(data)
            '''
            with open ("dictionary_averaged.txt", mode = "a") as writefile:
                data = str(data)
                writefile.write(data)
                writefile.write("\n")
            writefile.close()
            '''
    file.close()
   
            
    with open("dictionary_averaged.txt", mode = "r") as file:
        for lines in file:
            #print(lines)
            tl = eval(lines)
            dictionary.append(tl)
            tl0 = tl[0].strip()
            all_words_dict.append(tl0)

    print(len(all_words_dict))
    

    dict_len = len(all_words_dict)
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

    classifier_NB.fit(train_matrix,train_labels)

    
    test_dir = 'C:/Users/krupa krishnamurthy/NB/NB_MT/dataset/bare/p2/dataset/data/'

    all_words_test = []
    test_folder = []
    spam_addr = []
    nonspam_addr = []
    
    for root, dirs, files in os.walk(test_dir):
        for name in files:
            test_folder.append(os.path.join(root, name))

    
    split_spam = []
    split_non = []
    m=0

    for i in test_folder:
        if 'spm' in i:
            #print(i)
            spam_addr.append(i)
        else:
            nonspam_addr.append(i)
            #print(i)

    for i in range(len(spam_addr)):
                
        test_file_spam = open(spam_addr[i], 'r', encoding = "ISO-8859-1")
        test_file_nonspam = open(nonspam_addr[i], 'r', encoding = "ISO-8859-1")
        for lines in test_file_spam:
            split = lines.split()
            for word in split:
                split_spam.append(word)
            
        for lines in test_file_nonspam:
            split_2 = lines.split()
            for word in split_2:
                split_non.append(word)

    #print(split_spam)
    #print(split_non)

    split_spam_len = len(split_spam)
    split_non_len = len(split_non)
    print(split_spam_len)
    #print(split_non_len)

    #min_len = min(split_spam_len, split_non_len)
    #print(min_len)
    for m in range(0,300):
        print(m)
      
  
        try:
            
            if split_non[m] not in split_spam:
                #print("\n")
                #print("split_non")
                #print(split_non[m])
                split_non.remove(split_non[m])
                
            if split_spam[m] not in split_non:
                #print("\n")
                #print("split_spam")
                #print(split_spam[m])
                split_non.append(split_spam[m])
                print("ADDED TO THE LIST")
                split_spam.remove(split_spam[m])
            print("\n")
            #print("LENGTH OF THE LISTS")
            #print(len(split_spam))
            #print(len(split_non))

            split_spam_join = " ".join(split_spam)
            split_non_join = " ".join(split_non)

            #print(split_spam_join)
            #print(split_non_join)

            
            with open ( spam_addr[i], 'w', encoding='utf-8') as f:
                f.write(split_spam_join)
            f.close()
            #print("completed writing spam")

            with open (nonspam_addr[i], 'w', encoding = 'utf-8') as f:
                f.write(split_non_join)
            f.close()
            #print("completed writing non - spam")
            
            
            all_words_test = []
            r = []
            
            for root, dirs, files in os.walk(test_dir):
                for name in files:
                    r.append(os.path.join(root, name))

            #print(len(all_words))
            #print(len(direc))
            
            #Extract Features
            #print(len(direc))
            test_features_matrix = np.zeros((len(r),dict_len))
            #print(features_matrix)
             
            docID = 0;
            for s in r:
                #print(s)
                test_file = open(s, 'r', encoding = "utf")
                for i,line in enumerate(test_file):
                    words = line.split()
                    #print(words)
                    
                    for word in words:
                        #print(word)
                        wordID = 0
                        
                        for t,d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = t
                                
                                test_features_matrix[docID,wordID] = words.count(word)
                docID = docID + 1
        
            test_matrix = test_features_matrix
            
            #print(test_matrix)
            test_labels = np.zeros(len(r))
            #print(test_labels)
            #print(len(direc))
            
            for p in r:
                split = p.split("/")[-1]
                #print(split)
                if split.startswith("spm"):
                    #print(r.index(i))
                    index = r.index(p)
                    #print(index)
                    test_labels[index] = 1
            #print(test_labels)
            
            result1 = classifier_NB.predict(test_matrix)
            print(result1)
            print("*************************************")
            #sleep(2)
            
            

        except Exception as e:
            print(e)
            continue
           



