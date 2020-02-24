import sys
import json
import os
import re
from collections import Counter
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
import nltk
from nltk.tag import pos_tag
#nltk.download('punkt')
#nltk.download('stopwords')


if __name__ == "__main__":
    folder_path = 'reuters/'
    r=[]
    print("3000")    
    all_words = []
    
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            r.append(os.path.join(root, name))

    for i in r:
        print(i)
        test_file = open(i, 'r', encoding = "ISO-8859-1")
        for lines in test_file:

            #lowercase the lines
            #print(lines)
            lines_lowercase = lines.lower()
            #print(lines_lowercase)

            #remove HTML tags and  HTML texts can also contain entities, that are not enclosed in brackets such as '&nsbm'.
            lines_noHTML = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', lines_lowercase)
            #print(lines_noHTML)

            #remove URLs
            lines_noURL = re.sub(r'^https?:\/\/.*[\r\n]*', 'httpaddr', lines_noHTML, flags=re.MULTILINE)
            #print(lines_noURL)

            #remove emails
            lines_noEmail = re.sub('\S*@\S*\s?', 'emailaddr', lines_noURL)
            #print(lines_noEmail)

            #replace digits with text "numbers"
            lines_noNumbers = re.sub('[0-9]+', 'number', lines_noEmail)
            #print(lines_noNumbers)

            #replace $ with 'dollar'
            lines_noDollar = re.sub('$', '', lines_noNumbers)
            #print(lines_noNumbers)
            #print(lines)
            tagged_sentence = nltk.tag.pos_tag(lines_noDollar.split())
            edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
            lines_tagged = ' '.join(edited_sentence)
            
            ps = PorterStemmer()
            words = word_tokenize(lines_tagged)

            words= [word.lower() for word in words if word.isalpha()]
            
            words = [word for word in words if word not in stopwords.words('english')]

            words = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']

            words = lines_noDollar.split()
            for word in words:
                #nltk.tag.pos_tag
                all_words.append(word)

    


    #print(len(all_words))
    #print(all_words)
    #Create Dictionary
    dictionary = Counter(all_words)
        
    print(len(dictionary))
    #print("\n")
    dictionary = dictionary.most_common(3000)
    with open ("dictionary.txt" , mode = "a") as writefile:
        for line in dictionary:
            line = str(line)
            writefile.write(line)
            writefile.write("\n")
    writefile.close()

    #print(dictionary)
    dict_len = len(dictionary)
    #print(dict_len)

    

