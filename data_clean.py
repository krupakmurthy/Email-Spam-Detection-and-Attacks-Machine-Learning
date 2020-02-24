import sys
import json
import os
import re
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
  

if __name__ == "__main__":
    tweets_folder_path = sys.argv[1]+'/'
    r=[]
    tid_set = set()
    
    for root, dirs, files in os.walk(tweets_folder_path):
        for name in files:
            r.append(os.path.join(root, name))

    for i in r:
        print(i)
        test_file = open(i, 'r', encoding = "utf")
        for lines in test_file:
            lines_tokenised = []

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

            #stemming
            ps = PorterStemmer()
            words = word_tokenize(lines_noDollar)
            #print(words)

            words=[word.lower() for word in words if word.isalpha()]

            words = [word for word in words if word not in stopwords.words('english')]

            for word in words:
                #print(word)
                w = ps.stem(word)
                #print(w)
                #print("\n")
                lines_tokenised.append(str(w))

            lines_tokenised = TreebankWordDetokenizer().detokenize(lines_tokenised)
        #print(lines_tokenised)
        #print("\n")
        name = i.split("/")[-1]
        path = "C:/Users/krupa krishnamurthy/NB/NB_MT/dataset/bare/cleaned_data/"
        pn = path + name
        print(pn)
        with open (pn, mode = 'a') as file1:
            file1.write(lines_tokenised)
            file1.write("\n")
        file1.close()
        test_file.close()






        

       
