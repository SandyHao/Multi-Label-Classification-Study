from CustomizeClassifierChain import customizeClassifierChain
from CustomizeNaiveBayes import customizenaivebayes
import numpy as np 
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import re # for splitting
from nltk.stem.snowball import SnowballStemmer # for stemming

# @param sentence input string to be converted
# @output string with stem words
# @notice stem means find root for the word. say amusement amused have the same root amus -> ensure
#         similiar words with same sentiment be categorized under same category
# @link https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
def stemming(sentence):
    stemmer = SnowballStemmer("english")
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

# Add stop words given from Assignment 5
csvPath = './somesentences.csv'
stopPath = './stoplist.txt'
stopList = []
with open(stopPath, 'r', encoding='utf-8') as f:
    for stopWord in f.readlines():
        stopList.append(stopWord[:-1]) # -1 indexing for removing the \n at the end of the word


x = [] 
y = []
ID = []
# PreProcessing on the data
with open(csvPath, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')

    beginInd = 0
    for row in reader:
        # every row is in [index, ID, content, class1, class2] shape
        if beginInd != 0:
            # remove punctuations, see https://cloud.tencent.com/developer/ask/27468
            # and convert all characters to lower case
            row[2] = re.sub(r'[^\w\s]', '', row[2]).lower()
            # stemming
            row[2] = stemming(row[2])
            # Remove the stop words in the content
            curX = ''.join(w+' ' for w in row[2].split() if w not in stopList)
            x.append(curX)

            # ID
            ID.append(row[1])
            # Convert to onehot encoding 
            firstClass = int(row[3])
            secondClass = int(row[4])

            oneHotY = [0 for i in range(firstClass -1)]
            oneHotY.append(1)
            for i in range(firstClass + 1,10):
                oneHotY.append(0)
            if row[4] != '99':
                oneHotY[secondClass-1] = 1  
            y.append(oneHotY)

        beginInd += 1

# TF-IDF: frequency in the doc * uniqueness of the word among all docs
vectorizer = CountVectorizer(input='content', encoding='utf-8', lowercase=True, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", \
    ngram_range=(1, 3), analyzer='word', max_features=600, vocabulary=None, binary=False, dtype=np.int64)
vectorizer.fit(x)


y = np.asarray(y)
# Splitting into train and test set
# Keep IDs tho. What if results make sense somehow :)
x_train, x_test, y_train, y_test, ID_train, ID_test = train_test_split(x, y, ID, test_size=0.2, random_state=0)

x_train = vectorizer.transform(x_train).toarray()
x_test = vectorizer.transform(x_test).toarray()
print(len(x_train))
print(len(x_train[0]))
print(len(y_train))
print(len(y_train[0]))
#print(len(np.array(x_test)))
#print(len(x_test[0]))
# BernoulliNB
bnb=customizenaivebayes()
alpha=1
bnb.fit(x_train,y_train)
y_predbnb=bnb.predict(x_test)
print('BernoulliNB acc = ', accuracy_score(np.array(y_test), y_predbnb))
