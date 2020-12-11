import re
import string
import nltk
from sklearn import metrics
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from nltk.stem import PorterStemmer 
from nltk import FreqDist 
from nltk.tokenize import RegexpTokenizer
import numpy as np
import sys
sys.path.append('../pyTsetlinMachineParallel/')
from tm import MultiClassTsetlinMachine
nltk.download('wordnet')
from time import time 
stop_words = set(stopwords.words('english'))
tokenizerR = RegexpTokenizer(r'\w+')
from numpy import save


#%%%%%%%%%%%%%Read the Sentiment Score Binary File%%%%%%%%%%%%%%%%%

dfSenti = pd.read_csv('restSCBinary.csv', header = None)    #For Laptop dataset change this to 'laptopSCBinary.csv'
restSC = dfSenti.iloc[:,:].values



alpha = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

#%%%%%%%%%%%%% Read the Restaurant Splitted File %%%%%%%%%%%%%%%%%
df = pd.read_csv('restSplit.csv')    #For Laptop dataset change this to 'laptopSplit.csv'
target = df.iloc[:,0:1].values
text = df.iloc[:,1:2].values
textSplit1 =df.iloc[:,2:3].values
textSplit2 = df.iloc[:,3:4].values
label = df.iloc[:,4:5].values
y = np.reshape(label, len(label))


#%%%%%%%%%%%%% Read the Restaurant Original File to create the Vocabulary %%%%%%%%%%%%%%%%%
dfOriginal = pd.read_csv('rest.csv')    #For Laptop dataset change this to 'laptop.csv'
textOrig = dfOriginal.iloc[:,1:2].values


#%%%%%%%%%%%%% Read the Opinion Lexicon File %%%%%%%%%%%%%%%%%
dflex = pd.read_csv('lexicon.csv')
posLex = dflex.iloc[:,1:2].values
negLex = dflex.iloc[:,0:1].values


#Data Preprocessing
def prepreocess(data):
    input_data=[]
    vocab   = []
    for i in data:
        for j in i:
            j = j.lower()
            j = j.replace("\n", "")
            j = j.replace('n\'t', 'not')
            j = j.replace('\'ve', 'have')
            j = j.replace('\'ll', 'will')
            j = j.replace('\'re', 'are')
            j = j.replace('\'m', 'am')
            j = j.replace('/', ' / ')
            j = j.replace('-', ' ')
            j = j.replace('!', ' ')
            j = j.replace('?', ' ')
            j = j.replace('+', ' ')
            j = j.replace('*', ' ')
            while "  " in j:
                j = j.replace('  ', ' ')
            while ",," in j:
                j = j.replace(',,', ',')
            j = j.strip()
            j = j.strip('.')
            j = j.strip()

            temp1 = tokenizerR.tokenize(j)  #tokenize and removes the punctuations and symbols

            temp2 = [x for x in temp1 if not x.isdigit()]  #removes the digits
            
            temp3 = [w for w in temp2 if not w in alpha]	# removes the single alphabets

            input_data.append(temp3)
 
    return input_data

input_text = prepreocess(textOrig)
input_text1 = prepreocess(textSplit1)     
input_text2 = prepreocess(textSplit2)
input_target = prepreocess(target)



# Replace the words in the datasets with a common tag ''positive'' and ''negative''.
def lexicon_based(data1):
    for i in data1:
        for j in range(len(i)):
            if i[j] in posLex[:]:
                i[j] = 'positive'
            if i[j] in negLex[:]:
                i[j] = 'negative'
    return data1

lexicon_input = lexicon_based(input_text)
lexicon_input1 = lexicon_based(input_text1)
lexicon_input2 = lexicon_based(input_text2)
lexicon_target = lexicon_based(input_target)

# Use PortStemmet to stem the words in the sentence
inputtext = []
for i in lexicon_input:
    ps = PorterStemmer()
    temp4 = []
    for m in i:
        temp_temp =ps.stem(m)
        temp4.append(temp_temp)
    inputtext.append(temp4)
	
# Use PortStemmet to stem the words in the target 	
inputtarget=[]
for i in lexicon_target:
    ps = PorterStemmer()
    temp5 = []
    for l in i:
        temp_temp =ps.stem(l)
        temp5.append(temp_temp)
    inputtarget.append(temp5)


#Create a vocabulary list
newVocab =[]
for i in inputtext:
    for j in i:
            newVocab.append(j)
            
for i in inputtarget:
    for j in i:
            newVocab.append(j)

print(len(set(newVocab)))

#Use FreqDist to get most common 2500 words
fdist1 = FreqDist(newVocab)
tokens1 = fdist1.most_common(2500)

#Save the common words in a list
full_token_fil = []
for i in tokens1:
    full_token_fil.append(i[0])

vocab_unique = full_token_fil
print(len(vocab_unique))


#Create a 3 bit binary input for the first part of the split sentence representing the position of positive, negative and no sentiment tokens.
def binarization_additionalInfo1(data4):
    feature_set = np.zeros([4728, 3], dtype=np.uint8)
    tnum=0
    for t in data4:
        if 'positive' in t:
            feature_set[tnum][0] = 1
        if 'negative' in t:
            feature_set[tnum][2] = 1
        else:
            feature_set[tnum][1] = 1
        tnum += 1
    return feature_set


#Create a 3 bit binary input for the second part of the split sentence representing the position of positive, negative and no sentiment tokens.
def binarization_additionalInfo2(data4):
    feature_set = np.zeros([4728, 3], dtype=np.uint8)
    tnum=0
    for t in data4:
        if 'positive' in t:
            feature_set[tnum][0] = 1
        if 'negative' in t:
            feature_set[tnum][2] = 1
        else:
            feature_set[tnum][1] = 1
        tnum += 1
    return feature_set


#Create a Bag of Words (BOW) for the input sentence and the target word.
def binarization_text(data4):
    feature_set = np.zeros([4728, len(vocab_unique)], dtype=np.uint8)
    tnum=0
    for t in data4:
        for w in t:
            if (w in vocab_unique):
                idx = vocab_unique.index(w)
                feature_set[tnum][idx] = 1
        tnum += 1
    return feature_set


Loc_vec1 = binarization_additionalInfo1(lexicon_input1)	#get 3 bit binary representation of first part of sentence (LOC_vec^1)
Loc_vec2 = binarization_additionalInfo2(lexicon_input2)	#get 3 bit binary representation of second part of sentence (LOC_vec^2)

X_text = binarization_text(inputtext)
X_target = binarization_text(inputtarget)

#Concatenate all the input to form final input.
X_final1 = np.concatenate((X_text, X_target), axis = 1)
X_final2 = np.concatenate((Loc_vec1, Loc_vec2), axis = 1)

X_final3 = np.concatenate((X_final1, X_final2), axis = 1)
X_final4 = np.concatenate((X_final3, restSC), axis = 1)

#Split training and testing samples
X_train = X_final4[0:3608,:]  #For Laptop dataset change this from 3608 to 2328
X_test  = X_final4[3608:,:]	#For Laptop dataset change this from 3608 to 2328
ytrain = y[0:3608]	#For Laptop dataset change this from 3608 to 2328
ytest = y[3608:]	#For Laptop dataset change this from 3608 to 2328


#%%%%%%%%%%%%%%%%%% Initialize Tsetlin Machine %%%%%%%%%%%%%%%%%%%%%%%
tm1 =  MultiClassTsetlinMachine(700, 90*100,15, weighted_clauses=True)  #number of clause= 700, T = 90*100 and s = 15
#tm1.fit(X_train, ytrain, epochs=0)
print("\nTraining Classification Layer...")

print("\nAccuracy over 1000 epochs:\n")
max = 0
for i in range(500):
	start_training = time()
	tm1.fit(X_train, ytrain, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result2 = 100*(tm1.predict(X_train) == ytrain).mean()
	result1 = 100*(tm1.predict(X_test) == ytest).mean()
	y_pred = tm1.predict(X_test)
	f1= metrics.f1_score(ytest, y_pred, average='macro')
	if result1>max:
		max = result1
		pred = tm1.predict(X_test)
		ta_state = tm1.get_state()
	stop_testing = time()
	print("#%d AccuracyTrain: %.2f%% AccuracyTest: %.2f%% F1-Score: %.2f%%  Training: %.2fs Testing: %.2fs" % (i+1, result2, result1, f1*100, stop_training-start_training, stop_testing-start_testing))




#%%%%%%%%%%%%%%%%% To extrat the clause with its literal of particular class %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tm1.set_state(ta_state)

#Save trained model with highest testing accuracy
np.savez_compressed("laptopModel.npz", ta_state)

#Load trained TM to evaulate results
ta_state = np.load("laptopModel.npz")['arr_0']

tm1.set_state(ta_state)
print(tm1.predict(X_final4[1990:1991,:]))
number_of_features = 4212
print("\nClass 2 Positive Clauses:\n")
for j in range(0, 300, 2):					#0 is negative, 1 is neutral and 2 is positive class (change accordingly)
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm1.ta_action(1, j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				continue
	print(" ∧ ".join(l))
