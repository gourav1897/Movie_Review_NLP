#########################################################################################
####  File: classifyKaggle.py
####  Desc: Python processing for classifying movie review data set
####  Source: Nancy McCracken
####  Author: Rohit Sharma, Achal Velani
#########################################################################################
'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import *
import sentiment_read_subjectivity
import sentiment_read_LIWC_pos_neg_words
from nltk.metrics import ConfusionMatrix
from nltk.metrics import scores

# possibly filter tokens
stopwords = nltk.corpus.stopwords.words('english')
newstopwords = [word for word in stopwords if word not in ['do','does','did','doing','t','can','don','aren','couldn','didn','doesn','hadn','hasn','haven' ,'isn','mightn','needn','shouldn','shan','wasn','weren','won','wouldn','should','have','has','had','having']]

#### switchcase map to be used in writeFeatureSets function for getting featureline ####
swithcase = {
  0:  lambda  featureline: featureline + str("neg"),
  1:  lambda  featureline: featureline + str("sneg"),
  2:  lambda  featureline: featureline + str("neu"),
  3:  lambda  featureline: featureline + str("spos"),
  4:  lambda  featureline: featureline + str("pos"),
}

#########################################################################################
####  Function writeFeatureSets ####
#########################################################################################
# Function writeFeatureSets:
#    takes featuresets defined in nltk and convert them to csv file for input in weka
#    any feature value in the featuresets should not contain ",", "'" or " itself
#    and write the file to the outpath location. outpath should include the name of the csv file

def writeFeatureSets(featuresets, outpath):
    # open outpath for writing
    f = open(outpath, 'w')
    # get the feature names from the feature dictionary in the first featureset
    featurenames = featuresets[0][0].keys()
    # create the first line of the file as comma separated feature names
    #    with the word class as the last feature name
    featurenameline = ''
    for featurename in featurenames:
        # replace forbidden characters with text abbreviations
        featurename = featurename.replace(',','CM')
        featurename = featurename.replace("'","DQ")
        featurename = featurename.replace('"','QU')
        featurenameline = featurenameline + featurename + ','
    featurenameline = featurenameline + 'class'
    f.write(featurenameline)
    f.write('\n')
    # convert each feature set to a line in the file with comma separated feature values,
    # each feature value is converted to a string, for booleans this is the words true and false
    #   for numbers, this is the string with the number
    for featureset in featuresets:
        featureline = ''
        for key in featurenames:
            featureline += str(featureset[0][key]) + ','
        featureline = swithcase[featureset[1]] (featureline)
        f.write(featureline)
        f.write('\n')
    f.close()

#### Read subjectivity from subjectivity py file ####
# this function returns a dictionary where you can look up words and get back 
# the four items of subjectivity information -- strength, posTag, isStemmed, polarity
SL = sentiment_read_subjectivity.readSubjectivity('SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')

#########################################################################################
####  Function slFeatures ####
#########################################################################################
# Function slFeatures:
#    returns feature set based on subjectivity.
def slFeatures(document, wordFeatures, SL):
    document_words = set(document)
    features = {}
    for word in wordFeatures:
        features['contains({})'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = strongPos = weakNeg = strongNeg  = weakNeu = strongNeu = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            if strength == 'weaksubj' and polarity == 'neutral':
                weakNeu += 1
            if strength == 'strongsubj' and polarity == 'neutral':
                strongNeu += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
            features['neutralcount'] = weakNeu + (2 * strongNeu)
    if 'positivecount' not in features:
      features['positivecount']=0
    if 'negativecount' not in features:
      features['negativecount']=0     
    if 'neutralcount' not in features:
      features['neutralcount']=0   
    return features

#########################################################################################
####  Function liwcFeatures ####
#########################################################################################
# Function liwcFeatures:
#    returns feature set based on LIWC postive and negative emotion list.
def liwcFeatures(document, wordFeatures):
  (poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words() #read emotion list
  document_words = set(document)
  features = {}
  for word in wordFeatures:
    features['contains({})'.format(word)] = (word in document_words)
  posCount=0
  negCount=0
  for word in document_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(word, poslist): #checks if word present in pos list
      posCount+=1
    if sentiment_read_LIWC_pos_neg_words.isPresent(word, neglist): #checks if word present in neg list
      negCount+=1
    features['positivecount'] = posCount
    features['negativecount'] = negCount
  if 'positivecount' not in features:
    features['positivecount']=0
  if 'negativecount' not in features:
    features['negativecount']=0     
  return features

#########################################################################################
####  Function posFeatures ####
#########################################################################################
# Function posFeatures:
#    returns feature set based on POS tagging done using stanford pos tagger.
def posFeatures(document, wordFeatures):
  documentWords = set(document)
  features = {}
  taggedWords = nltk.pos_tag(document)
  for word in wordFeatures:
    features['contains({})'.format(word)] = (word in documentWords)
  countNoun = countVerb = countAdj = countAdv = countModal = countSym = countDet = countInterj = 0
  for (word, tag) in taggedWords:
    if tag.startswith('N'): countNoun+=1
    if tag.startswith('V'): countVerb+=1
    if tag.startswith('J'): countAdj+=1
    if tag.startswith('R'): countAdv+=1
    if tag.startswith('M'): countModal+=1
    if tag.startswith('S'): countSym+=1
    if tag.startswith('D'): countDet+=1
    if tag.startswith('U'): countInterj+=1
  features['nouns'] = countNoun
  features['verbs'] = countVerb
  features['adjectives'] = countAdj
  features['adverbs'] = countAdv
  features['modals'] = countModal
  features['symbols'] = countSym
  features['determiners'] = countDet #useful for unprocessed data
  features['interjections'] = countInterj
  return features

#########################################################################################
####  Function notFeatures ####
#########################################################################################
# Function notFeatures:
#    returns not feature set.
def notFeatures(document, wordFeatures, negationWords):
    features = {}
    for word in wordFeatures:
        features['contains({})'.format(word)] = False
        features['contains(NOT{})'.format(word)] = False
    # loop through each word in the document in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationWords) or (word.endswith("n't"))):
            i += 1
            features['contains(NOT{})'.format(document[i])] = (document[i] in wordFeatures)
        else:
            features['contains({})'.format(word)] = (word in wordFeatures)
    return features

#########################################################################################
####  Function getBigramFeatures ####
#########################################################################################
# Function getBigramFeatures:
#    returns feature set of 500 best bigrams based on chi_sq
def getBigramFeatures(allWordsList):
  bigram_measures = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(allWordsList)
  finder.apply_freq_filter(6)
  bigram_features = finder.nbest(bigram_measures.chi_sq, 500)
  return bigram_features

#########################################################################################
####  Function getBigramDocumentFeatures ####
#########################################################################################
# Function getBigramDocumentFeatures:
#    returns feature set of bigrams for document
def getBigramDocumentFeatures(document, word_features, bigram_features):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    for bigram in bigram_features:
        features['bigram({} {})'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
    return features

#########################################################################################
####  Function getTrigramFeatures ####
#########################################################################################
# Function getTrigramFeatures:
#    returns feature set of 500 best trigrams based on chi_sq
def getTrigramFeatures(allWordsList):
  trigram_measures = nltk.collocations.TrigramAssocMeasures()
  finder = TrigramCollocationFinder.from_words(allWordsList)
  finder.apply_freq_filter(6)
  trigram_features = finder.nbest(trigram_measures.chi_sq, 500)
  return trigram_features

#########################################################################################
####  Function getTrigramDocumentFeatures ####
#########################################################################################
# Function getTrigramDocumentFeatures:
#    returns feature set of trigrams for document
def getTrigramDocumentFeatures(document, word_features, trigram_features):
    document_words = set(document)
    document_trigrams = nltk.trigrams(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    for trigram in trigram_features:
        features['trigram({} {} {})'.format(trigram[0], trigram[1], trigram[2])] = (trigram in document_trigrams)    
    return features

#########################################################################################
####  Function calculateAccuracy ####
#########################################################################################
# Function calculateAccuracy:
# Function to  calculate and print accuracy using Naive Bayes classifier -- train and test data in one set
def calculateAccuracy(featuresets):
  #alloting 90% of featuresets to the train set and 10% to the test set
  divisionSize = int(0.9*len(featuresets))
  trainSet = featuresets[:divisionSize]
  testSet = featuresets[divisionSize:]
  classifier = nltk.NaiveBayesClassifier.train(trainSet)
  print ("Classifier Accuracy - ")
  print (nltk.classify.accuracy(classifier, testSet))
  print ("---------------------------------------------------")
  print (classifier.show_most_informative_features(30))
  printConfusionMatrix(classifier, testSet)
  print ("") 

#########################################################################################
####  Function calculateAccuracyForTest ####
#########################################################################################
# Function calculateAccuracyForTest:
# Function to  calculate and print accuracy using Baysian classifier -- for separate train and test set
def calculateAccuracyForTest(trainFeatureSet, testFeatureSet):
  classifier = nltk.NaiveBayesClassifier.train(trainFeatureSet)
  print ("Classifier Accuracy - SEPARATE TEST SET")
  print (nltk.classify.accuracy(classifier, testFeatureSet))
  print ("---------------------------------------------------")
  print (classifier.show_most_informative_features(30))
  printConfusionMatrix(classifier, testFeatureSet)
  print ("")
  
#########################################################################################
####  Function printConfusionMatrix ####
#########################################################################################
# Function printConfusionMatrix:
# Function to print confusion matrix for the testSet
def printConfusionMatrix(classifier, testSet):
  refList = []
  testList = []
  for (features, label) in testSet:
    refList.append(label)
    testList.append(classifier.classify(features))
  print ("---------Confusion Matrix---------")
  confMat = ConfusionMatrix(refList, testList)
  print (confMat)

#########################################################################################
####  Function documentFeatures ####
#########################################################################################
# Function documentFeatures:
# Define features of document for Unigram baseline
# part of the feature set if the word is in document
def documentFeatures(document, wordFeatures):
  document_words = set(document)
  features = {}
  for word in wordFeatures:
    features['contains(%s)' % word] = (word in document_words)
  return features

#########################################################################################
####  Function bagOfWordsFeature ####
#########################################################################################
# Function bagOfWordsFeature:
# function which will return the most common 500 words in the word list after freq dist
def bagOfWordsFeature(wordsInDocs):
  allWords = nltk.FreqDist(w for w in wordsInDocs)
  wordItems = allWords.most_common(500)
  wordFeatures = [word for (word, freq) in wordItems]
  return wordFeatures

#########################################################################################
####  Pre-processing phrases  ####
#########################################################################################
def preProcessingPhrases(phrase):
  # define regex for removing punctuations and numbers from phrase.
  punctuation = re.compile(r'[-.?!/\%@,":;()|0-9]')
  # "create list of lower case words"
  wordList = re.split('\s+', phrase.lower())
  wordList = [punctuation.sub("", word) for word in wordList] 
  modifiedWordList = []
  for word in wordList:
    if word not in newstopwords:
      modifiedWordList.append(word)
  finalPhrase = " ".join(modifiedWordList)
  return finalPhrase 

#########################################################################################
####  Function processkaggle ####
#########################################################################################
# Function processkaggle:
# function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  os.chdir(dirPath)
  f = open('./train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  processedPhraseDocs = []
  #define a tokenizer which will get rid of all the punctuations
  tokenizer = RegexpTokenizer(r'\w+')
  # add all the phrases
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))
    phrase[0] = preProcessingPhrases(phrase[0])
    processedTokens = tokenizer.tokenize(phrase[0])
    processedPhraseDocs.append((processedTokens,int(phrase[1])))  
  
  # get all words and create word features
  wordsInDocs = []
  for i in range(len(phrasedocs)):
    for token in phrasedocs[i][0]:
      wordsInDocs.append(token)

  # get all processed words and create word features    
  processedWordsInDocs = []
  for i in range(len(processedPhraseDocs)):
    for token in processedPhraseDocs[i][0]:
      processedWordsInDocs.append(token)

  # get bag of words features for unprocessed words in document    
  wordFeatures = bagOfWordsFeature(wordsInDocs)
  # get bag of words features for processed words in document
  processedWordFeatures = bagOfWordsFeature(processedWordsInDocs)

  #########################################################################################
  #### Start processing TEST data from separate test.tsv file
  g = open('./test.tsv', 'r')

  phrasedataTEST = []

  for line in g:
    if (not line.startswith('Phrase')):
      line = line.strip()
      phrasedataTEST.append(line.split('\t')[2:3])

  random.shuffle(phrasedataTEST)
  phraselistTEST = phrasedataTEST[:limit]

  phrasedocsTEST = []
  for phrase in phraselistTEST:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocsTEST.append((tokens, int(2)))

  wordsInDocsTEST = []
  for i in range(len(phrasedocsTEST)):
    for token in phrasedocsTEST[i][0]:
      wordsInDocsTEST.append(token)
  
  allWordsTEST = nltk.FreqDist(w for w in wordsInDocsTEST)
  wordItemsTEST = allWordsTEST.most_common(500)
  wordFeaturesTEST = [word for (word, freq) in wordItemsTEST]
  processedFeaturesetsTEST = [(documentFeatures(d, wordFeaturesTEST), c) for (d, c) in phrasedocsTEST]
  #### End processing TEST data from separate test.tsv file -- got processedFeaturesetsTEST --use to get Accuracy for TEST set
  ##################################`#######################################################
  
  ########## Unprocessed feature set - Normal ######################
  unProcessedFeaturesets = [(documentFeatures(d, wordFeatures), c) for (d, c) in phrasedocs]
  #writeFeatureSets(unProcessedFeaturesets,"features_normal.csv")

  ########## Preprocessed feature set - Normal ######################
  preProcessedFeaturesets = [(documentFeatures(d, processedWordFeatures), c) for (d, c) in processedPhraseDocs]
  writeFeatureSets(preProcessedFeaturesets,"features_preprocessed.csv")
  
  ########## Unprocessed SL feature set ######################
  SLFeaturesets = [(slFeatures(d, wordFeatures, SL), c) for (d, c) in phrasedocs]
  #writeFeatureSets(SLFeaturesets,"features_SL_wo_preprocessed.csv")

  ########## Processed SL feature set ######################
  SLFeaturesetsProcessed = [(slFeatures(d, processedWordFeatures, SL), c) for (d, c) in processedPhraseDocs]
  writeFeatureSets(SLFeaturesetsProcessed,"features_SL_preprocessed.csv")
  
  ########## Unprocessed NOT feature set ######################
  negationwords = ['no', 'not', 'none', 'never', 'nothing', 'noone', 'nowhere', 'rather', 'hardly', 'rarely', 'scarcely', 'seldom', 'neither', 'nor']
  NOTFeaturesets = [(notFeatures(d, wordFeatures, negationwords), c) for (d, c) in phrasedocs]
  #writeFeatureSets(NOTFeaturesets,"features_NOT_wo_preprocessed.csv")

  ########## Processed NOT feature set ######################
  NOTFeaturesetsProcessed = [(notFeatures(d, processedWordFeatures, negationwords), c) for (d, c) in processedPhraseDocs]
  #writeFeatureSets(NOTFeaturesetsProcessed,"features_NOT_preprocessed.csv")
  
  ########## Unprocessed BIGRAM feature set ######################
  bigramFeatures= getBigramFeatures(wordsInDocs)
  bigramFeaturesets = [(getBigramDocumentFeatures(d, wordFeatures, bigramFeatures), c) for (d, c) in phrasedocs]
  #writeFeatureSets(bigramFeaturesets,"features_Bigram_wo_preprocessed.csv")  

  ########## Processed BIGRAM feature set ######################
  processedBigramFeatures = getBigramFeatures(processedWordsInDocs)
  processedBigramFeaturesets = [(getBigramDocumentFeatures(d, processedWordFeatures, processedBigramFeatures), c) for (d, c) in processedPhraseDocs]
  writeFeatureSets(processedBigramFeaturesets,"features_Bigram_preprocessed.csv")
  
  ########## Unprocessed TRIGRAM feature set ######################
  trigramFeatures= getTrigramFeatures(wordsInDocs)
  trigramFeaturesets = [(getTrigramDocumentFeatures(d, wordFeatures, trigramFeatures), c) for (d, c) in phrasedocs]
  #writeFeatureSets(trigramFeaturesets,"features_Trigram_wo_preprocessed.csv")  

  ########## Processed TRIGRAM feature set ######################
  processedTrigramFeatures = getTrigramFeatures(processedWordsInDocs)
  processedTrigramFeaturesets = [(getTrigramDocumentFeatures(d, processedWordFeatures, processedTrigramFeatures), c) for (d, c) in processedPhraseDocs]
  writeFeatureSets(processedTrigramFeaturesets,"features_Trigram_preprocessed.csv")
  
  ########## Unprocessed LIWC feature set ######################
  LIWCFeaturesets = [(liwcFeatures(d, wordFeatures), c) for (d, c) in phrasedocs]
  #writeFeatureSets(LIWCFeaturesets,"features_LIWC_wo_preprocessed.csv")

  ########## Processed LIWC feature set ######################
  LIWCFeaturesetsProcessed = [(liwcFeatures(d, processedWordFeatures), c) for (d, c) in processedPhraseDocs]
  writeFeatureSets(LIWCFeaturesetsProcessed,"features_LIWC_preprocessed.csv")

  ########## Unprocessed POS feature set ######################
  POSFeaturesets = [(posFeatures(d, wordFeatures), c) for (d, c) in phrasedocs]
  #writeFeatureSets(POSFeaturesets,"features_POS_wo_preprocessed.csv")

  ########## Processed POS feature set ######################
  POSFeaturesetsProcessed = [(posFeatures(d, processedWordFeatures), c) for (d, c) in processedPhraseDocs]
  writeFeatureSets(POSFeaturesetsProcessed,"features_POS_preprocessed.csv")

  # calculate accuracy for all feature sets - processed and unprocessed
  print("---Unprocessed Featuresets---")
  calculateAccuracy(unProcessedFeaturesets)
  print("---Pre-processed Featuresets---")
  calculateAccuracy(preProcessedFeaturesets)
  print("---SL Featuresets---")
  calculateAccuracy(SLFeaturesets)
  print("---SL Featuresets Processed---")
  calculateAccuracy(SLFeaturesetsProcessed)
  print("---NOT Featuresets---")
  calculateAccuracy(NOTFeaturesets)
  print("---NOT Featuresets Processed---")
  calculateAccuracy(NOTFeaturesetsProcessed)
  print("---Bigram Featuresets---")
  calculateAccuracy(bigramFeaturesets)
  print("---Processed Bigram Features---")
  calculateAccuracy(processedBigramFeaturesets)
  print("---LIWC Featuresets---")
  calculateAccuracy(LIWCFeaturesets)
  print("---LIWC Featuresets Processed---")
  calculateAccuracy(LIWCFeaturesetsProcessed)
  print("---POS Featuresets---")
  calculateAccuracy(POSFeaturesets)
  print("---POS Featuresets Processed---")
  calculateAccuracy(POSFeaturesetsProcessed)
  print("---Trigram Featuresets---")
  calculateAccuracy(trigramFeaturesets)
  print("---Processed Trigram Features---")
  calculateAccuracy(processedTrigramFeaturesets)
  print("---Accuracy for TEST data with preprocessed normal feature set---")
  calculateAccuracyForTest(preProcessedFeaturesets, processedFeaturesetsTEST)

"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])
