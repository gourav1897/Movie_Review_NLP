#########################################################################################
####  File: run_sklearn_model_performance.py
####  Desc: file for sci-kit learn library
####  Author: Rohit Sharma, Achal Velani
#########################################################################################

import sys
import pandas
import numpy
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

def process(filepath):
  # number of folds for cross-validation
  kFolds = 10

  # read CSV file with panda library package using filepath passed in the function
  trainSet = pandas.read_csv(filepath)
  
  # this is a data frame for the data
  print ('Shape of feature data - num instances with num features + class label')
  print (trainSet.shape)
  
  # Convert data to NUMpy array for sklearn package
  trainArray = trainSet.values
  
  # get the last column with the class labels into a vector y
  trainY = trainArray[:,-1]
  
  # get the remaining rows and columns into the feature matrix X
  trainX = trainArray[:,:-1]

  #  ** choose one of these classifiers **
  #print '** Results from Linear SVM'
  # now call sklearn with SVC to get a model
  #classifier = LinearSVC(C=1, penalty='l1', dual=False, class_weight='auto')
  
  print ('** Results from Logistic Regression with liblinear')
  ## solver options:  solver : {'newton-cg', 'lbfgs', 'liblinear'}
  ## multi-class options: multi_class : str, {'ovr', 'multinomial'} but multinomial only for lbfgs
  classifier = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='multinomial')
  
  yPred = cross_val_predict(classifier, trainX, trainY, cv=kFolds)
  
  # classification report compares predictions from the k fold test sets with the gold
  print(classification_report(trainY, yPred))
  
  # confusion matrix from same
  cm = confusion_matrix(trainY, yPred)
  print('\n')
  print(pandas.crosstab(trainY, yPred, rownames=['ActualValue'], colnames=['PredictedValue'], margins=True))
      
# use a main so can get feature file as a command line argument
if __name__ == '__main__':
    # Make a list of command line arguments, omitting the [0] element
    # which is the script itself.
    args = sys.argv[1:]
    if not args:
        print ('usage: python run_sklearn_model_performance.py [featurefile]')
        sys.exit(1)
    trainfile = args[0]
    process(trainfile)
