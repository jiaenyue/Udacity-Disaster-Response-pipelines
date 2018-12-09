# import libraries
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect

 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support,accuracy_score,label_ranking_average_precision_score
from sklearn.model_selection  import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

from tranformer import TextLengthExtractor

import lightgbm as lgb

import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath):
    """ Load data from sqlite database then covert to X(traing data) and  Y(target labels)

    Args:
    database_filepath: file path of the database

    Returns:
    seriers: X, messages
    seriers: Y, labels
    category_names: label names   
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    inspector = inspect(engine)

    # Get table information
    print("Tables name: ", inspector.get_table_names())
    df = pd.read_sql("SELECT * FROM disaster_response", engine)

    X = df['message']

    Y_names = list(set(df.columns.values) - set( ['id','message', 'original', 'genre']))
    Y = df[Y_names]

    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    #init stopwords and WordNetLemmatizer
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
     
    # tokenize text
    tokens = word_tokenize(text)
     
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
 
    return tokens


def build_model():
    """
        Builds a classification pipeline use random grid search

        Args: none.


        Returns: A random grid sklearn model
    """

    pipeline = Pipeline([
    ('features', FeatureUnion([       
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
        ('text_total_length', TextLengthExtractor()) # Add a new feature
        ])),
    ('clf', MultiOutputClassifier(lgb.LGBMClassifier(num_leaves=70))) # Change the estimator to LGBMClassifier
    ])

    parameters = {
            'features__tfidfvect__ngram_range': [(1, 1), (1, 2)],
            'features__tfidfvect__max_df': [0.75, 1.0],
            'features__tfidfvect__max_features': [2000, 5000],
            'features__transformer_weights': (
                {'tfidfvect': 1, 'text_total_length': 0.5},
                {'tfidfvect': 1, 'text_total_length': 1}
            ),
            'clf__estimator__boosting_type': ['gbdt', 'dart'],
            'clf__estimator__n_estimators': [10, 15, 25],
            'clf__estimator__learning_rate': [0.5, 0.1]
            }

    rscv = RandomizedSearchCV(pipeline, param_distributions=parameters, cv=5)
    
    return rscv

def evaluate_model(model, X_test, y_test, category_names):
    """ Evaluate model on test set,
        Predict results for each category.
        
    Args:
        model: trained model
        X_test: pandas.DataFrame for predict 
        y_test: pandas.DataFrame for labeled test set
        category_names: list for category names
        `
    Returns: 
        none.
    
    """
    
    # predict test df
    Y_pred = model.predict(X_test)
    tot_acc = 0
    tot_f1 = 0
    # print report 
    for i, cat in enumerate(category_names):    
        metrics =  classification_report(y_test[y_test.columns[i]], Y_pred[:,i])
        tot_acc += accuracy_score(y_test[y_test.columns[i]], Y_pred[:,i])
        tot_f1 += precision_recall_fscore_support(y_test[y_test.columns[i]], Y_pred[:,i], average = 'weighted')[2]
        print(cat, 'accuracy: {:.5f}'.format(accuracy_score(y_test[y_test.columns[i]], Y_pred[:,i])))
        print(metrics)
    print('total accuracy {:.5f}'.format(tot_acc/len(category_names)))
    print('total f1 {:.5f}'.format(tot_f1/len(category_names)))


def save_model(model, model_filepath):
    """ Persisit mode to disk use 

        Args: 
            model: a sklearn model.
            model_filepath: the path for save model


        Returns: none.
    """
    joblib.dump(model, model_filepath, compress = 1)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()