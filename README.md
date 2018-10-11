# instal the required packages
    import os
    import pandas as pd
    import numpy as np
    from IPython.display import display

    from bs4 import BeautifulSoup
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cross_validation import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb


# instal the training set
     train = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# Converting into dataframe and View 
     print('Dimension of Labeled Training Data: {}.'.format(train.shape))
     print('There are {0} samples and {1} variables in the training data.'.format(train.shape[0], train.shape[1]))
     display(train.head())
     print(train.review[0])

## Clean the training set
   # 1. Remove HTML
        reviews_text = list(map(lambda x: BeautifulSoup(x, 'html.parser').get_text(), reviews))
   # 2. Remove non-letters
        reviews_text = list(map(lambda x: re.sub("[^a-zA-Z]"," ", x), reviews_text))
   # 3. Convert words to lower case and split them
        words = list(map(lambda x: x.lower().split(), reviews_text))
   # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
        set_of_stopwords = set(stopwords.words("english"))
        meaningful_words = list(map(lambda x: [w for w in x if not w in set_of_stopwords], words)
   # 5. Optionally stem the words
        if stem:
        porter_stemmer = PorterStemmer()
        wordnet_lemmatizer = WordNetLemmatizer()
        stemmed_words = list(map(lambda x: [porter_stemmer.stem(w) for w in x], meaningful_words))
        stemmed_words = list(map(lambda x:[wordnet_lemmatizer.lemmatize(w) for w in x], stemmed_words))
   # 6. Join the words to a single string
        clean_review = map(lambda x: ' '.join(x), stemmed_words)
        else:
        clean_review = list(map(lambda x: ' '.join(x), meaningful_words))
    
        return clean_review
   
### OR Splitting the words into Vector in stead of 4 ### 
        train['review_words'] = train['review_letters_only'].apply(lambda x: x.lower().split())

        set_of_stopwords = set(stopwords.words("english"))
        train['review_meaningful_words'] = train['review_words'].apply(lambda x: [w for w in x if not w in set_of_stopwords])
    
#### 
        num_removed = len(train['review_words'][0]) - len(train['review_meaningful_words'][0])  
        print('For the first review entry, the number of stop words removed is {0}.'.format(num_removed))    
    
        # train['review_cleaned'] = train['review_stemmed'].apply(lambda x: ' '.join(x)) # uncomment if using stemming# train[ 
        train['review_cleaned'] = train['review_meaningful_words'].apply(lambda x: ' '.join(x)) # comment if using stemming

####Converting into data frame from cleaned text###
        train.drop(['review', 'review_bs', 'review_letters_only', 'review_words', 'review_meaningful_words'],axis=1, inplace=True)
        display(train.head())

        print(train['review_cleaned'][0])


        vectorizer  = CountVectorizer(analyzer="word", preprocessor=None, tokenizer=None, stop_words=None, max_features=5000)
        train_data_features = vectorizer.fit_transform(list(train['review_cleaned'].values))

# Numpy arrays are easy to work with, so convert the result to an array
        train_data_features = train_data_features.toarray()

##View the first element in the training data
        train_data_features[0]

        print('The dimension of train_data_features is {}.'.format(train_data_features.shape))


        vocab = vectorizer.get_feature_names()
        print(vocab)


## Read the test data
        test = pd.read_csv("../input/testData.tsv", header=0, delimiter='\t', quoting=3)

# Verify that there are 25,000 rows and 2 columns
        print('The dimension of test data is {}.'.format(test.shape))

# Get a bag of words for the test set, and convert to a numpy array
        clean_test_reviews = clean_reviews(list(test['review'].values), remove_stopwords=True)
        test_data_features = vectorizer.transform(clean_test_reviews)
        test_data_features = test_data_features.toarray()


## Random Forest
# Initialize a Random Forest classifier with 100 trees
        rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0) 

# Use cross validation to evaluate the performance of Random Forest
        rf_clf_error = 1 - cross_val_score(rf_clf, train_data_features, train['sentiment'],cv=5, scoring='accuracy', n_jobs=-1).mean()
        print('Random Forest training error: {:.4}'.format(rf_clf_error))


## XG Boost
 # Create xgb trianing set and parameters
       dtrain = xgb.DMatrix(train_data_features, label=train['sentiment'])
       params = {'silent': 1, 'nthread': -1, 'eval_metric': 'error'}

 # Use cross validation to evaluate the performance of XGBoost
       print('The cross validation may take a while...')
       xgb_cv_results = xgb.cv(params, dtrain, num_boost_round=100, nfold=5, show_stdv=False, seed=0)
       xgb_error = xgb_cv_results['test-error-mean'].mean()
       print('XGBoost trianing error: {:.4}'.format(xgb_error))

# Fit the forest to the training set, using the bag of words as features and the sentiment labels as labels
 # This may take a few minutes to run
       rf_clf.fit(train_data_features, train['sentiment'])

 # Use the random forest to make sentiment label predictions
       result = rf_clf.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column an a "sentiment" column
       output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

# Use pandas to write the comma-separated output file
       output.to_csv("Bag_of_Words_sutithi.csv", index=False, quoting=3)   
       print("Wrote results to Bag_of_Words_sutithi.csv")
