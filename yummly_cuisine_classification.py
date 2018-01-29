import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import Counter
import json
import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import re
import pickle
from pandas import Series, DataFrame
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gzip
import shutil
import pprint
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import logging
from sklearn.externals import joblib

logging.getLogger().setLevel(logging.INFO)
# Stemming 
stemmer = WordNetLemmatizer()



def clean_recipe(recipes):
	"""
	This function cleans the input recipes data and return in same format.
	    :type recipes: list
	    :rtype: list
	"""
    # To lowercase
	recipes = [ str.lower(i) for i in recipes ]

	# Remove some special characters and digits
	# Stem ingredients
	return ' '.join([stemmer.lemmatize(re.sub('[^a-z]', ' ', item)) for item in recipes]).strip()


def load_test_data(df):
	"""
	This function load data from pandas DataFrame and clean them and save as a list.
	    :type df: DataFrame
	    :rtype: list
	"""

	# Load data from Pandas Dataframe into List
	x = df['ingredients'].apply(lambda x: clean_recipe(x)).tolist()
	return x


def convertFileIntoJSON(fileName):
	"""
	This function decompress .gz file and save .json file.
	    :type fileName: str
	    :rtype: string
	"""
	if fileName.endswith('.gz'):
	    with gzip.open(fileName, 'rb') as f_in, open(fileName[:-3], 'wb') as f_out:
	        shutil.copyfileobj(f_in, f_out)

	return fileName[:-3]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def predict_unlabeled_data(x_test):
    """Step 0: load trained model and parameters"""
    checkpoint_dir = './trained_model_1517178549/'
    if not checkpoint_dir.endswith('/'):
        checkpoint_dir += '/'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
    logging.info("Training model loaded...")

    """Step 1: Preparing data for prediction"""

    vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_test)))
    logging.info("Vectorization completed...")

    """Step 2: compute the predictions"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            probs = graph.get_operation_by_name("output/probs").outputs[0]
            
            batches = batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
            # all_predictions = []
            all_prob_scores = np.array([])
            for i, x_test_batch in enumerate(batches):
                batch_predictions, batch_scores = sess.run([predictions, probs], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                if i == 0:
                    all_prob_scores = batch_scores
                else: 
                    all_prob_scores = np.concatenate((all_prob_scores, batch_scores), 0)

    return(all_prob_scores)


def generateFinalResultsCsv(tf_test, all_prob_scores):
	"""
    This function finds the top 3 related cuisine results for each recipe.
	    :type tf_test         : DataFrame
	    	  all_prob_scores : list of json
	    :rtype: list of json
    """
	def findTop3Predictions(pred_probas):
		return sorted(zip(pred_probas, labels), reverse=True)[:3]
    
	final_list = []
	for index, row in tf_test.iterrows():
		for each_receipe in findTop3Predictions(all_prob_scores[index]):
			dictn = {}
			dictn['id'] = row['id']
			dictn['cuisine'] = each_receipe[1]
			dictn['confidence'] = round(each_receipe[0], 4)
			final_list.append(dictn)

	return(final_list)


def saveIntoCSV(final_list, classfier_type):
	"""
    This function saves all results into csv format and save file locally.
	    :type final_list     : list of json
	    	  classfier_type : str
	    :rtype: void
    """
	output_filename = "yummly_submission_" + classfier_type + ".csv"
	df_result = pd.DataFrame(final_list, columns=['id', 'cuisine', 'confidence'])         
	df_result[['id' , 'cuisine' , 'confidence' ]].to_csv(output_filename, index=False)
	print("\n\n\tOutput CSV is saved as '{0}'\n\n".format(output_filename))


# Main Function
if __name__ == '__main__':
	test_file = sys.argv[1]
	tf_test = pd.read_json(convertFileIntoJSON(test_file))

	# labels.json was saved during training, and it has to be loaded during prediction
	labels = json.loads(open('./labels.json').read())

	method = input("Please choose classfier from this list:\n\t\t 1. Logistic regression(Acc=0.669)\n\t\t 2. Random forests(Acc=0.613)\n\t\t 3. Text CNN(Acc=0.633)\n\nEnter your choice: ")

	if method == '1':
		if os.path.isfile('yummly_lg.pkl') :
			# Load classifier
			clf = joblib.load('yummly_lg.pkl')
			logging.info("Training model loaded...")

			# Load and clean data...
			x_test = load_test_data(tf_test)
			logging.info('\nTotal number of test inputs: {}'.format(len(x_test)))
			logging.info("Test data loaded...")

			# Load Vectorizer from Pickel file (Created while training...)
			vectorizertr = pickle.load( open( "vectorizer.pk", "rb" ) )
			
			# Vectorized Input...
			test_X=vectorizertr.transform(x_test)
			logging.info("Vectorization completed...")
			
			# Generate Results with class probabilities...
			finalResults = generateFinalResultsCsv(tf_test, clf.predict_proba(test_X))
			logging.info("Preparing for CSV file..")

			# Save All Results to CSV...
			saveIntoCSV(finalResults, "logistic_regression")

		else:
			print("yummly_lg.pkl is not existed in the working directory.")
			print("Please run Jupyter Notebook first.")
			sys.exit()

	elif method == '2':
		if os.path.isfile('yummly_rf.pkl') :
			clf = joblib.load('yummly_rf.pkl')
			logging.info("Training model loaded...")

			# Load and clean data...
			x_test = load_test_data(tf_test)
			logging.info('\nTotal number of test inputs: {}'.format(len(x_test)))
			logging.info("Test data loaded...")

			# Load Vectorizer from Pickel file (Created while training...)
			vectorizertr = pickle.load( open( "vectorizer.pk", "rb" ) )
			
			# Vectorized Input...
			test_X=vectorizertr.transform(x_test)
			logging.info("Vectorization completed...")
			
			# Generate Results with class probabilities...
			finalResults = generateFinalResultsCsv(tf_test, clf.predict_proba(test_X))
			logging.info("Preparing for CSV file..")

			# Save All Results to CSV...
			saveIntoCSV(finalResults, "random_forests")

		else:
			print("yummly_rf.pkl is not existed in the working directory.")
			print ("Please run Jupyter Notebook first.")
			sys.exit()
	else:
		# Parameters for Text CNN (Used while Training...)
		params = {
            "num_epochs": 50,
            "batch_size": 32,
            "num_filters": 64,
            "filter_sizes": "2,3,5",
            "embedding_dim": 128,
            "l2_reg_lambda": 1e-4,
            "evaluate_every": 100,
            "dropout_keep_prob": 0.5
        }

		# Load and clean data...
		x_test = load_test_data(tf_test)
		logging.info('\nTotal number of test inputs: {}'.format(len(x_test)))
		logging.info("Test data loaded...")
		
		# Generate Results with class probabilities...
		all_prob_scores = predict_unlabeled_data(x_test)
		finalResults = generateFinalResultsCsv(tf_test, all_prob_scores)
		logging.info("Preparing for CSV file..")

		# Save All Results to CSV...
		saveIntoCSV(finalResults, "text_cnn")