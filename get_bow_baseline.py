###################################################
## HELPER FUNCTIONS
## Test tokenization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn import linear_model
from collections import Counter
import pandas as pd
import sklearn.naive_bayes

def count_words(corpus):
	c = Counter()
	for doc in corpus:
		c.update(Counter(doc))
	return c

import sklearn.metrics
score_functions = {}
for name, val in sklearn.metrics.__dict__.items(): # iterate through every module's attributes
	if callable(val):					  # check if callable (normally functions)
		if 'score' in name:
			#print(name,val.__code__.co_varnames[0:2])
			score_functions[name] = val

warning_scores = ['adjusted_mutual_info_score','v_measure_score','normalized_mutual_info_score']
for col in warning_scores:
	del score_functions[col]

import json
import numpy as np

def get_score(y_true,pred,score,scores=False):
	d = {}
	if scores==False:
		scores = score_functions

	for name,func in scores.items():
		arguments = set(func.__code__.co_varnames)
		if 'score_func' in arguments:
			y_pred = score
		elif 'y_score' in arguments:
			y_pred = score
		elif 'labels_pred' in arguments:
			y_pred = pred
		elif 'y_pred' in arguments:
			y_pred = pred
		else:
			continue
		try:
			d[name] = func(y_true,y_pred)
		except:
			continue

	return d


from sklearn.linear_model import LogisticRegression
def get_nbratio(x,y,epsilon):
	p = x[y==1].sum(0)+1
	q = x[y==0].sum(0)+1
	r = np.log((p/p.sum())/(q/q.sum()))
	return r
import json
import numpy as np
###################################################

## Search for hyper parameters
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import csv
from hyperopt import Trials
from hyperopt import fmin
# optimization algorithm
tpe_algorithm = tpe.suggest
# Define the search space
#clf_space = ['logistic','nb_log','nb']
space_nbsvm = {'C':hp.loguniform('C', np.log(0.01), np.log(100)),'sign':hp.choice('sign',[False,True]),
#'best_classifier':hp.choice('best_classifier',clf_space), # leave this one out
'epsilon':hp.uniform('epsilon',0,2)}

# Global variable
global  ITERATION

ITERATION = 0
# Run optimization
#MAX_EVALS = 125

###################################################
import os
import time
from collections import Counter
class TokenizationTest():

	def __init__(self, train_df, test_df,text_col='text',log=None,outfile=None,y_col='y',MAX_EVALS=125,scoring_function='roc_auc_score'):

		# Create Y.
		le = preprocessing.LabelEncoder()
		le.fit(train_df[y_col].values)
		self.y_train = le.transform(train_df[y_col].values)
		self.y_test = le.transform(test_df[y_col])
		# define log mode
		self.log = log
		self.outfile = outfile
		if outfile!=None:x
			if not os.path.isfile(outfile):
				of_connection = open(outfile, 'a')
				writer = csv.writer(of_connection)
				header = ['loss', 'params', 'ITERATION', 'run_time','best_classifier','tokenizer_name','n_samples','n_features','t']
				writer.writerow(header)#['loss', 'params', 'ITERATION', 'run_time','best_classifier','tokenizer_name'])
				of_connection.close()

		# define Text
		self.train_docs = train_df[text_col].values
		self.test_docs = test_df[text_col].values
		## for error analysis purposes keep test_df
		self.test_df = test_df
		#self.x_train
		#self.y_train
		#self.tokenize_name
		#self.tokenize_func
		## Hyperparameter search
		self.MAX_EVALS = MAX_EVALS
		self.scoring_function = scoring_function
	def preprocess(self, tokenize_func,tokenizer_name):
		self.tokenize_func = tokenize_func
		self.tokenizer_name = tokenizer_name

		train_corpus=[]
		#c = Counter()
		for text in self.train_docs:
			tokens = tokenize_func(text)
			#for token in tokens:
			#	c[token]+=1
			train_corpus.append(' '.join(tokens))
		self.train_corpus = train_corpus

		test_corpus = []
		for text in self.test_docs:
			test_corpus.append(' '.join(tokenize_func(text)))
		self.test_corpus = test_corpus
		## Transform to Sparse Document-Term matrix with Tri-Gram features.
		vectorizer = CountVectorizer(ngram_range=(1,3), analyzer=str.split, max_features=500000,min_df=2)
		x_train = vectorizer.fit_transform(train_corpus)
		x_test = vectorizer.transform(test_corpus)
		self.x_train = x_train
		self.x_test = x_test

	def nblog(self,params={'sign':True,'epsilon':1,'C': 0.1},n_folds=10,sub_sampling=None
		  ,scoring_function='roc_auc_score'):
		"""Objective function for NBSVM Hyperparameter Optimization"""
		if sub_sampling==None:
			sub_samling = 1/n_folds
		scoring_function = self.scoring_function
		x_train = self.x_train
		n_features = x_train.shape[1]
		y_train = self.y_train
		tokenizer_name = self.tokenizer_name
		log=self.log
		outfile=self.outfile
		start = timer()
		 # Keep track of evals
		global ITERATION
		sign,epsilon,C = params['sign'],params['epsilon'],params['C']
		ITERATION += 1
		if sign:
			X=x_train.sign()
		else:
			X = x_train.copy()
		Y = y_train.copy()
		n = int(X.shape[0]*sub_sampling)
		res = []
		for nfold in range(n_folds):
			## Subsample for n-fold cross validation
			sort = np.random.permutation(np.arange(X.shape[0]))
			X,Y = X[sort],Y[sort]
			x,y = X[:n],Y[:n]
			x_test,y_test = X[n:],Y[n:]
			## Train classifiers
			## NB
			r = get_nbratio(x,y,epsilon)
			#r = np.log(prob(x,y,1,epsilon)/prob(x,y,0,epsilon))
			b = np.log((y==1).mean() / (y==0).mean())
			#nb prediction
			probs = x_test @ r.T + b
			probs = np.asarray(probs)[:,0]
			preds = probs.T>0

			if sum(preds)==0:
				acc = sum(preds == y_test)/len(y_test)
				d = {'accuracy_score':acc,scoring_function:0}

			else:
				d = get_score(y_test,preds,probs)


			d['name'] = 'nb'
			d['sign'] = sign
			d['epsilon'] = epsilon
			d['C'] = C
			d['nfold'] = nfold
			d['ITERATION'] = ITERATION
			d['tokenizer_name'] = tokenizer_name
			d['n_samples'] = n
			d['n_features'] = n_features
			t = time.time()
			d['t'] = t
			res.append(d)

			# LOGISTIC
			m = LogisticRegression(C=C, dual=True,solver='liblinear',max_iter=1000)
			m.fit(x, y);

			preds = m.predict(x_test)
			probs = m.predict_proba(x_test)

			if sum(preds)==0:
				acc = sum(preds == y_test)/len(y_test)
				d = {'accuracy_score':acc,scoring_function:0}

			else:
				d = get_score(y_test,preds,probs[:,1])

			d['name'] = 'logistic'
			d['sign'] = sign
			d['epsilon'] = epsilon
			d['C'] = C
			d['nfold'] = nfold
			d['ITERATION'] = ITERATION
			d['tokenizer_name'] = tokenizer_name
			d['n_samples'] = n
			d['n_features'] = n_features
			t = time.time()
			d['t'] = t
			res.append(d)
			#logistic with nb features
			x_nb = x.multiply(r)

			m = LogisticRegression(dual=True, C=C,solver='liblinear',max_iter=1000)
			m.fit(x_nb, y);

			val_bows_nb = x_test.multiply(r)
			probs = m.predict_proba(val_bows_nb)
			preds = m.predict(val_bows_nb)

			if sum(preds)==0:
				acc = sum(preds == y_test)/len(y_test)
				d = {'accuracy_score':acc,scoring_function:0}

			else:
				d = get_score(y_test,preds,probs[:,1])

			d['name'] = 'nb_log'
			d['sign'] = sign
			d['epsilon'] = epsilon
			d['C'] = C
			d['nfold'] = nfold
			d['ITERATION'] = ITERATION
			d['tokenizer_name'] = tokenizer_name
			d['n_samples'] = n
			d['n_features'] = n_features
			t = time.time()
			d['t'] = t
			res.append(d)


		if log!=None:
			f = open(self.log,'a')

			for d in res:
				new_d = {}
				for key in d:

					typ = type(d[key])
					if typ == np.ndarray:
						new_d[key] = list(d[key])
					elif typ==tuple:
						new_d[key] = list(d[key])
					else:
						new_d[key] = d[key]
				try:
					del new_d['precision_recall_fscore_support']
				except:
					pass
				f.write(json.dumps(new_d)+'\r\n')
			f.close()
		run_time = timer() - start

		# Extract the best score
		## Best classifier is assumed to be nb_log
		res_df = pd.DataFrame(res)
		## score should be maximized
		

		#best = mean_score.sort_values()

		#best_classifier = best.index[-1]

		mean_score = res_df.groupby('name')[scoring_function].mean()
		best_classifier = 'nb_log'
		score = res_df[res_df.name=='nb_log'][scoring_function].mean()


		#score = best.values[-1]

		# Loss must be minimized
		loss = 1 - score
		params['best_classifier'] = best_classifier

		# Write to the csv file ('a' means append)
		if outfile!=None:
			n_samples = n
			t = time.time()
			of_connection = open(outfile, 'a')
			writer = csv.writer(of_connection)
			writer.writerow([loss, params, ITERATION, run_time,best_classifier,tokenizer_name,n_samples,n_features,t])
			of_connection.close()
		# Dictionary with information for evaluation
		return {'loss': loss, 'params': params, 'iteration': ITERATION,
				'train_time': run_time, 'status': STATUS_OK,'best_classifier':best_classifier}

	def search_hyperparameters(self):
		# Keep track of results
		global ITERATION
		ITERATION = 0
		bayes_trials = Trials()
		self.trials = bayes_trials
		MAX_EVALS = self.MAX_EVALS
		best = fmin(fn = self.nblog, space = space_nbsvm, algo = tpe.suggest,
			max_evals = MAX_EVALS, trials = self.trials, rstate = np.random.RandomState(50))

		# Get min value
		best_ = min(bayes_trials.results,key=lambda x: x['loss'])
		best_params = best_['params']
		#best_clf = best_['best_classifier']
		self.best_params2 = best_params
		#self.best_clf2 = best_clf
		self.min_loss = best_
		self.best_params = best
		self.best_clf = 'nb_log'
		#self.best_clf = best['best_classifier']

	def evaluate(self, tokenizer_name,tokenize_func):
		# run tokenization scheme
		self.preprocess(tokenize_func,tokenizer_name)
		self.search_hyperparameters()
		# train final classifier
		## NB
		params = self.best_params

		sign = params['sign']
		epsilon = params['epsilon']
		C = params['C']
		if sign==1:
			x_train = self.x_train.sign()
			x_test = self.x_test.sign()
		else:
			x_train = self.x_train
			x_test = self.x_test

		y_test = self.y_test
		y = self.y_train
		#best_clf  = self.best_clf
		#best_clf = clf_space[best_clf]
		## asumme nl_log did best.
		best_clf = 'nb_log'
		if best_clf == 'nb':
			if sign:
				clf = sklearn.naive_bayes.BernoulliNB()
			else:
				clf = sklearn.naive_bayes.MultinomialNB()
			clf.fit(x_train,self.y_train)


			pred,probs = clf.predict(x_test),clf.predict_proba(x_test)[:,1]

			if sum(pred)==0:
				acc = sum(pred == y_test)/len(y_test)
				scores = {'accuracy_score':acc}
			else:
				scores = get_score(y_test,pred,probs)

			self.final_score = scores
			self.clf = clf

			print('Final accuracy and roc_auc score of tokenizer (%s) + %s: %.3f and %.3f'%(self.tokenizer_name,best_clf,scores['accuracy_score'],scores['roc_auc_score']))


		if best_clf == 'logistic':
			clf = LogisticRegression(C=C, dual=True,solver='liblinear',max_iter=1000)
			clf.fit(x_train, y);
			pred,probs = clf.predict(x_test),clf.predict_proba(x_test)[:,1]
			if sum(pred)==0:
				acc = sum(pred == y_test)/len(y_test)
				scores = {'accuracy_score':acc}
			else:
				scores = get_score(y_test,pred,probs)
			self.final_score = scores
			self.clf = clf

			print('Final accuracy and roc_auc score of tokenizer (%s) + %s: %.3f and %.3f'%(self.tokenizer_name,best_clf,scores['accuracy_score'],scores['roc_auc_score']))

		elif best_clf == 'nb_log':
			#r = np.log(prob(x_train,y,1,epsilon)/prob(x_train,y,0,epsilon))
			r = get_nbratio(x_train,y,epsilon)
			b = np.log((y==1).mean() / (y==0).mean())

			x_nb = x_train.multiply(r)

			clf = LogisticRegression(dual=True, C=C,solver='liblinear',max_iter=1000)
			clf.fit(x_nb, y);

			x_nb_test = x_test.multiply(r)
			pred,probs = clf.predict(x_nb_test),clf.predict_proba(x_nb_test)[:,1]
			if sum(pred)==0:
				acc = sum(pred == y_test)/len(y_test)
				scores = {'accuracy_score':acc}
			else:
				scores = get_score(y_test,pred,probs)

			self.final_score = scores
			self.clf = clf

			## Consider logging false predictions for later analysis.
			print('Final accuracy and roc_auc score of tokenizer (%s) + %s: %.3f and %.3f'%(self.tokenizer_name,best_clf,scores['accuracy_score'],scores['roc_auc_score']))
		## Consider logging false predictions for later analysis.
		#error_bool = pred != y_test
		#self.test_df['predict_prob'] = probs
		# dump index and probs
		#test_df['id'],probs






## Test should.
### Print the number of features. Number of features only in train. Percentage of features in train.
### Run cross-validation to get the best fit.
#### Represent 2(3) different features.
##### BOW - 1-2 ngrams
##### CBOW: Word2vec representation. Fitted.
##### NBLOG on CBOW and BOW
#### Should estimate learning rate at each step.
