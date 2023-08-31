#!/usr/bin/env python
# -*- coding: utf-8 -*-
from _csv import writer

import params
from collections import defaultdict
import numpy as np
from gensim import corpora, models, similarities
import csv

def loadData(filename, n):
    corpus, label = [], []
    fr = open(filename, 'r')
    all_words = set([])

    for i in range(n):
        line = fr.readline().strip().split()
        words = []
        for j in range(len(line)-1):
        	x = line[j].split(':')
        	word = x[0]
        	num = int(x[1]) # bag of words
        	for k in range(num):
        		words.append(word)
        	

        	# if params.useStopList is True and word not in params.stoplist:
        	# 	all_words.add(word)

        	# print (words)
        if line != []:
        	polarity = line[-1].split(':')
        	# print (polarity)
        	polarity = polarity[1]
	        if polarity == 'positive':
	        	label.append(1)
	        elif polarity == 'negative':
	        	label.append(-1)
	        corpus.append(' '.join(words))
    return corpus, label

def write_to_file(filename,data_val):
    with open(filename, 'a',newline='') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerows(data_val)
        # Close the file object
        f_object.close()


def Format(documents):

	stoplist = []
	with open('stopwords.txt', encoding="utf8") as fr:
	    for word in fr.readlines():
	        stoplist.append(word.strip())
	stoplist = set(stoplist)

	# stoplist = set('for a of the and to in i not\' his he <num>\' not was their who <num> one more you\' not you  all if my her\' about one what how they we which some so very no only other just me out like when time first even\' her she your many'.split())
	if params.use_stoplist == False:
		stoplist = set('for a of the and to i in not'.split())	# stop words list

	texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
	# print (len(texts))

	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1
	texts = [[token for token in text if frequency[token] > 100 and len(token) > 2]	# stop words with too high or low frequency
				for text in texts]

	# tt = np.array(texts)
	# print (tt.shape)
	# print (texts[0])
	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]
	# print corpus

	n = len(dictionary)
	print ('dictionary size:', n)
	m = len(texts)
	# print (m)

	num_topics = params.num_topics

	lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
	doc_transformed = list(lda.get_document_topics(corpus))
	# # print doc_transformed

	# topic matrix
	data_topic = np.zeros((m, num_topics))
	i = 0
	for doc in doc_transformed:
		for word in doc:
			data_topic[i][word[0]] = word[1]
		i += 1
	# print (lda.print_topics(num_topics))

	# bag of words matrix
	data_bow = np.zeros((m, n))
	i = 0
	for corp in corpus:
		for word in corp:
			data_bow[i][word[0]] = word[1]
		i += 1

	#write_to_file("Bow-Data.csv", data_bow)
	# print (data_topic.shape,data_bow.shape)
	# print (data_bow)
	data_mat = np.concatenate((data_bow, data_topic), axis=1)
	# print (data_mat.shape)
	p=np.array(data_bow)
	# write_to_file("data_file1.csv", dictionary.id2token.values())
	# write_to_file("data_file.csv",p)

	return [data_mat,data_bow, data_topic]

	# return lda, dictionary, corpus

def Doc(domain, args):
	filename = params.data_folder + domain
	# print (filename)
	corpus_neg_src, label_neg_src = loadData(filename+'/negative.review', params.train_num)
	corpus_pos_src, label_pos_src = loadData(filename+'/positive.review', params.train_num)
	
	corpus_target, label_target = loadData(filename+'/unlabeled.review', params.unlabel[args.tar_id])
	# print (len(corpus_neg_src), len(label_neg_src), len(corpus_pos_src), len(label_pos_src), len(corpus_target), len(label_target)) # 1000, 1000, 1000, 1000, _, _,
	documents = corpus_neg_src + corpus_pos_src + corpus_target
	labels = label_neg_src + label_pos_src + label_target
	
	
	return documents, labels


def evaluate(pred, label):
	cnt = 0
	for i in range(len(label)):
		if label[i] == pred[i]:
			cnt += 1
	return cnt * 1.0 / len(label)


