#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
0: books
1: dvd
2: electronics
3: kitchen
'''


# train_id = 0
# test_id = 2
rep = 3
num_data_target = 100
train_num = 1000
num_topics = 4
use_stoplist = False
diff = 2000
numIt = 50

unlabel = [4465, 3586, 5681, 5945]
#unlabel = [500, 500, 500, 500]
data_folder = './data/processed_acl/'
domains = ['books', 'dvd', 'electronics', 'kitchen']


def print_params():

	print (16*'*' + ' parameters ' + 16*'*')

	
	print ('repetition:', rep)
	print ('number of train samples for each polarity:', train_num)
	print ('number of labeled data in target domain:', num_data_target)
	print ('num_topics:', num_topics)
	print ('numIt:', numIt)
	print ('use_stoplist:', use_stoplist)
	print ('data_folder:', data_folder)
	print ('domains:', domains)

	print (22*'*' + 22*'*')