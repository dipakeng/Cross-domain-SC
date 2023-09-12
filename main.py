import csv
import numpy
import time
import selector as slctr


import math
from csv import writer
from sklearn.metrics import jaccard_score
from pyitlib.discrete_random_variable import information_mutual, entropy_cross
from scipy import stats
from pyitlib import discrete_random_variable as drv
from sklearn.metrics.pairwise import cosine_similarity, check_pairwise_arrays
import argparse
import pandas as pd
from gensim.parsing import PorterStemmer
import params
from utils import *

import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import csv
from matplotlib import pyplot as plt
import xlsxwriter
import NN
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics import mutual_info_score
# from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame

def optimization(TrainFile, TestFile):
    # Select optimizers
    vIGWO=True
    GWO = True
    optimizer = [vIGWO,GWO]

    #datasets = ["filtered_cross_train_data - Label"]
    datasets = [TrainFile]

    # Select number of repetitions for each experiment.

    NumOfRuns = 1

    # Select general parameters for all optimizers (population size, number of iterations)
    PopulationSize = 100
    Iterations = 500

    # Export results ?
    Export = True

    # ExportToFile="YourResultsAreHere.csv"
    # Automaticly generated file name by date and time
    ExportToFile = "experiment" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv"

    # Check if it works at least once
    Flag = False

    # CSV Header for for the cinvergence
    CnvgHeader = []

    for l in range(0, Iterations):
        CnvgHeader.append("Iter" + str(l + 1))

    #trainDataset = TrainFile #"filtered_cross_train_data - Label.csv"
    #testDataset = TestFile #"data_Label_test_after_filter.csv"
    for j in range(0, len(datasets)):  # specfiy the number of the datasets
        for i in range(0, len(optimizer)):

            if ((optimizer[i] == True)):  # start experiment if an optimizer and an objective function is selected
                for k in range(0, NumOfRuns):

                    func_details = ["costNN", -1, 1]
                    trainDataset = datasets[j] + "Train.csv"
                    testDataset = datasets[j] + "Test.csv"
                    x = slctr.selector(i, func_details, PopulationSize, Iterations, trainDataset, testDataset)

                    if (Export == True):
                        with open(ExportToFile, 'a', newline='\n') as out:
                            writer = csv.writer(out, delimiter=',')
                            if (Flag == False):  # just one time to write the header of the CSV file
                                header = numpy.concatenate([["Optimizer", "Dataset", "objfname", "Experiment", "startTime",
                                                             "EndTime", "ExecutionTime", "trainAcc", "trainTP", "trainFN",
                                                             "trainFP", "trainTN", "testAcc", "testTP", "testFN", "testFP",
                                                             "testTN"], CnvgHeader])
                                writer.writerow(header)
                            a = numpy.concatenate([[x.optimizer, datasets[j], x.objfname, k + 1, x.startTime, x.endTime,
                                                    x.executionTime, x.trainAcc, x.trainTP, x.trainFN, x.trainFP, x.trainTN,
                                                    x.testAcc, x.testTP, x.testFN, x.testFP, x.testTN], x.convergence])
                            writer.writerow(a)
                        out.close()
                    Flag = True  # at least one experiment

    if (Flag == False):  # Faild to run at least one experiment
        print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convert_to_binary(vector1):
    prob = []
    vector1 = np.asarray(vector1)
    vector1[:][vector1 >= 1] = 1

def convert_to_probability(vector1):  # pass matrix
    [rw, cl] = vector1.shape
    prob = []
    # prob= np.zeros((rw,cl))
    for i in vector1:
        ans = sum(i)
        if ans != 0:
            i = i / ans
        prob.append(i)
    prob = np.asarray(prob)
    return prob


def find_Jaccard_sim(vect1,vect2):  # find cross entropy between each element of vect1 with vect2 and find max similarity
    j = 0
    vect1 = np.asarray(vect1)
    vect2 = np.asarray(vect2)
    vect1 = convert_to_binary(vect1)
    vect2 = convert_to_binary(vect2)
    [rw, cl] = vect2.shape  # uncomment if in first for loop vect2 is there
    # [rw, cl] = vect1.shape   # uncomment if in first for loop vect1 is there
    max_ele = np.zeros((rw, 3))
    for line1 in vect2:
        temp_sim = []
        for line2 in vect1:
            temp = jaccard_score(line2, line1)
            # if np.isnan(temp)*1 ==1:
            #     temp=0
            temp_sim.append(temp)
        temp_sim = np.asarray(temp_sim)
        max_ele[j][0] = max(temp_sim)
        print(np.argmax(temp_sim, axis=0))
        max_ele[j][1] = np.argmax(temp_sim, axis=0)  # get position of max smililarity for lable
        max_ele[j][2] = j
        j = j + 1
    result1 = sorted(max_ele, key=lambda x: x[0], reverse=True)
    result1 = np.asarray(result1)
    return result1


def find_cosine_sim(vect1,vect2):  # find the similarity of each element of vect2 with vect1 and stores it in sorted array
    j = 0
    vect1 = np.asarray(vect1)
    vect2 = np.asarray(vect2)
    result = cosine_similarity(vect2, vect1)

    [rw, cl] = result.shape
    result1 = np.zeros((rw, 3))
    max_ele = np.max(result, axis=1)
    max_pos = np.argmax(result, axis=1)
    i = 0
    for ele in max_ele:
        result1[i][0] = ele
        i = i + 1
    i = 0
    for pos in max_pos:
        result1[i][1] = pos
        result1[i][2] = i
        i = i + 1

    result1 = sorted(result1, key=lambda x: x[0], reverse=True)
    result1 = np.asarray(result1)
    return result1

def bhatacharya_dist(p, q):
    q = np.asarray(q)
    # q[q==0]=1
    sum = 0.0
    for i in range(len(p)):
        if q[i] != 0 and p[i] != 0:
            sum = sum + (math.log(math.sqrt(p[i] * q[i]), math.e))
    return -sum

def find_bhatacharya(vect1, vect2):  # find bhatacharya distance
    j = 0
    vect1 = np.asarray(vect1)
    vect2 = np.asarray(vect2)
    vect1 = convert_to_probability(vect1)
    vect2 = convert_to_probability(vect2)
    [rw, cl] = vect2.shape  # uncomment if in first for loop vect2 is there
    # [rw, cl] = vect1.shape   # uncomment if in first for loop vect1 is there
    max_ele = np.zeros((rw, 3))
    for line1 in vect2:  # vect2 is target domain
        temp_sim = []
        for line2 in vect1:  # vect1 is Source domain
            temp = bhatacharya_dist(line2, line1)
            temp_sim.append(temp)
        # temp_sim.append(-1)              # assign some large value which is uaeful if all values are 0 in temp_sim
        temp_sim = np.asarray(temp_sim)
        max_ele[j][0] = max(temp_sim)
        max_ele[j][1] = np.argmax(temp_sim, axis=0)  # get position of max smililarity for lable
        max_ele[j][2] = j
        j = j + 1
    result1 = sorted(max_ele, key=lambda x: x[0], reverse=True)
    # result1 = sorted(max_ele, key=lambda x: x[0])
    result1 = np.asarray(result1)
    return result1


def KLDiversion(p, q):  # need to modify for q*log p
    q = np.asarray(q)
    # q[q==0]=1
    sum = 0.0
    for i in range(len(p)):
        if q[i] != 0 and p[i] != 0:
            sum = sum + (p[i] * np.log10(p[i] / q[i]))
    return sum


def find_KLDiversion(vect1,vect2):  # find cross entropy between each element of vect1 with vect2 and find max similarity
    j = 0
    vect1 = np.asarray(vect1)
    vect2 = np.asarray(vect2)
    vect1 = convert_to_probability(vect1)
    vect2 = convert_to_probability(vect2)
    [rw, cl] = vect2.shape  # uncomment if in first for loop vect2 is there
    # [rw, cl] = vect1.shape   # uncomment if in first for loop vect1 is there
    max_ele = np.zeros((rw, 3))
    for line1 in vect2:  # vect2 is target domain
        temp_sim = []
        for line2 in vect1:  # vect1 is Source domain
            temp = KLDiversion(line2, line1)
            temp_sim.append(temp)
        # temp_sim.append(-1)              # assign some large value which is uaeful if all values are 0 in temp_sim
        temp_sim = np.asarray(temp_sim)
        max_ele[j][0] = min(temp_sim)
        max_ele[j][1] = np.argmin(temp_sim, axis=0)  # get position of max smililarity for lable
        max_ele[j][2] = j
        j = j + 1
    result1 = sorted(max_ele, key=lambda x: x[0])
    # result1 = sorted(max_ele, key=lambda x: x[0])
    result1 = np.asarray(result1)
    return result1


def cross_entropy(p, q):  # need to modify for q*log p
    q = np.asarray(q)
    # q[q==0]=1
    sum = 0.0
    for i in range(len(p)):
        if q[i] != 0:
            sum = sum + (p[i] * np.log2(q[i]))
    return -sum


def mod_cross_entropy(p, q):  # p source and q target
    q = np.asarray(q)
    p = np.asarray(p)
    # q[q==0]=1
    sum = 0.0
    for i in range(len(p)):
        if q[i] != 0 and p[i] != 0:
            sum = sum + ((1 - p[i]) * np.log10(q[i])) * (1 - abs(p[i] - q[i]))

    return -sum


def find_mod_cross_entropy(vect1,vect2):  # find cross entropy between each element of vect1 with vect2 and find max similarity
    j = 0
    vect1 = np.asarray(vect1)
    vect2 = np.asarray(vect2)
    vect1 = convert_to_probability(vect1)
    vect2 = convert_to_probability(vect2)

    # np.savetxt('Training1.csv', vect1, fmt='%f', delimiter=",")
    # np.savetxt('Test1.csv', vect2, fmt='%f', delimiter=",")

    [rw, cl] = vect2.shape  # uncomment if in first for loop vect2 is there
    # [rw, cl] = vect1.shape   # uncomment if in first for loop vect1 is there
    max_ele = np.zeros((rw, 3))
    for line1 in vect2:  # vect2 is source domain
        temp_sim = []
        for line2 in vect1:  # vect1 is target domain
            temp = mod_cross_entropy(line2, line1)
            # if np.isnan(temp)*1 ==1:
            #     temp=0
            temp_sim.append(temp)
        # temp_sim.append(-1)              # assign some large value which is uaeful if all values are 0 in temp_sim
        temp_sim = np.asarray(temp_sim)

        max_ele[j][0] = max(temp_sim)
        # print(np.argmax(temp_sim,axis=0))
        max_ele[j][1] = np.argmax(temp_sim, axis=0)  # get position of max smililarity for lable
        max_ele[j][2] = j
        j = j + 1

    result1 = sorted(max_ele, key=lambda x: x[0], reverse=True)
    # result1 = sorted(max_ele, key=lambda x: x[0])
    result1 = np.asarray(result1)
    return result1


def find_cross_entropy(vect1,vect2):  # find cross entropy between each element of vect1 with vect2 and find max similarity
    j = 0
    vect1 = np.asarray(vect1)
    vect2 = np.asarray(vect2)
    vect1 = convert_to_probability(vect1)
    vect2 = convert_to_probability(vect2)

    #np.savetxt('Training1.csv', vect1, fmt='%d', delimiter=",")
    #np.savetxt('Test1.csv', vect2, fmt='%d', delimiter=",")
    [rw, cl] = vect2.shape  # uncomment if in first for loop vect2 is there
    # [rw, cl] = vect1.shape   # uncomment if in first for loop vect1 is there
    max_ele = np.zeros((rw, 3))
    for line1 in vect2:
        temp_sim = []
        for line2 in vect1:
            temp = cross_entropy(line2, line1)
            # if np.isnan(temp)*1 ==1:
            #     temp=0
            temp_sim.append(temp)
        temp_sim.append(500)  # assign some large value which is uaeful if all values are 0 in temp_sim
        temp_sim = np.asarray(temp_sim)
        temp_sim = temp_sim[temp_sim != 0]
        r = temp_sim.shape
        # if(r==0):
        #     temp_sim[0]=500   # assign some large value
        max_ele[j][0] = min(temp_sim)
        # print(np.argmin(temp_sim,axis=0))
        max_ele[j][1] = np.argmin(temp_sim, axis=0)  # get position of max smililarity for lable
        max_ele[j][2] = j
        j = j + 1

    # result1=sorted(max_ele, key=lambda x: x[0],reverse=True)
    result1 = sorted(max_ele, key=lambda x: x[0])
    result1 = np.asarray(result1)
    return result1


def find_mutual_info(vect1, vect2):  # find the MI of each element of vect2 with vect1 and stores it in sorted array
    j = 0
    vect1 = np.asarray(vect1)
    vect2 = np.asarray(vect2)

    result = information_mutual(vect2, vect1, cartesian_product=True)

    [rw, cl] = result.shape
    result1 = np.zeros((rw, 2))
    max_ele = np.max(result, axis=1)
    max_pos = np.argmax(result, axis=1)
    i = 0
    for ele in max_ele:
        result1[i][0] = ele
        i = i + 1
    i = 0
    for pos in max_pos:
        result1[i][1] = pos
        i = i + 1

    result1 = sorted(result1, key=lambda x: x[0], reverse=True)
    result1 = np.asarray(result1)
    return result1


def Create_doc_term_matrix1(documents):
    stoplist = []
    with open('stopwords.txt', encoding="utf8") as fr:
        for word in fr.readlines():
            stoplist.append(word.strip())
    stoplist = set(stoplist)

    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    # print (len(texts))
    stemmer = PorterStemmer()

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            token = stemmer.stem(token)  # apply stemming
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 20 and len(token) > 2]
             # stop words with too high or low frequency
             for text in texts]
    texts = list(texts)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    n = len(dictionary)
    m = len(texts)
    data_bow = np.zeros((m, n))
    i = 0
    for corp in corpus:
        for word in corp:
            data_bow[i][word[0]] = word[1]
        i += 1
    # print corpus

#    return dense
def Create_doc_term_matrix(documents):

    vectorizer = TfidfVectorizer(max_df=0.65, min_df=80, stop_words='english', use_idf=True, norm=None)
    # vectorizer = TfidfTransformer()
    # vectorizer = TfidfVectorizer()
    doc_term_mat = vectorizer.fit_transform(documents)
    dense = doc_term_mat.todense()
    # denselist = dense.tolist()
    # return DataFrame(denselist,columns=vectorizer.get_feature_names())
    return dense


def train_and_evaluate_NN(X_train, Y_train, X_test, Y_test):
    # Create a model
    model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                                                 alpha=0.0001, batch_size='auto', learning_rate='constant',
                                                 learning_rate_init=0.001, power_t=0.5,
                                                 max_iter=500, shuffle=True, random_state=None, tol=0.0001,
                                                 verbose=False, warm_start=False, momentum=0.9,
                                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                                 beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                 n_iter_no_change=10)

    # Train the model on the whole data set
    model.fit(X_train, Y_train)
    # Evaluate on test data
    predictions = model.predict(X_test)
    return predictions


def main(Laa, params, args):
    src_id_1 = int(Laa[0])
    src_id_2 = int(Laa[1])
    src_id_3 = int(Laa[2])
    tar_id_1 = int(Laa[3])
    pred_2 = []
    train_doc_1, train_labels_1 = Doc(params.domains[src_id_1], args)
    train_doc_2, train_labels_2 = Doc(params.domains[src_id_2], args)
    train_doc_3, train_labels_3 = Doc(params.domains[src_id_3], args)
    test_doc1, test_labels1 = Doc(params.domains[tar_id_1], args)

    Statm_P=250
    Statm_N=250

    len_test = len(train_doc_1)
    train_doc_1 = train_doc_1[0:Statm_P] + train_doc_1[1000:1000+Statm_N]
    train_labels_1 = train_labels_1[0:Statm_P] + train_labels_1[1000:Statm_N]

    len_test = len(train_doc_2)
    train_doc_2 = train_doc_2[0:Statm_P] + train_doc_2[1000:Statm_N]
    train_labels_2 = train_labels_2[0:Statm_P] + train_labels_2[1000:Statm_N]

    len_test = len(train_doc_3)
    train_doc_3 = train_doc_3[0:Statm_P] + train_doc_3[1000:Statm_N]
    train_labels_3 = train_labels_3[0:Statm_P] + train_labels_3[1000:Statm_N]

    len_test = len(test_doc1)
    test_doc = test_doc1[0:Statm_P] + test_doc1[1000:Statm_N]
    test_labels = test_labels1[0:Statm_P] + test_labels1[1000:Statm_N]

    # =======================================================
    train_doc = train_doc_1 + train_doc_2 + train_doc_3
    train_labels = train_labels_1 + train_labels_2 + train_labels_3
    tr_per = 1.0
    total_train_num = int(np.round(len(train_labels) * tr_per))
    total_train_num_1 = 0
    filtered_cross_train_data = []
    filtered_cross_train_label = []
    data_test_after_filter = []
    label_test_after_filter = []
    filtered_cross_train_data_500 = []
    filtered_cross_train_label_500 = []


    type = 6  # 1 for BOW model and 2 for TF-IDF model based cosine similarities and 3 for cosine similarities 4 for cross entropy 5 for jaccard similarity 6 for modified cross entropy 7 for KL Diversion
    if type == 1:
        print("hi")
    elif type == 4:  # cross entropy
        # ==== for frequency model ========
        model = "Freq"  # freq model or TF-IDF
        if model == "Freq":
            data_mat, data_bow, data_topic = Format(
                train_doc[:total_train_num] + test_doc[
                                              total_train_num_1:total_train_num_1 + params.unlabel[args.tar_id]])
            fin_lab = train_labels[:total_train_num] + test_labels[
                                                       total_train_num_1:total_train_num_1 + params.unlabel[
                                                           args.tar_id]]
            data_train = data_mat[:total_train_num]  # LDA + BOW
            label_train = fin_lab[:total_train_num]

            data_train_1 = data_bow[:total_train_num]  # BOW with frequency
            data_test_1 = data_bow[total_train_num:]

            data_test = data_mat[total_train_num:]
            label_test = fin_lab[total_train_num:]
            label_test = np.array(label_test)
            # no_att = len(data_test[0])
            # no_training_sample=len(data_train)
            #no_training_sample = len(data_train_1)
            no_att = len(data_test_1[0])

            ans = find_cross_entropy(data_train_1, data_test_1)

            count = 0
            mrange = 300
            for i in range(0, mrange):
                # data_train_cosine[i][:]=data_train_1[ans[i][1]][:]
                # label_train_cosine[i][:] = label_train[ans[i][1]]
                if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
                    count = count + 1
                    filtered_cross_train_data.append(data_test_1[int(
                        ans[i][2])])  # get target domain data which are found to be most similar with source domain
                    filtered_cross_train_label.append(label_test[int(ans[i][2])])  # get corresponding lables
                else:
                    data_test_after_filter.append(data_test_1[int(ans[i][2])])
                    label_test_after_filter.append(label_test[int(ans[i][2])])
            # print("Count for first ", mrange, " : ", count)

            # for i in range(mrange, Statm_P+Statm_N):
            #     if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
            #         count = count + 1
            # print("Count for rest of is:", count)

            for i in range(mrange, Statm_P+Statm_N):
                data_test_after_filter.append(data_test_1[int(ans[i][2])])
                label_test_after_filter.append(label_test[int(ans[i][2])])

            filtered_cross_train_label_500 = filtered_cross_train_label.copy()

            for i in range(0, len(data_train_1)):
                filtered_cross_train_data.append(data_train_1[i])  # consider original statements from all threee domains
                filtered_cross_train_label.append(label_train[i])

            filtered_cross_train_data = np.asarray(filtered_cross_train_data)
            filtered_cross_train_label = np.asarray(filtered_cross_train_label)
            data_test_after_filter = np.asarray(data_test_after_filter)
            label_test_after_filter = np.asarray(label_test_after_filter)

            pred_2 = train_and_evaluate_NN(filtered_cross_train_data, filtered_cross_train_label,
                                           data_test_after_filter, label_test_after_filter)
            pred_2 = np.asarray(pred_2)
            pred_2[pred_2 == -1] = 0

        else:
            # =============================== for tf-idf model
            # ==== TF-IDF vectorization

            # fin_lab = train_labels[:total_train_num] + test_labels[
            #                                           total_train_num_1:total_train_num_1 + params.unlabel[args.tar_id]]
            label_train = train_labels[:total_train_num]
            label_test = test_labels
            # count_vect = TfidfVectorizer()
            train_test_doc = train_doc + test_doc
            data_train_test = Create_doc_term_matrix(train_test_doc)
            data_train_test = np.array(data_train_test)
            data_train = data_train_test[:total_train_num]
            data_test = data_train_test[total_train_num:]

            no_att = len(data_test[0])
            no_training_sample = len(data_train)
            ans = find_mutual_info(data_test, data_train)
            count = 0
            for i in range(0, 250):
                # data_train_cosine[i][:]=data_train_1[ans[i][1]][:]
                # label_train_cosine[i][:] = label_train[ans[i][1]]
                if label_train[int(ans[i][1])] == label_test[int(i)]:
                    count = count + 1
            print(count)
    elif type == 5:
        data_mat, data_bow, data_topic = Format(
            train_doc[:total_train_num] + test_doc[
                                          total_train_num_1:total_train_num_1 + params.unlabel[args.tar_id]])
        fin_lab = train_labels[:total_train_num] + test_labels[
                                                   total_train_num_1:total_train_num_1 + params.unlabel[
                                                       args.tar_id]]
        data_train = data_mat[:total_train_num]  # LDA + BOW
        label_train = fin_lab[:total_train_num]

        data_train_1 = data_bow[:total_train_num]  # BOW with frequency
        data_test_1 = data_bow[total_train_num:]

        data_test = data_mat[total_train_num:]
        label_test = fin_lab[total_train_num:]
        label_test = np.array(label_test)
        # no_att = len(data_test[0])
        # no_training_sample=len(data_train)
        no_training_sample = len(data_train_1)
        no_att = len(data_test_1[0])
        ans = find_Jaccard_sim(data_train_1, data_test_1)

        count = 0
        mrange=300

        for i in range(0, mrange):
            # data_train_cosine[i][:]=data_train_1[ans[i][1]][:]
            # label_train_cosine[i][:] = label_train[ans[i][1]]
            if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
                count = count + 1
                filtered_cross_train_data.append(data_test_1[int(
                    ans[i][2])])  # get target domain data which are found to be most similar with source domain
                filtered_cross_train_label.append(label_test[int(ans[i][2])])  # get corresponding lables
            else:
                data_test_after_filter.append(data_test_1[int(ans[i][2])])
                label_test_after_filter.append(label_test[int(ans[i][2])])
        print("Count for first", mrange ,": ", count)
        for i in range(mrange, 500):
            data_test_after_filter.append(data_test_1[int(ans[i][2])])
            label_test_after_filter.append(label_test[int(ans[i][2])])

        filtered_cross_train_label_500 = filtered_cross_train_label.copy()

        for i in range(0, len(data_train_1)):
            filtered_cross_train_data.append(data_train_1[i])  # consider original statements from all threee domains
            filtered_cross_train_label.append(label_train[i])

        filtered_cross_train_data = np.asarray(filtered_cross_train_data)
        filtered_cross_train_label = np.asarray(filtered_cross_train_label)
        data_test_after_filter = np.asarray(data_test_after_filter)
        label_test_after_filter = np.asarray(label_test_after_filter)

        pred_2 = train_and_evaluate_NN(filtered_cross_train_data, filtered_cross_train_label,
                                       data_test_after_filter, label_test_after_filter)
        pred_2 = np.asarray(pred_2)
        pred_2[pred_2 == -1] = 0
    elif type == 6:  # modified cross entroy
        data_mat, data_bow, data_topic = Format(
            train_doc[:total_train_num] + test_doc[
                                          total_train_num_1:total_train_num_1 + params.unlabel[args.tar_id]])
        fin_lab = train_labels[:total_train_num] + test_labels[
                                                   total_train_num_1:total_train_num_1 + params.unlabel[
                                                       args.tar_id]]

        data_train = data_mat[:total_train_num]  # LDA + BOW
        label_train = fin_lab[:total_train_num]

        data_train_1 = data_bow[:total_train_num]  # BOW with frequency
        data_test_1 = data_bow[total_train_num:]

        data_test = data_mat[total_train_num:]
        label_test = fin_lab[total_train_num:]
        label_test = np.array(label_test)
        # no_att = len(data_test[0])
        # no_training_sample=len(data_train)
        no_training_sample = len(data_train_1)
        no_att = len(data_test_1[0])
        ans = find_mod_cross_entropy(data_train_1, data_test_1)
        count = 0
        mrange = 300
        for i in range(0, mrange):

            if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
                count = count + 1
                filtered_cross_train_data.append(data_test_1[int(
                    ans[i][2])])  # get target domain data which are found to be most similar with source domain
                # filtered_cross_train_data.append(data_train_1[int(ans[i][1])])  #get corresponding train data that matches with test dataset for training of machine

                filtered_cross_train_label.append(label_test[int(ans[i][2])])  # get corresponding lables
                # filtered_cross_train_label.append(label_train[int(ans[i][1])])
            # else:
            #     data_test_after_filter.append(data_test_1[int(ans[i][2])])
            #     label_test_after_filter.append(label_test[int(ans[i][2])])
        print("Count for first", mrange, ":", count)
        count = 0
        # for i in range(mrange, 500):
        #     if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
        #         count = count + 1
        # print("Count for rest of is:", count)

        for i in range(mrange, Statm_P+Statm_N):
            data_test_after_filter.append(data_test_1[int(ans[i][2])])
            label_test_after_filter.append(label_test[int(ans[i][2])])

        for i in range(0, mrange):
            data_test_after_filter.append(data_test_1[int(ans[i][2])])
            label_test_after_filter.append(label_test[int(ans[i][2])])

        filtered_cross_train_label_500 = filtered_cross_train_label.copy()
        # uncomment this
        for i in range(0, len(data_train_1)):
            filtered_cross_train_data.append(data_train_1[i])  # consider original statements from all threee domains
            filtered_cross_train_label.append(label_train[i])

        # commment this
        # for i in range(0, len(data_train_cosine)):
        #     filtered_cross_train_data.append(data_train_cosine[i])  # consider original statements from all threee domains
        #     filtered_cross_train_label.append(label_train_cosine[i])

        filtered_cross_train_data = np.asarray(filtered_cross_train_data)
        filtered_cross_train_label = np.asarray(filtered_cross_train_label)
        data_test_after_filter = np.asarray(data_test_after_filter)
        label_test_after_filter = np.asarray(label_test_after_filter)

        filtered_cross_train_data_label = np.concatenate([filtered_cross_train_data,filtered_cross_train_label],axis=1)
        data_label_test_after_filter = np.concatenate([data_test_after_filter,label_test_after_filter],axis=1)


        np.savetxt('filtered_cross_train_data-Label.csv', filtered_cross_train_data_label, fmt='%d', delimiter=",")
        #np.savetxt('filtered_cross_train_label.csv', filtered_cross_train_label, fmt='%d', delimiter=",")
        np.savetxt('data_test_after_filter.csv', data_label_test_after_filter, fmt='%d', delimiter=",")
        #np.savetxt('label_test_after_filter.csv', label_test_after_filter, fmt='%d', delimiter=",")

        optimization("filtered_cross_train_data-Label.csv","data_test_after_filter.csv")


        # pred_2 = train_and_evaluate_NN(filtered_cross_train_data, filtered_cross_train_label, data_test_after_filter,
        #                                label_test_after_filter)
        #
        # pred_2 = np.asarray(pred_2)
        # pred_2[pred_2 == -1] = 0
    elif type == 7:  # KLDiversion
        data_mat, data_bow, data_topic = Format(
            train_doc[:total_train_num] + test_doc[
                                          total_train_num_1:total_train_num_1 + params.unlabel[args.tar_id]])
        fin_lab = train_labels[:total_train_num] + test_labels[
                                                   total_train_num_1:total_train_num_1 + params.unlabel[
                                                       args.tar_id]]
        data_train = data_mat[:total_train_num]  # LDA + BOW
        label_train = fin_lab[:total_train_num]

        data_train_1 = data_bow[:total_train_num]  # BOW with frequency
        data_test_1 = data_bow[total_train_num:]

        data_test = data_mat[total_train_num:]
        label_test = fin_lab[total_train_num:]
        label_test = np.array(label_test)
        # no_att = len(data_test[0])
        # no_training_sample=len(data_train)
        no_training_sample = len(data_train_1)
        no_att = len(data_test_1[0])
        ans = find_KLDiversion(data_train_1, data_test_1)
        count = 0
        mrange = 300
        for i in range(0, mrange):
          if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
                count = count + 1
                filtered_cross_train_data.append(data_test_1[int(
                    ans[i][2])])  # get target domain data which are found to be most similar with source domain
                filtered_cross_train_label.append(label_test[int(ans[i][2])])  # get corresponding lables
            # else:
            #     data_test_after_filter.append(data_test_1[int(ans[i][2])])
            #     label_test_after_filter.append(label_test[int(ans[i][2])])
        print("Count for first", mrange, ":", count)
        count = 0
        for i in range(mrange, Statm_P+Statm_N):
            if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
                count = count + 1
        print("Count for rest of is:", count)

        for i in range(mrange, Statm_P+Statm_N):
            data_test_after_filter.append(data_test_1[int(ans[i][2])])
            label_test_after_filter.append(label_test[int(ans[i][2])])

        # for i in range(0, 300):
        #     data_test_after_filter.append(data_test_1[int(ans[i][2])])
        #     label_test_after_filter.append(label_test[int(ans[i][2])])

        filtered_cross_train_label_500 = filtered_cross_train_label.copy()

        for i in range(0, len(data_train_1)):
            filtered_cross_train_data.append(data_train_1[i])  # consider original statements from all threee domains
            filtered_cross_train_label.append(label_train[i])

        filtered_cross_train_data = np.asarray(filtered_cross_train_data)
        filtered_cross_train_label = np.asarray(filtered_cross_train_label)
        data_test_after_filter = np.asarray(data_test_after_filter)
        label_test_after_filter = np.asarray(label_test_after_filter)

        pred_2 = train_and_evaluate_NN(filtered_cross_train_data, filtered_cross_train_label, data_test_after_filter,
                                       label_test_after_filter)

        # pred_2 =Local_classifier_train_test(filtered_cross_train_data, filtered_cross_train_label, data_test_after_filter)

        pred_2 = np.asarray(pred_2)
        pred_2[pred_2 == -1] = 0

    elif type == 8:  # Bhatacharya
        data_mat, data_bow, data_topic = Format(
            train_doc[:total_train_num] + test_doc[
                                          total_train_num_1:total_train_num_1 + params.unlabel[args.tar_id]])
        fin_lab = train_labels[:total_train_num] + test_labels[
                                                   total_train_num_1:total_train_num_1 + params.unlabel[
                                                       args.tar_id]]
        data_train = data_mat[:total_train_num]  # LDA + BOW
        label_train = fin_lab[:total_train_num]

        data_train_1 = data_bow[:total_train_num]  # BOW with frequency
        data_test_1 = data_bow[total_train_num:]

        data_test = data_mat[total_train_num:]
        label_test = fin_lab[total_train_num:]
        label_test = np.array(label_test)
        # no_att = len(data_test[0])
        # no_training_sample=len(data_train)
        no_training_sample = len(data_train_1)
        no_att = len(data_test_1[0])
        ans = find_bhatacharya(data_train_1, data_test_1)
        count = 0
        mrange = 300
        for i in range(0, mrange):
            # data_train_cosine.append(data_train_1[int(ans[i][1])])  #comment
            # label_train_cosine.append(label_train[int(ans[i][1])])   #comment
            if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
                count = count + 1
                filtered_cross_train_data.append(data_test_1[int(
                    ans[i][2])])  # get target domain data which are found to be most similar with source domain
                filtered_cross_train_label.append(label_test[int(ans[i][2])])  # get corresponding lables
            # else:
            #     data_test_after_filter.append(data_test_1[int(ans[i][2])])
            #     label_test_after_filter.append(label_test[int(ans[i][2])])
        print("Count for first", mrange, ":", count)
        count = 0
        for i in range(mrange, Statm_P+Statm_N):
            if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
                count = count + 1
        print("Count for rest of is:", count)

        for i in range(mrange, Statm_P+Statm_N):
            data_test_after_filter.append(data_test_1[int(ans[i][2])])
            label_test_after_filter.append(label_test[int(ans[i][2])])

        # for i in range(0, 300):
        #     data_test_after_filter.append(data_test_1[int(ans[i][2])])
        #     label_test_after_filter.append(label_test[int(ans[i][2])])

        filtered_cross_train_label_500 = filtered_cross_train_label.copy()

        for i in range(0, len(data_train_1)):
            filtered_cross_train_data.append(data_train_1[i])  # consider original statements from all threee domains
            filtered_cross_train_label.append(label_train[i])

        filtered_cross_train_data = np.asarray(filtered_cross_train_data)
        filtered_cross_train_label = np.asarray(filtered_cross_train_label)
        data_test_after_filter = np.asarray(data_test_after_filter)
        label_test_after_filter = np.asarray(label_test_after_filter)

        pred_2 = train_and_evaluate_NN(filtered_cross_train_data, filtered_cross_train_label, data_test_after_filter,
                                       label_test_after_filter)

        pred_2 = np.asarray(pred_2)
        pred_2[pred_2 == -1] = 0
    elif type == 3:  # for cosine similarities
        data_mat, data_bow, data_topic = Format(
            train_doc[:total_train_num] + test_doc[total_train_num_1:total_train_num_1 + params.unlabel[args.tar_id]])
        fin_lab = train_labels[:total_train_num] + test_labels[
                                                   total_train_num_1:total_train_num_1 + params.unlabel[args.tar_id]]

        data_train = data_mat[:total_train_num]  # LDA + BOW
        label_train = fin_lab[:total_train_num]

        data_train_1 = data_bow[:total_train_num]  # BOW with frequency
        data_test_1 = data_bow[total_train_num:]

        data_test = data_mat[total_train_num:]
        label_test = fin_lab[total_train_num:]
        label_test = np.array(label_test)
        # no_att = len(data_test[0])
        # no_training_sample=len(data_train)
        no_training_sample = len(data_train_1)
        no_att = len(data_test_1[0])
        ans = find_cosine_sim(data_train_1, data_test_1)  # this is right

        count = 0
        mrange = 300
        for i in range(0, mrange):
            # data_train_cosine[i][:]=data_train_1[ans[i][1]][:]
            # label_train_cosine[i][:] = label_train[ans[i][1]]
            if label_train[int(ans[i][1])] == label_test[int(ans[i][2])]:
                count = count + 1
                filtered_cross_train_data.append(data_test_1[int(
                    ans[i][2])])  # get target domain data which are found to be most similar with source domain
                filtered_cross_train_label.append(label_test[int(ans[i][2])])  # get corresponding lables
            else:
                data_test_after_filter.append(data_test_1[int(ans[i][2])])
                label_test_after_filter.append(label_test[int(ans[i][2])])
        print("Count for first", mrange, ": ", count)
        for i in range(mrange, Statm_P+Statm_N):
            data_test_after_filter.append(data_test_1[int(ans[i][2])])
            label_test_after_filter.append(label_test[int(ans[i][2])])

        filtered_cross_train_label_500 = filtered_cross_train_label.copy()

        for i in range(0, len(data_train_1)):
            filtered_cross_train_data.append(data_train_1[i])  # consider original statements from all threee domains
            filtered_cross_train_label.append(label_train[i])

        filtered_cross_train_data = np.asarray(filtered_cross_train_data)
        filtered_cross_train_label = np.asarray(filtered_cross_train_label)
        data_test_after_filter = np.asarray(data_test_after_filter)
        label_test_after_filter = np.asarray(label_test_after_filter)

        pred_2 = train_and_evaluate_NN(filtered_cross_train_data, filtered_cross_train_label,
                                       data_test_after_filter, label_test_after_filter)
        pred_2 = np.asarray(pred_2)
        pred_2[pred_2 == -1] = 0

    else:
        # ==== TF-IDF vectorization

        # fin_lab = train_labels[:total_train_num] + test_labels[
        #                                           total_train_num_1:total_train_num_1 + params.unlabel[args.tar_id]]
        label_train = train_labels[:total_train_num]
        label_test = test_labels
        # count_vect = TfidfVectorizer()
        train_test_doc = train_doc + test_doc
        data_train_test = Create_doc_term_matrix(train_test_doc)
        data_train_test = np.array(data_train_test)
        data_train = data_train_test[:total_train_num]
        data_test = data_train_test[total_train_num:]
        # print(data_test)
        # k means Clustering
        # true_k = 2
        # model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        # model.fit(data_test)
        # pred_2 = model.predict(data_test)
        # pred_2=Local_classifier_train_test(data_train, label_train, data_test)
        no_att = len(data_test[0])
        no_training_sample = len(data_train)
        # new code========
        ans = find_cosine_sim(data_train, data_test)
        # [rw, cl] = data_train.shape
        # data_train_cosine = np.asarray(np.zeros((250, cl)))
        # label_train_cosine = np.asarray(np.zeros((250)))
        count = 0
        for i in range(0, 20):
            # data_train_cosine[i][:]=data_train_1[ans[i][1]][:]
            # label_train_cosine[i][:] = label_train[ans[i][1]]
            if label_train[int(ans[i][1])] == label_test[int(i)]:
                count = count + 1
        print(count)
        # new code end =======

        label_test = np.array(label_test)
        label_test[label_test == -1] = 0


    new_lbl_pred = np.append(filtered_cross_train_label_500, pred_2)  # new_lbl for final label of 500 test statements
    new_lbl_ori = np.append(filtered_cross_train_label_500,
                            label_test_after_filter)  # new_lbl (the original one ) for comparison
    new_lbl_pred[new_lbl_pred == -1] = 0
    new_lbl_ori[new_lbl_ori == -1] = 0
    return [pred_2, label_test_after_filter, new_lbl_pred, new_lbl_ori, no_training_sample]
def perf_evalution_CM(y, y_pred):
    T1 = y_pred  # [numpy.argsort(y_pred)]
    # Loc_0 = np.where(y == 0)[0]
    # Loc_1 = np.where(y == 1)[0]
    # ii = np.random.choice(Loc_1, round(Loc_1.shape[0] / 2))
    # T1[ii] = 1
    # T1 = T1[numpy.argsort(T1)]
    # y = y[numpy.argsort(y)]
    try:
        try:
            VVV = confusion_matrix(np.asarray(y), np.asarray(T1)).ravel()
            # VVV = np.sort([TN[0], TN[1], FN[0], FN[1]])
        except:
            TN, FN, TP, FP = confusion_matrix(np.asarray(y), np.asarray(T1))
            VVV = np.sort([TN, FN, TP, FP])
        TN = VVV[0][0]
        FP = VVV[0][1]
        FN = VVV[1][0]
        TP = VVV[1][1]
        SEN = (TP) / (TP + FN)  # Recall
        SPE = (TN) / (TN + FP)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        FMS = (2 * TP) / (2 * TP + FP + FN)
        PRE = (TP) / (TP + FP)  # precision
        REC = SEN
        TS = (TP) / (TP + FP + FN)  # Threat score
        NPV = (TN) / (TN + FN)  # negative predictive value
        FOR = (FN) / (FN + TN)  # false omission rate
        MCC = matthews_corrcoef(y, T1)  # Matthews correlation coefficient
        perf = [ACC, SEN, SPE, PRE, REC, FMS]
    except:
       print("Error")
    print([ACC, SEN, SPE, PRE, REC, FMS, TS, NPV, FOR, MCC])
    return [ACC, SEN, SPE, PRE, REC, FMS, TS, NPV, FOR, MCC]

def MAIN_PERF_EVAL_save_all():
    params.print_params()
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_test', type=str2bool, default=False, help='local test verbose')
    parser.add_argument('--src_id', type=int, default=0, help='source domain id')
    parser.add_argument('--tar_id', type=int, default=3, help='target domain id')
    args = parser.parse_args()

    #seq_1 = [0, 1, 3, 2]
    seq_1 = [0, 3, 2, 1]
    # seq_4 = [3, 1, 2, 0]
    # seq_1 =  [0,1,2,3]
    Laa = seq_1  # np.random.permutation(4)
    # Laa = seq_2
    # Laa = seq_3
    # Laa = seq_4


    [pred_2, label_test, pred_test, pred_ori, no_training_sample] = main(Laa, params, args)
    label_test[label_test == -1] = 0

    [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2] = perf_evalution_CM(label_test,pred_2)  # accuracy on partial data after cross entropy
    [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1] = perf_evalution_CM(pred_ori,pred_test)  # accuracy on all 500 data
    perf_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2]
    perf_3 = [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1]
    # perf_B.append(perf_2)
    # perf_2.append(no_attributes)
    # perf_2.append(no_training_sample)
    # print(Laa)
    # print(perf_2)
    # print(perf_3)

    with open('result.csv', 'a') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
        writer_object.writerow(perf_2)
        writer_object.writerow(Laa)

        # Close the file object
        f_object.close()



MAIN_PERF_EVAL_save_all()
