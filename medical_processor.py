import codecs
import json
import pickle
import random

import jieba
import numpy as np
import pandas as pd
import torch
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, top_k_accuracy_score
from sklearn.svm._libsvm import predict_proba
from xgboost import XGBClassifier
import warnings
from utils import compute_tfidf_fast

warnings.filterwarnings('ignore')


def tf_idf(vocab, reviews, word_tfidf_threshold):
    # (1) Filter words by both tfidf and frequency.
    review_tfidf = compute_tfidf_fast(vocab, reviews)
    all_removed_words = []
    all_remained_words = []
    for rid, data in enumerate(reviews):
        doc_tfidf = review_tfidf[rid].toarray()[0]
        remained_words = [wid for wid in set(data) if doc_tfidf[wid] >= word_tfidf_threshold]

        removed_words = set(data).difference(remained_words)  # only for visualize
        removed_words = [vocab[wid] for wid in removed_words]
        _remained_words = [vocab[wid] for wid in remained_words]
        all_removed_words.append(removed_words)
        all_remained_words.append(_remained_words)
    return all_remained_words


def textprocess(data):
    jieba.load_userdict('data/raw_data/user_dict.txt')
    stopwords = [line.strip() for line in open('data/raw_data/stopwords.txt', encoding='utf-8').readlines()]
    text_clean = []
    for item in data:
        cut = jieba.lcut(item['desc'])
        tmp = []
        for wd in cut:
            if wd not in stopwords:
                tmp.append(wd)
        text_clean.append(tmp)
    uni_words = sorted(set([j for i in text_clean for j in i]))

    entities2int = {}
    #     词序号从1开始
    for i in range(len(uni_words)):
        entities2int['%s' % uni_words[i]] = i + 1
    entities2int['无记录'] = 0
    int_text = []
    for i in text_clean:
        tmp = []
        for j in i:
            tmp.append(entities2int[j])
        int_text.append(tmp)

    remain_word = tf_idf(list(entities2int.keys()), int_text, 0.15)
    int_text = []
    for i in remain_word:
        tmp = []
        for j in i:
            tmp.append(entities2int[j])
        int_text.append(tmp)
    with open('medical_data/dictionary/text_idx.txt', 'wb') as f:
        pickle.dump(int_text, f)
    with open('medical_data/dictionary/word.txt', 'wb') as f:
        pickle.dump(entities2int, f)
    return int_text


def read_file(path):
    txt = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines()[:800]:
            txt.append(json.loads(line))
    return txt


def create_dictionary(entity, name):
    entity = sorted(set(entity))
    entity2int = {}
    #     词典序号从1开始
    for i in range(len(entity)):
        entity2int['%s' % entity[i]] = i + 1
    entity2int['无记录'] = 0
    with open('medical_data/dictionary/%s.txt' % name, 'wb') as f:
        pickle.dump(entity2int, f)
    return entity2int


def create_relation(relation_entity_list, voc_len):
    temp = [[] for i in range(voc_len)]
    for data in relation_entity_list:
        if len(data) == 0:
            continue
        elif len(data) == 1:
            head_entity = data[0]
            tail = []
        else:
            head_entity = data[0]
            tail = list(set(data[1]))
            if head_entity in tail:
                tail.remove(head_entity)
            if tail is None:
                tail = []
        temp[head_entity] = tail
    return temp


def create_dataset(data, int_text, text):
    random.seed(1)
    virtual_data = []
    #   确保每个训练集中包含所有疾病
    addition_data = []
    for i in range(len(data)):
        for j in range(2):
            temp = []
            symptom = data[i][0]
            disease = data[i][1]
            #   一条记录采样两次，一次0.5, 0.7
            sample_symptom = random.sample(symptom, int((0.5 + 0.2 * j) * len(symptom)))
            if j == 1:
                temp_data = []
                temp_data.append(random.sample(symptom, int(0.6 * len(symptom))))
                temp_data.append([disease])
                temp_data.append(int_text[i])
                temp_data.append([0])
                temp_data.append([0])
                temp_data.append(text[i])
                addition_data.append(temp)
            temp.append(sample_symptom)
            temp.append([disease])
            temp.append(int_text[i])
            temp.append([0])
            temp.append([0])
            temp.append(text[i])
            virtual_data.append(temp)

    random.shuffle(virtual_data)
    alpha = 0.6
    train_data = virtual_data[:int(alpha * len(virtual_data))] + addition_data
    test = virtual_data[int(alpha * len(virtual_data)):]

    return train_data, test


def data_squeeze(input_list):
    output_list = []
    des = []
    me = []
    for i in range(len(input_list)):
        # describe
        if (input_list[i][1], input_list[i][2]) not in des:
            des.append((input_list[i][1][0], input_list[i][2]))
        for s in input_list[i][0]:
            temp = []
            # mention
            me.append((s, input_list[i][2]))

            temp.append([s])
            temp.append(input_list[i][1])
            temp.append(input_list[i][2])
            temp.append([0])
            temp.append([0])
            output_list.append(temp)

    return output_list, me, des


def get_dict_data(test_data):
    test_dic = []
    for i in test_data:
        disease = i[1][0]
        sym = i[0]
        dic = {}
        dic.update({disease: sym})
        dic.update({'word': i[2]})
        dic.update({'attribute_idxs': i[3]})
        dic.update({'attribute_text': i[4]})
        dic.update({'text': i[5]})
        test_dic.append(dic)
    return test_dic


def norm(array):
    array = np.array(array)
    max = np.max(array)
    min = np.min(array)
    array = (array - min) / (max - min)
    return array


def ML(flag, model_name):
    if flag == 1:
        with open('medical_data/tmp/medical_data/train_dataset.pkl', 'rb') as f:
            train_set = pickle.load(f)
        with open('medical_data/tmp/medical_data/test_dataset.pkl', 'rb') as f:
            test_set = pickle.load(f)
    else:

        with open('data/my_dataset/train_dataset.pkl', 'rb') as f:
            train_set = pickle.load(f)
        with open('data/my_dataset/test_dataset.pkl', 'rb') as f:
            test_set = pickle.load(f)
    tran_x = []
    tran_y = []
    test_x = []
    test_y = []
    for item in train_set:
        for i in item:
            if type(i) == int:
                tran_y.append(i)
                tran_x.append(item[i])
    y = sorted(list(set(tran_y)))
    y_dict = {}
    for i in range(len(y)):
        y_dict[y[i]] = i
    for item in test_set:
        for i in item:
            if type(i) == int:
                if i in y:
                    test_y.append(i)
                    test_x.append(item[i])
    for i in range(len(tran_y)):
        tran_y[i] = y_dict[tran_y[i]]
    for j in range(len(test_y)):
        test_y[j] = y_dict[test_y[j]]
    print(len(test_set) - len(test_y))

    for i in range(len(tran_x)):
        tran_x[i] = np.pad(tran_x[i], (0, 10 - len(tran_x[i])), 'constant', constant_values=(0, 0))
    for i in range(len(test_x)):
        test_x[i] = np.pad(test_x[i], (0, 10 - len(test_x[i])), 'constant', constant_values=(0, 0))
    tran_x = norm(tran_x)
    test_x = norm(test_x)
    tran_y = np.array(tran_y)
    test_y = np.array(test_y)
    if model_name == 'svm':
        model = svm.SVC(kernel='poly', decision_function_shape='ovr', probability=True)

    else:
        model = XGBClassifier(learning_rate=0.2,
                              n_estimators=60,  # 树的个数-10棵树建立xgboost
                              max_depth=20,  # 树的深度
                              min_child_weight=1,  # 叶子节点最小权重
                              gamma=0.,  # 惩罚项中叶子结点个数前的参数
                              subsample=1,  # 所有样本建立决策树
                              colsample_btree=1,  # 所有特征建立决策树
                              scale_pos_weight=1,  # 解决样本个数不平衡的问题
                              random_state=0,  # 随机数
                              slient=0,
                              predictor='cpu_predictor'
                              )

    model.fit(tran_x, tran_y)
    result = model.predict(test_x)
    probs = model.predict_proba(test_x)

    f1 = f1_score(test_y, result, average='macro')
    test_acc = accuracy_score(test_y, result)
    pre = precision_score(test_y, result, average='macro')
    re = recall_score(test_y, result, average='macro')

    hit3 = top_k_accuracy_score(test_y, probs, k=3, labels=range(0, len(y_dict)))
    hit5 = top_k_accuracy_score(test_y, probs, k=5, labels=range(0, len(y_dict)))

    return np.array([test_acc, hit3, hit5, f1, pre, re])


def get_disease_smp(input_data):
    with open('medical_data/dictionary/症状有.txt', 'rb') as f:
        data = pickle.load(f)
        sym_dic = {v: k for k, v in data.items()}
    with open('medical_data/dictionary/疾病有.txt', 'rb') as f:
        data = pickle.load(f)
        dis_dic = {v: k for k, v in data.items()}

    dic = {}
    for i in input_data:
        temp_dic = {}
        dis_index = i[1][0]
        dis = dis_dic[dis_index]
        if not dic.get(dis):
            dic.update({dis: {'index': dis_index, 'Symptom': {}}})
            for j in i[0]:
                temp_dic.update({sym_dic[j]: j})
            dic[dis]['Symptom'] = temp_dic
        else:
            for j in i[0]:
                _dic = dic[dis]['Symptom']
                _dic[sym_dic[j]] = j
                dic[dis]['Symptom'] = _dic

    with open('medical_data/disease_symptom.p', 'wb') as f:
        pickle.dump(dic, f)


def std(result):
    mean = np.mean(result, axis=0)
    # var = np.var(result, axis=0)
    std = np.std(result, axis=0)
    all_ = ['top-1', 'top-3', 'top-5', 'f1', 'recall', 'precision', ]
    print("*******************--result--********************")

    for i in range(len(mean)):
        print("{}:  ${}\\pm{}$".format(all_[i], round(mean[i], 4), round(std[i], 4)))


def processor():
    data = read_file('medical_data/medical_data.json')
    int_text = textprocess(data)
    text = [i['desc'] for i in data]
    #   entity
    disease = []
    surgery = []
    drug = []
    symptom = []
    word = []
    #   relation
    disease_symptom = []
    disease_surgery = []
    disease_drugs = []
    related_disease = []
    related_symtom = []

    # 提取实体和关系
    for item in data:
        disease.append(item['name'])
        symptom.extend(item['symptom'])
        try:
            surgery.extend(item['cure_way'])
        except KeyError:
            pass
        drug.extend(item['recommand_drug'])
    disease2int = create_dictionary(disease, '疾病有')
    symptom2int = create_dictionary(symptom, '症状有')
    drug2int = create_dictionary(drug, '药物有')
    surgery2int = create_dictionary(surgery, '手术')
    my_data = []
    for item in data:
        disease = disease2int[item['name']]
        symptom = [symptom2int[i] for i in item['symptom']]
        temp = []
        temp.append(symptom)
        temp.append(disease)
        my_data.append(temp)
        try:
            surgery = [surgery2int[i] for i in item['cure_way']]
        except KeyError:
            pass
        drug = [drug2int[i] for i in item['recommand_drug']]
        try:
            accompany = [disease2int[i] for i in item['acompany']]
        except KeyError:
            accompany = []
        disease_symptom.append((disease, symptom))
        disease_surgery.append((disease, surgery))
        related_disease.append((disease, accompany))
        disease_drugs.append((disease, drug))
        for i in symptom:
            j = list(set(symptom) - set([i]))
            related_symtom.append((i, j))
    disease_symptom_tail = create_relation(disease_symptom, len(disease2int))
    disease_surgery_tail = create_relation(disease_surgery, len(disease2int))
    related_disease_tail = create_relation(related_disease, len(disease2int))
    disease_drugs_tail = create_relation(disease_drugs, len(disease2int))
    related_symptom_tail = create_relation(disease_drugs, len(symptom2int))

    with open('medical_data/relation/disease_symptom_tail.txt', 'wb') as f:
        pickle.dump(disease_symptom_tail, f)
    with open('medical_data/relation/disease_surgery_tail.txt', 'wb') as f:
        pickle.dump(disease_surgery_tail, f)
    with open('medical_data/relation/related_disease_tail.txt', 'wb') as f:
        pickle.dump(related_disease_tail, f)
    with open('medical_data/relation/disease_drugs_tail.txt', 'wb') as f:
        pickle.dump(disease_drugs_tail, f)
    with open('medical_data/relation/related_symptom_tail.txt', 'wb') as f:
        pickle.dump(related_symptom_tail, f)

    alpha = 0.8
    train_data, test_data = create_dataset(my_data, int_text, text)

    #   my_train是经过展开后的数据，为构造三元组使用。每个症状对应一个疾病
    my_train, mention, describe_as = data_squeeze(train_data)
    my_test, _, _ = data_squeeze(test_data)

    mention_tail = create_relation(mention, len(symptom2int))
    describe_as_tail = create_relation(describe_as, len(disease2int))
    with open('medical_data/relation/mention_tail_tail.txt', 'wb') as f:
        pickle.dump(mention_tail, f)
    with open('medical_data/relation/describe_as_tail_tail.txt', 'wb') as f:
        pickle.dump(describe_as_tail, f)
    with open('medical_data/tmp/medical_data/train.pkl', 'wb') as f:
        pickle.dump(my_train, f)
    with open('medical_data/tmp/medical_data/test.pkl', 'wb') as f:
        pickle.dump(my_test, f)
    with open('medical_data/tmp/medical_data/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    #   dic以字典形式保存数据，{disease:[symptom1,symptom2...],...}
    #   test 和 train 无交集， test中可能会有train中没有出现过的数据
    test_dic = get_dict_data(test_data)
    train_dic = get_dict_data(train_data)
    with open('medical_data/tmp/medical_data/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dic, f)
    with open('medical_data/tmp/medical_data/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dic, f)

    get_disease_smp(test_data + train_data)
    result1 = []
    result2 = []
    print('svm....')
    # for i in range(5):
    #     result1.append(svm_baseline())
    # result1 = np.array(result1)
    # std(result1)


if __name__ == '__main__':

    processor()
    result1 = []
    result2 = []
    for i in range(1):
        result2.append(ML(1, 'svm1'))
    result = np.array(result2)
    std(result2)
