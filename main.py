import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn.svm as svm
import gensim
import torch
import pickle
import torch.utils.data as Data
from gensim.models import KeyedVectors
from sklearn.metrics import precision_score, f1_score, roc_auc_score

HAVE_DISEASE = '疾病有'
NO_DISEASE = '疾病无'
HAVE_SYMPTOM = '症状有'
NO_SYMPTOM = '症状无'
# SURGERY_HISTORY = '手术史有'
SURGERY = '手术'
MEDICINE = '药物有'
POS_EXAM = '检查结果阳性'
NEG_EXAM = '检查结果阴性'
WORD = 'word'

Entities = [HAVE_DISEASE, HAVE_SYMPTOM, SURGERY, MEDICINE, WORD]
# Relations
DISEASE_SYMPTOM = '疾病症状'
DISEASE_SURGERY = '疾病手术'
DISEASE_DRUG = '疾病药物'
RELATED_SYMPTOM = '相关症状'
RELATED_DISEASE = '相关疾病'
Relations = [DISEASE_SYMPTOM, DISEASE_SURGERY, DISEASE_DRUG, RELATED_SYMPTOM, RELATED_DISEASE]


class Net(nn.Module):
    def __init__(self, input_size, class_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, class_size + 1)
        self.softmax = nn.Softmax()

    def forward(self, X):
        out = self.fc1(X)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


def get_all_labels():
    with open('dictionary/%s.txt' % SURGERY, 'rb') as f:
        surgery_dic = pickle.load(f)

    with open('dictionary/%s.txt' % HAVE_DISEASE, 'rb') as f:
        have_disease_dic = pickle.load(f)
    with open('dictionary/%s.txt' % HAVE_SYMPTOM, 'rb') as f:
        have_symptom_dic = pickle.load(f)

    with open('dictionary/%s.txt' % MEDICINE, 'rb') as f:
        medicine_dic = pickle.load(f)

    with open('data/raw_data/text_idx.txt', 'rb') as f:
        text = pickle.load(f)

    print('*-*-* 疾病 -*-*-*' * 10)
    for i in have_disease_dic.keys():
        print(i)

    print('*-*-* 手术 -*-*-*' * 10)
    for i in surgery_dic.keys():
        print(i)

    print('*-*-* 症状 -*-*-*' * 10)
    for i in have_symptom_dic.keys():
        print(i)
    print('*-*-* 药物 -*-*-*' * 10)
    for i in medicine_dic.keys():
        print(i)


def show_relation():
    raw_data = []
    relation_list = []
    re_dict = {}
    with open('data/raw_data/dataset.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            raw_data.append(dic)
            relation_result = dic['relation_result']
            temp = []
            for i in relation_result:
                if i['relation'] == '疾病症状':
                    disease = i['subject']['entity']
                    symptom = i['object']['entity']
                    disease_symptoms = re_dict.get(disease, 0)
                    if disease_symptoms != 0:
                        disease_symptoms.append(symptom)
                    else:
                        re_dict.update({disease: [symptom]})
    max_len = 0
    for i in re_dict:
        re_dict[i] = list(set(re_dict[i]))
        if len(re_dict[i]) > max_len:
            max_len = len(re_dict[i])
    counter = [0 for i in range(max_len)]
    for j in re_dict:
        counter[len(re_dict[j]) - 1] += 1
    print(counter)
    X = list(range(1, len(counter) + 1))
    Y = counter
    fig = plt.figure()
    plt.bar(X, Y, 0.5, color="green")
    plt.xlabel("number of symptoms")
    plt.ylabel("number of diseases")
    plt.title("Statistical chart of the number of symptoms required to diagnose a disease")

    plt.show()
    plt.savefig("barChart.jpg")


def load_dataset():
    with open('data/tmp/Aier_Eye/train.pkl', 'rb') as f:
        train = pickle.load(f)
        print(train)


def svm_baseline():
    with open('data/tmp/Aier_Eye/train.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('data/tmp/Aier_Eye/test.pkl', 'rb') as f:
        test_set = pickle.load(f)
    tran_x = []
    tran_y = []
    test_x = []
    test_y = []
    for i in train_set:
        tran_x.append(i[0][0])
        tran_y.append(i[1][0])
    for i in test_set:
        test_x.append(i[0][0])
        test_y.append(i[1][0])
    model = svm.SVC(kernel="linear", decision_function_shape="ovo", probability=True)
    train_acc = model.fit(np.array(tran_x).reshape(-1, 1), np.array(tran_y).reshape(-1, 1))
    result = model.predict(np.array(test_x).reshape(-1, 1))
    r3 = model.predict_proba(np.array(test_x).reshape(-1, 1))
    x = np.array(test_y).reshape(-1, 1)
    t = f1_score(test_y, result, average='macro')
    q = roc_auc_score(x, r3, multi_class='ovo')
    # test_acc = model.score(np.array(test_x).reshape(-1, 1),np.array(test_y).reshape(-1, 1) )
    print(1)


def get_len(data):
    count = 0
    for i in data:
        count += len(i)
    return count


def dataset_counter():
    word = pickle.load(open('medical_data/dictionary/word.txt', 'rb'))
    surgery = pickle.load(open('medical_data/dictionary/手术.txt', 'rb'))
    disease = pickle.load(open('medical_data/dictionary/疾病有.txt', 'rb'))
    symptom = pickle.load(open('medical_data/dictionary/症状有.txt', 'rb'))
    durg = pickle.load(open('medical_data/dictionary/药物有.txt', 'rb'))

    described_as_tail = pickle.load(open('medical_data/relation/describe_as_tail_tail.txt', 'rb'))
    disease_drugs_tail = pickle.load(open('medical_data/relation/disease_drugs_tail.txt', 'rb'))
    disease_surgery_tail = pickle.load(open('medical_data/relation/disease_surgery_tail.txt', 'rb'))
    disease_symptom_tail = pickle.load(open('medical_data/relation/disease_symptom_tail.txt', 'rb'))
    mentions_tail = pickle.load(open('medical_data/relation/mention_tail_tail.txt', 'rb'))
    related_disease_tail = pickle.load(open('medical_data/relation/related_disease_tail.txt', 'rb'))
    related_symptom_tail = pickle.load(open('medical_data/relation/related_symptom_tail.txt', 'rb'))

    # print(len(word), len(surgery), len(disease), len(symptom), len(durg))
    print(get_len(disease_symptom_tail), get_len(disease_surgery_tail), get_len(disease_drugs_tail),
          get_len(related_symptom_tail) / 2,
          get_len(related_disease_tail), get_len(mentions_tail), get_len(described_as_tail))


if __name__ == '__main__':
    dataset_counter()
