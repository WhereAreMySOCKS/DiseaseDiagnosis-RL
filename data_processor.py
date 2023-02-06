import argparse
import jieba
import json

from tqdm import tqdm
from utils import *
import os
import data_utils
from knowledge_graph import KnowledgeGraph

#     因为原始数据中是中文，这里也使用中文进行判断
# Entities


HAVE_DISEASE = '疾病有'
HAVE_SYMPTOM = '症状有'
WORD = 'word'
SURGERY = '手术'
MEDICINE = '药物有'

# property
POS_EXAM = '检查结果阳性'
NO_DISEASE = '疾病无'
# Duration = '时长'
NO_SYMPTOM = '症状无'
CAUSE = '诱因'
REGURLAR = '规律'

ATTRIBUTE = [NO_DISEASE, NO_SYMPTOM, CAUSE, POS_EXAM]
ENTITIES = [HAVE_DISEASE, HAVE_SYMPTOM, SURGERY, MEDICINE]
# Relations
DISEASE_SYMPTOM = '疾病症状'
DISEASE_SURGERY = '疾病手术'
DISEASE_DRUG = '疾病药物'
RELATED_SYMPTOM = '相关症状'
RELATED_DISEASE = '相关疾病'
Relations = [DISEASE_SYMPTOM, DISEASE_SURGERY, DISEASE_DRUG, RELATED_SYMPTOM, RELATED_DISEASE]
raw_data = []


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


def text_process():
    '''
      对数据text部分进行预处理，去除停用词,载入词典。
      保存词典为dictionary/word.txt
      '''
    jieba.load_userdict('data/raw_data/user_dict.txt')
    stopwords = [line.strip() for line in open('data/raw_data/stopwords.txt', encoding='utf-8').readlines()]
    text = []
    with open('data/raw_data/dataset.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            text.append(dic['text'])

    text_clean = []
    words = [jieba.lcut(t.replace('\n', '')) for t in text]

    for wd in tqdm(words):
        tmp = []
        for w in wd:
            if w not in stopwords:
                tmp.append(w)
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
    with open('data/raw_data/text_idx.txt', 'wb') as f:
        pickle.dump(int_text, f)
    with open('data/dictionary/word.txt', 'wb') as f:
        pickle.dump(entities2int, f)


def create_attribute_dictionary():
    with open('data/raw_data/dataset.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            raw_data.append(dic)
    attr = []
    for item in raw_data:
        e_dic = item['entity_result']
        for i in e_dic:
            if i['type'] in ATTRIBUTE:
                attr.append(i['entity'])

    attr = sorted(set(attr))
    entities2int = {}
    #     词典序号从1开始
    for i in range(len(attr)):
        entities2int['%s' % attr[i]] = i + 1
    entities2int[' '] = 0
    with open('data/dictionary/attribute.txt', 'wb') as f:
        pickle.dump(entities2int, f)


def create_dictionary(e_type):
    """
             读取dataset.json文件，生成各个type的实体词典，存放在dictionary文件夹下
             save : /dictionary/
      """
    random.seed(124)
    entities = []
    with open('data/raw_data/dataset.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            e_dic = dic['entity_result']
            for i in e_dic:
                if i['type'] == e_type:
                    entities.append(i['entity'])

    entities = sorted(set(entities))
    entities2int = {}
    #     词典序号从1开始
    for i in range(len(entities)):
        entities2int['%s' % entities[i]] = i + 1
    entities2int['无记录'] = 0
    with open('data/dictionary/%s.txt' % e_type, 'wb') as f:
        pickle.dump(entities2int, f)


def create_structure_data():
    """
      读取dataset.json文件，创建结构化数据文件

      HAVE_DISEASE:
            DISEASE_SYMPTOMS --> HAVE_SYMPTOM
            DISEASE_SURGERY --> SURGERY_HISTORY
            DISEASE_DRUGS --> MEDICINE
            RELATED_DISEASE --> HAVE_DISEASE

      HAVE_SYMPTOM:
            RELATED_SYMPTOM --> HAVE_SYMPTOM
            DISEASE_SYMPTOMS --> HAVE_DISEASE

      """

    with open('data/dictionary/%s.txt' % SURGERY, 'rb') as f:
        surgery_dic = pickle.load(f)
    with open('data/dictionary/%s.txt' % HAVE_DISEASE, 'rb') as f:
        have_disease_dic = pickle.load(f)
        have_disease_dic_len = len(have_disease_dic)
    with open('data/dictionary/%s.txt' % HAVE_SYMPTOM, 'rb') as f:
        have_symptom_dic = pickle.load(f)
        have_symptom_dic_len = len(have_symptom_dic)
    with open('data/dictionary/%s.txt' % MEDICINE, 'rb') as f:
        medicine_dic = pickle.load(f)
    with open('data/dictionary/attribute.txt', 'rb') as f:
        attribute_dic = pickle.load(f)
    # with open('dictionary/word.txt', 'rb') as f:
    #       word_dic = pickle.load(f)
    #       word_dic = {v:k for k,v in word_dic.items()}
    #     此处的word_index为text_process处理后的text
    with open('data/raw_data/text_idx.txt', 'rb') as f:
        text = pickle.load(f)

    #     XX_entities 记录（head_entity,tail_entity）,并且其中有重复
    #     text中记录了每条数据中的‘主诉，现病史’
    related_disease_record, related_symptom_record = [], []
    disease_symptoms_record, disease_surgery_record, disease_drugs_record = [], [], []
    mentions_record, described_as_record = [], []
    count = 0
    #     s_d_w = [symptom,disease,word1,word2,word3.......]
    s_d_w = []
    sym2dis_record = []  # 记录每条记录中疾病和症状的对应关系
    dxy_dataset = []
    with open('data/raw_data/dataset.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            entity_result = dic['entity_result']
            relation_result = dic['relation_result']
            related_disease_entities, related_symptom_entities = [], []
            attribute_idxs, attribute_text = [], []
            for item in entity_result:
                #     提取 RELATED_DISEASE 和 RELATED_SYMPTOM
                if item['type'] == HAVE_DISEASE:
                    related_disease_entities.append(have_disease_dic.get(item['entity'], 0))
                elif item['type'] == HAVE_SYMPTOM:
                    related_symptom_entities.append(have_symptom_dic.get(item['entity'], 0))
                elif item['type'] in ATTRIBUTE:
                    attribute_idxs.append(attribute_dic.get(item['entity'], 0))
                    attribute_text.append(item['entity'])
                #    获得的是所有医疗记录中的数据，目前共2009条
            related_disease_record.append(related_disease_entities)
            related_symptom_record.append(related_symptom_entities)

            #     保存格式 （head_entity,tail_entity）
            #     虽然共2009条数据，relation == DISEASE_SYMPTOM 的数据只有817条
            temp_dict = {}
            for item in relation_result:

                if item['relation'] == DISEASE_SYMPTOM:
                    self_repo = list(set(text[count]))
                    disease = have_disease_dic.get(item['subject']['entity'], 0)
                    symptom = have_symptom_dic.get(item['object']['entity'], 0)
                    disease_symptoms_record.append((disease, symptom))

                    mentions_record.append((symptom, self_repo))
                    described_as_record.append((disease, self_repo))

                    #     single_record = [[症状有]，[疾病有],[word],[属性idx],[属性text]]
                    single_record = []
                    single_record.append([symptom])
                    single_record.append([disease])
                    single_record.append(self_repo)
                    single_record.append(list(set(attribute_idxs)))
                    single_record.append(list(set(attribute_text)))
                    s_d_w.append(single_record)
                    if not temp_dict.get(disease):
                        temp_dict.update({disease: [symptom]})
                    else:
                        temp_dict[disease].append(symptom)
                    temp_dict['word'] = self_repo
                    temp_dict['attribute_idxs'] = attribute_idxs
                    temp_dict['attribute_text'] = attribute_text
                    temp_dict['text'] = dic['text']
                if item['relation'] == DISEASE_SURGERY:
                    disease_surgery_record.append((have_disease_dic.get(item['subject']['entity'], 0),
                                                   surgery_dic.get(item['object']['entity'], 0)))

                if item['relation'] == DISEASE_DRUG:
                    disease_drugs_record.append((have_disease_dic.get(item['subject']['entity'], 0),
                                                 medicine_dic.get(item['object']['entity'], 0)))

            count += 1
            if temp_dict != {}:
                sym2dis_record.append(temp_dict)

    #     按照关系保存每条记录中的entity
    #     例如 related_disease_tail 中第i行就是have_disease[i]的所有相关疾病
    mentions_tail = create_re(mentions_record, have_symptom_dic_len)
    described_as_tail = create_re(described_as_record, have_disease_dic_len)
    related_disease_tail = create_relation(related_disease_record, have_disease_dic_len)
    related_symptom_tail = create_relation(related_symptom_record, have_symptom_dic_len)
    disease_symptoms_tail = create_re(disease_symptoms_record, have_disease_dic_len)
    disease_surgery_tail = create_re(disease_surgery_record, have_disease_dic_len)
    disease_drugs_tail = create_re(disease_drugs_record, have_disease_dic_len)
    with open('data/relation/related_disease_tail.txt', 'wb') as f:
        pickle.dump(related_disease_tail, f)
    with open('data/relation/related_symptom_tail.txt', 'wb') as f:
        pickle.dump(related_symptom_tail, f)
    with open('data/relation/disease_symptom_tail.txt', 'wb') as f:
        pickle.dump(disease_symptoms_tail, f)
    with open('data/relation/disease_surgery_tail.txt', 'wb') as f:
        pickle.dump(disease_surgery_tail, f)
    with open('data/relation/disease_drugs_tail.txt', 'wb') as f:
        pickle.dump(disease_drugs_tail, f)
    with open('data/relation/mentions_tail.txt', 'wb') as f:
        pickle.dump(mentions_tail, f)
    with open('data/relation/described_as_tail.txt', 'wb') as f:
        pickle.dump(described_as_tail, f)

    random.seed(0)
    random.shuffle(sym2dis_record)
    #     划分s_d_w作为测试集和训练集

    train_new = []
    test_new = []
    train_dict = sym2dis_record[:int(0.82 * len(sym2dis_record))]
    test_dict = sym2dis_record[int(0.82 * len(sym2dis_record)):]
    for item in train_dict:
        for dis in item:
            # int类型表明是疾病编号
            if type(dis) == int:
                for sym in item[dis]:
                    temp = []
                    temp.append([sym])
                    temp.append([dis])
                    temp.append(item['word'])
                    temp.append(item['attribute_idxs'])
                    temp.append(item['attribute_text'])
                    train_new.append(temp)

    for item in test_dict:
        for dis in item:
            # int类型表明是疾病编号
            if type(dis) == int:
                for sym in item[dis]:
                    temp = []
                    temp.append([sym])
                    temp.append([dis])
                    temp.append(item['word'])
                    temp.append(item['attribute_idxs'])
                    temp.append(item['attribute_text'])
                    test_new.append(temp)

    random.shuffle(test_new)
    random.shuffle(train_new)
    #     test_agent文件需要text,train为dict类型
    test_list = []
    for i in test_new:
        h_s = i[0][0]
        h_d = i[1][0]
        attr_idx = i[3]
        attr_text = i[4]
        test_list.append((h_s, h_d, attr_idx, attr_text))
    train_list = []
    for i in train_new:
        h_s = i[0][0]
        h_d = i[1][0]
        train_list.append((h_s, h_d))

    with open('data/tmp/Aier_Eye/train.pkl', 'wb') as f:
        pickle.dump(train_new, f)
    with open('data/tmp/Aier_Eye/test.pkl', 'wb') as f:
        pickle.dump(test_new, f)
    with open('data/tmp/Aier_Eye/test_list.pkl', 'wb') as f:
        pickle.dump(test_list, f)
    with open('data/tmp/Aier_Eye/train_list.pkl', 'wb') as f:
        pickle.dump(train_list, f)
    with open('data/my_dataset/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dict, f)
    with open('data/my_dataset/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dict, f)


#     处理 RELATED_SYMPTOM,RELATED_DISEASE
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
            tail = list(set(data[1:]))
            if head_entity in tail:
                tail.remove(head_entity)
            if tail is None:
                tail = []
        temp[head_entity] = tail
    return temp


#     处理 DISEASE_SYMPTOMS,DISEASE_SURGERY,DISEASE_DRUGS
def create_re(mylist, v_len):
    temp = [[] for i in range(v_len)]
    for i in mylist:
        head_index = i[0]
        tail_index = i[1]
        if temp[head_index] != [0]:
            if isinstance(tail_index, list):
                temp[head_index].extend(tail_index)
            else:
                temp[head_index].append(tail_index)
        else:
            if isinstance(tail_index, list):
                temp[head_index] = tail_index
            else:
                temp[head_index] = [tail_index]
    for j in range(len(temp)):
        temp[j] = list(set(temp[j]))
    return temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=Medical)
    args = parser.parse_args()
    # Create dataset
    # ========== BEGIN ========== #
    # text_process()
    # for i in ENTITIES:
    #     create_dictionary(i)
    # create_attribute_dictionary()
    # create_structure_data()

    # Create AmazonDataset instance for dataset.
    # ========== BEGIN ========== #
    print('Load', args.dataset, 'dataset from file...')
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
    dataset = data_utils.AierEyeDataset(DATASET_DIR[args.dataset])
    save_dataset(args.dataset, dataset)

    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    print('Create', args.dataset, 'knowledge graph from dataset...')
    dataset = load_dataset(args.dataset)
    kg = KnowledgeGraph(dataset, args.dataset)
    kg.compute_degrees()
    save_kg(args.dataset, kg)
    # =========== END =========== #
