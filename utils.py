from __future__ import absolute_import, division, print_function

import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import torch

# ATTRIBUTE 模型 'bert','atten','none','embedding','w2v'
# MODEL_TYPE = 'none'
#     设置为True时，前两个节点限制为symptom
NODE_LIMIT = False

# Dataset names.
BEAUTY = 'beauty'
CELL = 'cell'
CLOTH = 'cloth'
CD = 'cd'
Aier_EYE = 'Aier_Eye'
Medical = 'Medical'

# Dataset directories.
DATASET_DIR = {
      Aier_EYE: './raw_data/Aier_Eye',
      Medical: 'Medical',
      CELL: './raw_data/Amazon_Cellphones',
      CLOTH: './raw_data/Amazon_Clothing',
      CD: './raw_data/Amazon_CDs',
}

# Model result directories.
TMP_DIR = {
      Aier_EYE: 'data/tmp/Aier_Eye',
      Medical: 'medical_data/tmp/medical_data',
}


LABELS = {

      Aier_EYE: (TMP_DIR[Aier_EYE] + '/train_list.pkl', TMP_DIR[Aier_EYE] + '/test_list.pkl'),
      Medical: (TMP_DIR[Medical] + '/train_list.pkl', TMP_DIR[Medical] + '/test_list.pkl')

}

# Entities
HAVE_DISEASE = 'have_disease'
HAVE_SYMPTOM = 'have_symptom'
SURGERY = 'surgery'
MEDICINE = 'medicine'
WORD = 'word'

# ATTRIBUTE
POS_EXAM = 'pos_exam'
NO_SYMPTOM = 'no_symptom'
NO_DISEASE = 'no_disease'
ATTRIBUTE = [POS_EXAM, NO_SYMPTOM, NO_DISEASE]

# ATTRIBUTE 最大个数
attr_num = 20
# Relations
MENTION = 'mentions'
DESCRIBED_AS = 'described_as'
DISEASE_SYMPTOM = 'disease_symptom'
DISEASE_SURGERY = 'disease_surgery'
DISEASE_DRUG = 'disease_drug'
RELATED_SYMPTOM = 'related_symptom'
RELATED_DISEASE = 'related_disease'
SELF_LOOP = 'self_loop'  # only for kg env

Aier_KG_RELATION  = KG_RELATION= {
      HAVE_DISEASE: {
            DISEASE_SYMPTOM: HAVE_SYMPTOM,
            DESCRIBED_AS: WORD,
            DISEASE_SURGERY: SURGERY,
            DISEASE_DRUG: MEDICINE,
            RELATED_DISEASE: HAVE_DISEASE,

      },
      WORD: {
            MENTION: HAVE_SYMPTOM,
            DESCRIBED_AS: HAVE_DISEASE,
      },
      HAVE_SYMPTOM: {
            DISEASE_SYMPTOM: HAVE_DISEASE,
            MENTION: WORD,
            RELATED_SYMPTOM: HAVE_SYMPTOM,
      },
      SURGERY: {
            DISEASE_SURGERY: HAVE_DISEASE,
      },
      MEDICINE: {
            DISEASE_DRUG: HAVE_DISEASE,
      },
}
Medical_KG_RELATION = {
      HAVE_DISEASE: {
            DISEASE_SYMPTOM: HAVE_SYMPTOM,
            DESCRIBED_AS: WORD,
            DISEASE_SURGERY: SURGERY,
            DISEASE_DRUG: MEDICINE,
            RELATED_DISEASE: HAVE_DISEASE,

      },
      WORD: {
            MENTION: HAVE_SYMPTOM,
            DESCRIBED_AS: HAVE_DISEASE,
      },
      HAVE_SYMPTOM: {
            DISEASE_SYMPTOM: HAVE_DISEASE,
            MENTION: WORD,
            RELATED_SYMPTOM: HAVE_SYMPTOM,
      },
      SURGERY: {
            DISEASE_SURGERY: HAVE_DISEASE,
      },
      MEDICINE: {
            DISEASE_DRUG: HAVE_DISEASE,
      },
}

PATH_PATTERN = {
      0: ((None, HAVE_SYMPTOM), (DISEASE_SYMPTOM, HAVE_DISEASE)),
      # length = 3
      1: ((None, HAVE_SYMPTOM), (DISEASE_SYMPTOM, HAVE_DISEASE), (RELATED_DISEASE, HAVE_DISEASE)),
      2: ((None, HAVE_SYMPTOM), (RELATED_SYMPTOM, HAVE_SYMPTOM), (DISEASE_SYMPTOM, HAVE_DISEASE)),
      # 3: ((None, HAVE_SYMPTOM), (RELATED_SYMPTOM, HAVE_SYMPTOM), (DISEASE_SYMPTOM, HAVE_DISEASE)),
      3: ((None, HAVE_SYMPTOM), (MENTION, WORD), (DESCRIBED_AS, HAVE_DISEASE)),
      # length = 4
      11: ((None, HAVE_SYMPTOM), (DISEASE_SYMPTOM, HAVE_DISEASE), (DISEASE_SURGERY, SURGERY),
           (DISEASE_SURGERY, HAVE_DISEASE)),
      12: ((None, HAVE_SYMPTOM), (DISEASE_SYMPTOM, HAVE_DISEASE), (DISEASE_DRUG, MEDICINE),
           (DISEASE_DRUG, HAVE_DISEASE)),
      13: ((None, HAVE_SYMPTOM), (DISEASE_SYMPTOM, HAVE_DISEASE), (RELATED_DISEASE, HAVE_DISEASE),
           (RELATED_DISEASE, HAVE_DISEASE)),
      14: ((None, HAVE_SYMPTOM), (RELATED_SYMPTOM, HAVE_SYMPTOM), (DISEASE_SYMPTOM, HAVE_DISEASE),
           (RELATED_DISEASE, HAVE_DISEASE)),

      15: ((None, HAVE_SYMPTOM), (DISEASE_SYMPTOM, HAVE_DISEASE), (DESCRIBED_AS, WORD),
           (DESCRIBED_AS, HAVE_DISEASE)),
      16: ((None, HAVE_SYMPTOM), (MENTION, WORD), (MENTION, HAVE_SYMPTOM),
           (DISEASE_SYMPTOM, HAVE_DISEASE)),
      # 17: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RPRODUCT), (ALSO_BOUGHT, PRODUCT)),
      # 18: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RPRODUCT), (ALSO_VIEWED, PRODUCT)),
      #  18: ((None, USER), (MENTION, WORD), (MENTION, USER), (PURCHASE, PRODUCT)),
}


def get_entities(dataset_name):
      if dataset_name == 'Medical':
            return list(Medical_KG_RELATION.keys())
      else:

            return list(Aier_KG_RELATION.keys())


def get_relations(entity_head,kg_name):
      if kg_name == "Medical":
            return list(Medical_KG_RELATION[entity_head].keys())
      else:
            return list(Aier_KG_RELATION[entity_head].keys())


def get_entity_tail(entity_head, relation,kg_name):
      if kg_name == "Medical":
            return Medical_KG_RELATION[entity_head][relation]
      else:
            return Aier_KG_RELATION[entity_head][relation]


def compute_tfidf_fast(vocab, docs):
      """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
      # (1) Compute term frequency in each doc.
      data, indices, indptr = [], [], [0]
      for d in docs:
            term_count = {}
            for term_idx in d:
                  if term_idx not in term_count:
                        term_count[term_idx] = 1
                  else:
                        term_count[term_idx] += 1
            indices.extend(term_count.keys())
            data.extend(term_count.values())
            indptr.append(len(indices))
      tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

      # (2) Compute normalized tfidf for each term/doc.
      transformer = TfidfTransformer(smooth_idf=True)
      tfidf = transformer.fit_transform(tf)
      return tfidf


def get_logger(logname):
      logger = logging.getLogger(logname)
      logger.setLevel(logging.DEBUG)
      formatter = logging.Formatter('[%(levelname)s]  %(message)s')
      ch = logging.StreamHandler(sys.stdout)
      ch.setFormatter(formatter)
      logger.addHandler(ch)
      fh = logging.handlers.RotatingFileHandler(logname, mode='w')
      fh.setFormatter(formatter)
      logger.addHandler(fh)
      return logger


def set_random_seed(seed):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save_dataset(dataset, dataset_obj):
      dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
      with open(dataset_file, 'wb') as f:
            pickle.dump(dataset_obj, f)


def load_dataset(dataset):
      dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
      dataset_obj = pickle.load(open(dataset_file, 'rb'))
      return dataset_obj


def save_labels(dataset, labels, mode='train'):
      if mode == 'train':
            label_file = LABELS[dataset][0]
      elif mode == 'test':
            label_file = LABELS[dataset][1]
      else:
            raise Exception('mode should be one of {train, test}.')
      with open(label_file, 'wb') as f:
            pickle.dump(labels, f)


def load_labels(dataset, mode='train'):
      if mode == 'train':
            label_file = LABELS[dataset][0]
      elif mode == 'test':
            label_file = LABELS[dataset][1]
      else:
            raise Exception('mode should be one of {train, test}.')
      user_products = pickle.load(open(label_file, 'rb'))
      return user_products


def save_embed(dataset, embed):
      embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
      pickle.dump(embed, open(embed_file, 'wb'))


def load_embed(dataset):
      embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
      embed = pickle.load(open(embed_file, 'rb'))
      return embed


def save_kg(dataset, kg):
      kg_file = TMP_DIR[dataset] + '/kg.pkl'
      pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(dataset):
      kg_file = TMP_DIR[dataset] + '/kg.pkl'
      kg = pickle.load(open(kg_file, 'rb'))
      return kg


def load_dict(dict_name):
      with open('dictionary/' + dict_name, 'rb') as f:
            my_dict = pickle.load(f)
            x = dict([val, key] for key, val in my_dict.items())
            return x
