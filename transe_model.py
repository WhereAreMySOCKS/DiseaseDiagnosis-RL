from __future__ import absolute_import, division, print_function

import math

import jieba
from easydict import EasyDict as edict
import torch.nn as nn
from gensim.models import KeyedVectors
from transformers import AutoModel, AutoTokenizer
from utils import *


class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, args):
        super(KnowledgeEmbedding, self).__init__()

        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        self.relu = nn.ReLU()
        if args.dataset == Aier_EYE:
            self.attributes_num = dataset.attribute.vocab_size
        self.model_type = args.model_type
        self.embedding_type = args.embedding_type

        if self.model_type == 'bert':
            self.bert = AutoModel.from_pretrained('cyclone/simcse-chinese-roberta-wwm-ext')
            self.pooler = nn.Linear(768, self.embed_size)
            self.tokenizer = AutoTokenizer.from_pretrained('cyclone/simcse-chinese-roberta-wwm-ext')
        elif self.model_type == 'atten':
            self.WQ = nn.Linear(self.embed_size, self.embed_size)
            self.WK = nn.Linear(self.embed_size, self.embed_size)
            self.WV = nn.Linear(self.embed_size, self.embed_size)
            setattr(self, 'attribute', self._entity_embedding(self.attributes_num))
        elif self.model_type == 'w2v':
            #   w2v模型输出向量维度为512
            self.pooler = nn.Linear(512, self.embed_size, bias=False).float()

            with open('data/dictionary/attribute.txt', 'rb') as f:
                attribute = list(pickle.load(f).keys())
            attribute = [attribute[-1]] + attribute[:-1]
            word_vectors_2w = KeyedVectors.load_word2vec_format('w2v/Medical.txt', binary=False)
            embedding = []
            for i in attribute:
                temp = np.zeros(512)
                seg = jieba.lcut(i)
                word_count = 0
                for j in seg:
                    try:
                        temp = temp + word_vectors_2w[j]
                        word_count += 1
                    except:
                        temp = np.zeros(512)
                if word_count == 0:
                    embedding.append(temp)
                else:
                    embedding.append(temp / word_count)
            setattr(self, 'attribute', np.array(embedding))
        elif self.model_type == 'embedding':
            embed = self._entity_embedding(self.attributes_num)
            setattr(self, 'attribute', embed)

        # Initialize entity embeddings.
        self.entities = edict(
            have_disease=edict(vocab_size=dataset.have_disease.vocab_size),
            have_symptom=edict(vocab_size=dataset.have_symptom.vocab_size),
            surgery=edict(vocab_size=dataset.surgery.vocab_size),
            medicine=edict(vocab_size=dataset.medicine.vocab_size),
            word=edict(vocab_size=dataset.word.vocab_size)
        )

        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)
        # Initialize relation embeddings and relation biases.

        self.relations = edict(
            mentions=edict(
                et='word',
                et_distrib=self._make_distrib(dataset.review.word_distrib)),
            described_as=edict(
                et='word',
                et_distrib=self._make_distrib(dataset.review.word_distrib)),
            #   这里头尾实体顺序进行了更改
            disease_symptom=edict(
                et='have_disease',
                et_distrib=self._make_distrib(dataset.review.have_disease_uniform_distrib)),
            disease_surgery=edict(
                et='surgery',
                et_distrib=self._make_distrib(dataset.disease_surgery.et_distrib)),
            disease_drug=edict(
                et='medicine',
                et_distrib=self._make_distrib(dataset.disease_drug.et_distrib)),
            related_symptom=edict(
                et='have_symptom',
                et_distrib=self._make_distrib(dataset.related_symptom.et_distrib)),
            related_disease=edict(
                et='have_disease',
                et_distrib=self._make_distrib(dataset.related_disease.et_distrib)),
        )
        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)
        if self.embedding_type == 'TransR':
            for i in self.relations:
                Mr = nn.Parameter(torch.randn(self.embed_size, self.embed_size))
                setattr(self, "Mr_" + i, Mr)
        elif self.embedding_type == 'TransH':
            #  relation_hype _p
            for r in self.relations:
                embed = self._relation_embedding()
                setattr(self, r + '_hype', embed)

        self.to(args.device)
        # elif self.embedding_type == 'DistMult':

        #   计算attention分数

    def dot_product_attention(self, my_input):
        Q = self.WQ(my_input)
        K = self.WK(my_input)
        V = self.WV(my_input)
        score = torch.softmax(torch.bmm(Q, K.transpose(1, 2)), -1) / math.sqrt(attr_num)
        re = torch.sum(torch.bmm(score, V), dim=1)
        return re

    def _get_attribute_vec(self, attributes, embed_type):
        attributes_idxs = attributes[0]
        attributes_texts = attributes[1]
        if embed_type == 'atten':
            attribute_embedding = getattr(self, 'attribute')  # nn.Embedding
            embed_input = torch.from_numpy(attributes_idxs).to(self.device)
            out = torch.relu(attribute_embedding(embed_input))
            attr_vec = self.dot_product_attention(out)
        #   bert 嵌入方法有待更改
        elif embed_type == 'bert':
            attr_vec = []
            for t in attributes_texts:
                bert_input = self.tokenizer(t, padding=True, truncation=True, return_tensors='pt').to(self.device)
                out = self.bert(bert_input['input_ids'], bert_input['attention_mask'])[1]
                out = torch.mean(self.pooler(out), dim=0)
                attr_vec.append(out)
            attr_vec = torch.stack(attr_vec)
        elif embed_type == 'none':
            attr_vec = torch.zeros((len(attributes_idxs), self.embed_size)).to(self.device)
        elif embed_type == 'embedding':
            attribute_embedding = getattr(self, 'attribute')
            embed_input = torch.tensor(attributes_idxs).to(self.device)
            attr_vec = torch.mean(attribute_embedding(embed_input), dim=1)
        elif embed_type == 'w2v':
            attribute_embedding = getattr(self, 'attribute')
            embed_input = torch.tensor(attributes_idxs)
            tensor = torch.tensor(attribute_embedding[embed_input], dtype=torch.float32).to(self.device)
            attr_vec = self.pooler(tensor)
            attr_vec = torch.mean(attr_vec, dim=1)
        return attr_vec

    def _get_attribute_len(self, attribute):
        _len = 0
        for i in attribute:
            if i != 0:
                _len += 1
        return _len

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size, self.embed_size, padding_idx=0, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size, 1)
        bias.weight = nn.Parameter(torch.zeros(vocab_size, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss(self, batch_idxs):
        """Compute knowledge graph negative sampling loss.
        batch_idxs: batch_size * 6 array, where each row is
                (u_id, p_id, w_id, b_id, c_id, rp_id).
        """
        have_symptom_idxs = batch_idxs[:, 0]
        have_disease_idxs = batch_idxs[:, 1]
        word_idxs = batch_idxs[:, 2]
        surgery_idxs = batch_idxs[:, 3]
        medicine_idxs = batch_idxs[:, 4]
        related_disease_idxs = batch_idxs[:, 5]
        relate_symptom_idxs = batch_idxs[:, 6]
        attribute_idxs = np.array(batch_idxs[:, 7].tolist())
        attribute_texts = batch_idxs[:, 8].tolist()

        regularizations = []

        # have_symptom + disease_symptom -> have_disease
        up_loss, up_embeds = self.neg_loss('have_symptom', 'disease_symptom', 'have_disease', have_symptom_idxs,
                                           have_disease_idxs, (attribute_idxs, attribute_texts))
        regularizations.extend(up_embeds)
        loss = up_loss

        # have_symptom + mentions -> word
        uw_loss, uw_embeds = self.neg_loss('have_symptom', 'mentions', 'word', have_symptom_idxs, word_idxs,
                                           (attribute_idxs, attribute_texts))
        regularizations.extend(uw_embeds)
        loss += uw_loss

        # have_disease + described_as -> word
        pw_loss, pw_embeds = self.neg_loss('have_disease', 'described_as', 'word', have_disease_idxs, word_idxs,
                                           (attribute_idxs, attribute_texts))
        regularizations.extend(pw_embeds)
        loss += pw_loss

        # have_disease + disease_surgery -> surgery
        pb_loss, pb_embeds = self.neg_loss('have_disease', 'disease_surgery', 'surgery', have_disease_idxs,
                                           surgery_idxs, (attribute_idxs, attribute_texts))

        regularizations.extend(pb_embeds)
        loss += pb_loss

        # have_disease + disease_drug -> medicine
        pc_loss, pc_embeds = self.neg_loss('have_disease', 'disease_drug', 'medicine', have_disease_idxs,
                                           medicine_idxs, (attribute_idxs, attribute_texts))

        regularizations.extend(pc_embeds)
        loss += pc_loss

        # have_disease + related_disease -> have_disease
        pr1_loss, pr1_embeds = self.neg_loss('have_disease', 'related_disease', 'have_disease', have_disease_idxs,
                                             related_disease_idxs, (attribute_idxs, attribute_texts))
        regularizations.extend(pr1_embeds)
        loss += pr1_loss

        # have_symptom + related_symptom -> have_symptom
        pr2_loss, pr2_embeds = self.neg_loss('have_symptom', 'related_symptom', 'have_symptom', have_symptom_idxs,
                                             relate_symptom_idxs, (attribute_idxs, attribute_texts))
        regularizations.extend(pr2_embeds)
        loss += pr2_loss
        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs, attributes):
        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding
        entity_tail_embedding = getattr(self, entity_tail)
        entity_head_vec = entity_head_embedding(
            torch.from_numpy(entity_head_idxs.astype(int)).to(self.device))
        entity_tail_vec = entity_tail_embedding(
            torch.from_numpy(entity_tail_idxs.astype(int)).to(self.device))
        batch_size = entity_head_vec.size(0)
        if entity_head == HAVE_SYMPTOM:
            attr_vec = self._get_attribute_vec(attributes, self.model_type)
            entity_head_vec += attr_vec
            # entity_head_vec = torch.cat((entity_head_vec, attr_vec), dim=-1)

        else:
            zeros = (np.zeros((batch_size, attr_num), dtype=int), '无记录')
            attr_vec = self._get_attribute_vec(zeros, self.model_type)
            entity_head_vec += attr_vec
            # entity_head_vec = torch.cat((entity_head_vec, attr_vec), dim=-1)

        if entity_tail == HAVE_SYMPTOM:
            attr_vec = self._get_attribute_vec(attributes, self.model_type)
            entity_tail_vec += attr_vec
            # entity_tail_vec = torch.cat((entity_tail_vec, attr_vec), dim=-1)

        else:
            zeros = (np.zeros((batch_size, attr_num), dtype=int), [[''] * attr_num])
            attr_vec = self._get_attribute_vec(zeros, self.model_type)
            entity_tail_vec += attr_vec
            # entity_tail_vec = torch.cat((entity_tail_vec, attr_vec), dim=-1)

        relation_vec = getattr(self, relation)  # [1, embed_size]
        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]
        neg_sample_idx = torch.multinomial(entity_tail_distrib, self.num_neg_samples, replacement=True).view(-1)
        zeros = (np.zeros((len(neg_sample_idx), attr_num), dtype=int), '无记录')
        neg_vec = entity_tail_embedding(neg_sample_idx) + self._get_attribute_vec(zeros, self.model_type)
        relation_bias = relation_bias_embedding(torch.from_numpy(entity_tail_idxs.astype(int)).to(self.device)).squeeze(
            1)
        #   先属性嵌入，再TranR映射
        if self.embedding_type == 'TransR':
            Mr = getattr(self, 'Mr_' + relation)
            entity_head_vec = torch.matmul(entity_head_vec, Mr)
            entity_tail_vec = torch.matmul(entity_tail_vec, Mr)
            neg_vec = torch.matmul(neg_vec, Mr)
        elif self.embedding_type == 'TransH':
            r_hype = getattr(self, relation + '_hype')
            entity_head_vec = entity_head_vec - (torch.matmul(entity_head_vec, r_hype.T) * r_hype).squeeze()
            entity_tail_vec = entity_tail_vec - (torch.matmul(entity_tail_vec, r_hype.T) * r_hype).squeeze()
            neg_vec = neg_vec - (torch.matmul(neg_vec, r_hype.T) * r_hype).squeeze()

        return self.kg_neg_loss(entity_head_vec, entity_tail_vec, relation_vec, relation_bias,
                                neg_vec)

    def kg_neg_loss(self, entity_head_vec, entity_tail_vec, relation_vec, relation_bias, neg_vec):

        example_vec = entity_head_vec + relation_vec
        example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]
        pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
        pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
        pos_loss = -pos_logits.sigmoid().log()  # [batch_size]
        neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
        neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
        neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

        loss = (pos_loss + neg_loss).mean()
        return loss, [entity_head_vec, entity_tail_vec, neg_vec]
