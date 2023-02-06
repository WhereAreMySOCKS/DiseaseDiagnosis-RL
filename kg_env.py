from __future__ import absolute_import, division, print_function

import math
import numpy as np
import torch

from utils import *


class KGState(object):
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len  # mode: one of {full, current}
        if history_len == 0:
            self.dim = 2 * embed_size
        elif history_len == 1:
            self.dim = 4 * embed_size
        elif history_len == 2:
            self.dim = 6 * embed_size
        else:
            raise Exception('history length should be one of {0, 1, 2}')

    def __call__(self, user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                 older_relation_embed):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
        elif self.history_len == 2:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                                   older_relation_embed])
        else:
            raise Exception('mode should be one of {full, current}')


class BatchKGEnvironment(object):
    def __init__(self, args):
        self.max_acts = args.max_acts
        self.act_dim = args.max_acts + 1  # Add self-loop action, whose act_idx is always 0.
        self.max_num_nodes = args.max_path_len + 1  # max number of hops (= #nodes - 1)
        self.kg = load_kg(args.dataset)
        self.embeds = load_embed(args.dataset)
        self.embed_size = self.embeds[HAVE_SYMPTOM].shape[1]
        self.embeds[SELF_LOOP] = (np.zeros(self.embed_size), 0.0)
        self.state_gen = KGState(self.embed_size, history_len=args.state_history)
        self.state_dim = self.state_gen.dim
        self.embedding_type = args.embedding_type
        self.model_type = args.model_type
        self.path_limit = args.path_limit
        if self.embedding_type == 'TransR':
            self.Mr_disease_symptom = self.embeds.get('Mr_disease_symptom')
            self.Mr_mentions = self.embeds.get('Mr_mentions')
            self.Mr_described_as = self.embeds.get('Mr_described_as')
            self.Mr_disease_surgery = self.embeds.get('Mr_disease_surgery')
            self.Mr_disease_drug = self.embeds.get('Mr_disease_drug')
            self.Mr_related_disease = self.embeds.get('Mr_related_disease')
            self.Mr_related_symptom = self.embeds.get('Mr_related_symptom')

        # Compute path patterns
        self.patterns = []
        for pattern_id in [0, 1, 2, 3, 11, 12, 13, 14, 15, 16]:
            #   for pattern_id in [0, 1, 2, 11, 12, 13, 14]:
            pattern = PATH_PATTERN[pattern_id]
            pattern = [SELF_LOOP] + [v[0] for v in pattern[1:]]  # pattern contains all relations
            # if pattern_id == 1:
            #       pattern.append(SELF_LOOP)
            self.patterns.append(tuple(pattern))

        # Following is current episode information.
        self._batch_path = None  # list of tuples of (relation, node_type, node_id)
        self._batch_curr_actions = None  # save current valid actions
        self._batch_curr_state = None
        self._batch_curr_reward = None
        # Here only use 1 'done' indicator, since all paths have same length and will finish at the same time.
        self._done = False

    def _has_pattern(self, path):
        pattern = tuple([v[0] for v in path])
        return pattern in self.patterns

    def _batch_has_pattern(self, batch_path):
        return [self._has_pattern(path) for path in batch_path]

    def _get_actions(self, path, done):
        """Compute actions for current node."""
        path_length = len(path)
        curr_node_type, curr_node_id = path[-1][1], path[-1][2]
        actions = [(SELF_LOOP, curr_node_id)]  # self-loop must be included.

        # (1) If game is finished, only return self-loop action.
        if done:
            return actions

        # (2) Get all possible edges from original knowledge graph.
        # [CAVEAT] Must remove visited nodes!
        relations_nodes = self.kg(curr_node_type, curr_node_id)
        candidate_acts = []  # list of tuples of (relation, node_type, node_id)'
        visited_nodes = []
        for v in path:
            if self.path_limit == 0:
                visited_nodes.append((v[1], v[2]))
            else:
                if v[1] != HAVE_DISEASE:
                    visited_nodes.append((v[1], v[2]))
        visited_nodes = set(visited_nodes)

        for r in relations_nodes:
            if self.path_limit != 0:
                #  包括起点在内，前两个节点不生成疾病
                if len(path) <= 2 and r == DISEASE_SYMPTOM:
                    continue
            next_node_type = KG_RELATION[curr_node_type][r]
            next_node_ids = relations_nodes[r]
            next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]  # filter
            candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

        # (3) If candidate action set is empty, only return self-loop action.
        if len(candidate_acts) == 0:
            return actions

        # (4) If number of available actions is smaller than max_acts, return action sets.
        if len(candidate_acts) <= self.max_acts:
            candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            return actions

        # (5) If there are too many actions, do some deterministic trimming here!
        # user_embed = self.embeds[HAVE_SYMPTOM][path[0][2]]
        user_attr_embed = self._get_user_embed(path)
        scores = []
        for r, next_node_id in candidate_acts:
            next_node_type = KG_RELATION[curr_node_type][r]
            tail_embed = self.embeds[next_node_type][next_node_id]
            if next_node_type in [HAVE_SYMPTOM, HAVE_DISEASE, WORD]:
                relation_embed = self.embeds[r][0]
                if self.embedding_type == 'TransR':
                    Mr_I = np.linalg.inv(self.embeds.get('Mr_' + r))
                    relation_embed = np.matmul(relation_embed, Mr_I)
                elif self.embedding_type == 'TransH':
                    r_hype = self.embeds.get(r + '_hype')
                    user_attr_embed = user_attr_embed - (np.matmul(user_attr_embed, r_hype.T) * r_hype).squeeze()

                src_embed = user_attr_embed + relation_embed

            else:  # 症状 + 属性 + 症状疾病 = 疾病
                if self.embedding_type == 'TransR':
                    relation_embed = np.matmul(self.embeds[DISEASE_SYMPTOM][0],
                                               np.linalg.inv(self.embeds.get('Mr_disease_symptom')))
                    Mr_I = np.linalg.inv(self.embeds.get('Mr_' + r))
                    src_embed = user_attr_embed + relation_embed + np.matmul(self.embeds[r][0], Mr_I)

                elif self.embedding_type == 'TransH':
                    r_hype = self.embeds.get('disease_symptom_hype').squeeze()
                    user_attr_embed = self.embeds['disease_symptom'][0] + user_attr_embed - (
                            np.matmul(user_attr_embed, r_hype.T) * r_hype) \
                                      + (np.matmul(user_attr_embed, r_hype.T) * r_hype).squeeze()
                    r_hype = self.embeds.get(r + '_hype')
                    user_attr_embed = user_attr_embed - (np.matmul(user_attr_embed, r_hype.T) * r_hype)
                    src_embed = user_attr_embed + self.embeds[r][0]

                else:
                    relation_embed = self.embeds[DISEASE_SYMPTOM][0]
                    src_embed = user_attr_embed + relation_embed + self.embeds[r][0]
            if self.embedding_type == 'TransH':
                tail_embed = tail_embed - (np.matmul(tail_embed, r_hype.T) * r_hype).squeeze()
            score = np.matmul(src_embed, tail_embed)
            # This trimming may filter out target products!
            # Manually set the score of target products a very large number.
            # if next_node_type == PRODUCT and next_node_id in self._target_pids:
            #    score = 99999.0
            scores.append(score)

        candidate_idxs = np.argsort(scores)[-self.max_acts:]  # choose actions with larger scores
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions

    def _batch_get_actions(self, batch_path, done):
        return [self._get_actions(path, done) for path in batch_path]

    #   将属性嵌入至头实体embed中
    def _get_user_embed(self, path, uid=None):
        attr_vec = torch.zeros(self.embed_size)
        if uid:
            _user_embed = self.embeds[HAVE_SYMPTOM][uid].copy()
        else:
            _user_embed = self.embeds[HAVE_SYMPTOM][path[0][2]].copy()
        attribute_idxs = path[0][3]
        attribute_idxs.extend([0] * (attr_num - len(attribute_idxs)))
        attribute_text = path[0][4]
        if attribute_idxs and self.model_type == 'atten':
            attribute_embed = torch.relu(torch.from_numpy(self.embeds['attribute'][attribute_idxs]))
            WK_weight = torch.from_numpy(self.embeds['WK.weight'])
            WQ_weight = torch.from_numpy(self.embeds['WQ.weight'])
            WV_weight = torch.from_numpy(self.embeds['WV.weight'])
            WK_bias = torch.from_numpy(self.embeds['WK.bias'])
            WQ_bias = torch.from_numpy(self.embeds['WQ.bias'])
            WV_bias = torch.from_numpy(self.embeds['WV.bias'])
            K = torch.matmul(attribute_embed, WK_weight) + WK_bias
            Q = torch.matmul(attribute_embed, WQ_weight) + WQ_bias
            V = torch.matmul(attribute_embed, WV_weight) + WV_bias
            score = torch.softmax(torch.matmul(Q, K.transpose(0, 1)), -1) / math.sqrt(attr_num)
            attr_vec = torch.sum(torch.matmul(score, V), dim=0).detach().numpy()
        elif attribute_idxs and self.model_type == 'bert':
            bert_attr_vec = torch.from_numpy(self.embeds['bert_attr_vec'][attribute_idxs])
            attr_vec = torch.sum(bert_attr_vec, dim=0) / attr_num
            attr_vec = attr_vec.detach().numpy()
        elif self.model_type == 'none':
            attr_vec = attr_vec.numpy()
        elif self.model_type == 'embedding' or self.model_type == 'w2v':
            attribute_embed = torch.from_numpy(self.embeds['attribute'][attribute_idxs])
            attr_vec = torch.mean(attribute_embed, dim=0)
            attr_vec = attr_vec.detach().numpy()

        _user_embed += attr_vec
        return _user_embed

    def _get_state(self, path):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""
        attribute_idxs = path[0][3]
        attribute_text = path[0][4]
        attribute_idxs.extend([0] * (attr_num - len(attribute_idxs)))

        user_embed = self._get_user_embed(path)
        # user_embed
        zero_embed = np.zeros(self.embed_size)
        if len(path) == 1:  # initial state
            state = self.state_gen(user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed)
            return state

        older_relation, last_node_type, last_node_id = path[-2][0], path[-2][1], path[-2][2]
        last_relation, curr_node_type, curr_node_id = path[-1][0], path[-1][1], path[-1][2]
        if curr_node_type == HAVE_SYMPTOM:
            curr_node_embed = self._get_user_embed(path, curr_node_id)
        else:
            curr_node_embed = self.embeds[curr_node_type][curr_node_id]
        if last_node_type == HAVE_SYMPTOM:
            last_node_embed = self._get_user_embed(path, last_node_id)
        else:
            last_node_embed = self.embeds[last_node_type][last_node_id]
        last_relation_embed, _ = self.embeds[last_relation]  # this can be self-loop!
        if len(path) == 2:
            state = self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, zero_embed,
                                   zero_embed)
            return state

        older_node_type, older_node_id = path[-3][1], path[-3][2]
        if older_node_type == HAVE_SYMPTOM:
            older_node_embed = self._get_user_embed(path, older_node_id)
        else:
            older_node_embed = self.embeds[older_node_type][older_node_id]
        older_relation_embed, _ = self.embeds[older_relation]
        state = self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed,
                               older_relation_embed)
        return state

    def _batch_get_state(self, batch_path, path_attribute_ids=None, path_attribute_text=None):
        batch_state = [self._get_state(path) for path in batch_path]
        return np.vstack(batch_state)  # [bs, dim]

    def _get_reward(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        if len(path) <= 2:
            return 0.0

        if not self._has_pattern(path):
            return 0.0

        target_score = 0.0
        curr_node_type, curr_node_id = path[-1][1], path[-1][2]
        if curr_node_type == HAVE_DISEASE:
            # Give soft reward for other reached products.
            u_vec = self._get_user_embed(path)
            p_vec = self.embeds[HAVE_DISEASE][curr_node_id]
            if self.embedding_type == 'TransR':
                Mr = self.Mr_disease_symptom
                u_vec = np.matmul(u_vec, Mr)
                p_vec = np.matmul(p_vec, Mr)

            elif self.embedding_type == 'TransH':
                r_hype = self.embeds['disease_symptom_hype']
                u_vec = u_vec - (np.matmul(u_vec, r_hype.T) * r_hype).squeeze()
                p_vec = p_vec - (np.matmul(p_vec, r_hype.T) * r_hype).squeeze()
            u_vec += self.embeds[DISEASE_SYMPTOM][0]
            num = float(np.dot(u_vec, p_vec))
            denom = np.linalg.norm(u_vec) * np.linalg.norm(p_vec)
            target_score = 0.5 + 0.5 * (num / denom) if denom != 0 else 0
        return target_score

    def _batch_get_reward(self, batch_path):
        batch_reward = [self._get_reward(path) for path in batch_path]
        return np.array(batch_reward)

    def _is_done(self):
        """Episode ends only if max path length is reached."""
        return self._done or len(self._batch_path[0]) >= self.max_num_nodes

    def reset(self, path_attribute_ids, path_attribute_text, uids=None):
        if uids is None:
            all_uids = list(self.kg(HAVE_SYMPTOM).keys())
            uids = [random.choice(all_uids)]

        # # each element is a tuple of (relation, entity_type, entity_id)
        # for uid, attri_id, attri_text in zip(uids, path_attribute_ids, path_attribute_text):
        #     self._batch_path.append()
        self._batch_path = [[(SELF_LOOP, HAVE_SYMPTOM, uid, attri_id, attri_text)] for uid, attri_id, attri_text in
                            zip(uids, path_attribute_ids * len(uids), path_attribute_text * len(uids))]
        self._done = False
        self._batch_curr_state = self._batch_get_state(self._batch_path, path_attribute_ids, path_attribute_text)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state

    def batch_step(self, batch_act_idx, ):
        """
        Args:
            batch_act_idx: list of integers.
        Returns:
            batch_next_state: numpy array of size [bs, state_dim].
            batch_reward: numpy array of size [bs].
            done: True/False
        """
        assert len(batch_act_idx) == len(self._batch_path)

        # Execute batch actions.
        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            curr_node_type, curr_node_id = self._batch_path[i][-1][1], self._batch_path[i][-1][2]
            relation, next_node_id = self._batch_curr_actions[i][act_idx]
            if relation == SELF_LOOP:
                next_node_type = curr_node_type
            else:
                next_node_type = KG_RELATION[curr_node_type][relation]
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._is_done()  # must run before get actions, etc.
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state, self._batch_curr_reward, self._done

    def batch_action_mask(self, dropout=0.0):
        """Return action masks of size [bs, act_dim]."""
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_idxs = list(range(len(actions)))
            if dropout > 0 and len(act_idxs) >= 5:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = [act_idxs[0]] + tmp
            act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)

    def print_path(self):
        for path in self._batch_path:
            msg = 'Path: {}({})'.format(path[0][1], path[0][2])
            for node in path[1:]:
                msg += ' =={}=> {}({})'.format(node[0], node[1], node[2])
            print(msg)
