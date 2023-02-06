from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score
import math
import os
import argparse
from math import log
import pandas as pd
from tqdm import tqdm
from functools import reduce
from kg_env import BatchKGEnvironment
from train_agent import ActorCritic
from utils import *
import warnings

warnings.filterwarnings("ignore")


def evaluate(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []

    for i in range(len(topk_matches)):
        label = test_user_products[i]
        pred_list = topk_matches[i]
        if len(pred_list) == 0:
            continue

        dcg = 0.0
        hit_num = 0.0
        if label in pred_list:
            dcg += 1. / (log(i + 2) / log(2))
            hit_num += 1
        # idcg
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
        avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))


def batch_beam_search(env, model, input_data, args):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    topk = [25, 5, 1]
    uids, attri_id, attri_text = input_data
    state_pool = env.reset([attri_id], [attri_text], uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    #     更改path路径时，这里也需要改
    for hop in range(args.max_path_len):
        state_tensor = torch.FloatTensor(state_pool.astype(float)).to(args.device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(args.device)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_paths(pretrain_sd, path_file, args):
    env = BatchKGEnvironment(args)

    model = pretrain_sd
    if args.dataset == Aier_EYE:
        with open('data/my_dataset/test_dataset.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
    elif args.dataset == Medical:
        with open('medical_data/tmp/medical_data/test_dataset.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
    input_list = []
    #   构造输入数据
    for item in test_dataset:
        for key in item:
            if type(key) == int:
                d = key
                s = item[key]
                temp = []
                temp.append(s)
                temp.append(d)
                temp.append(item['attribute_idxs'])
                temp.append(item['attribute_text'])
                input_list.append(temp)
    test_uids = [i[0] for i in input_list]
    test_products = [i[1] for i in input_list]
    test_attir_id = [i[2] for i in input_list]
    test_attir_text = [i[3] for i in input_list]
    all_paths, all_probs = [], []
    for i in tqdm(range(len(test_uids))):
        batch_uids = test_uids[i]
        batch_attri_id = test_attir_id[i]
        batch_attri_text = test_attir_text[i]
        if not batch_uids:
            all_paths.append([])
            all_probs.append([])
        else:
            paths, probs = batch_beam_search(env, model, (batch_uids, batch_attri_id, batch_attri_text), args)
            #     这里不能一股脑全加进去，要按顺序存放
            all_paths.append([paths])
            all_probs.append([probs])

    # 保存预测的路径
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))


def get_properties_vec(embed, idxs):
    if idxs == []:
        return np.zeros_like(embed[0])
    re = np.zeros_like(embed[0])
    for i in idxs:
        re += embed[i]
    return re / len(idxs)


def get_embedding_scores(embeds, attribute_embeds, path, syms):
    _, attr_id, attr_text = path[0][-3:]
    pid = path[-1][-1]
    have_symptom = embeds['have_symptom'][syms].mean(axis=0)
    have_disease = embeds['have_disease'][pid]
    disease_symptom_vec = embeds['disease_symptom'][0]
    attr_id.extend([0] * (attr_num - len(attr_id)))
    attribute_vec = attribute_embeds[attr_id]
    attri_vec = torch.mean(attribute_vec, dim=0).numpy()
    have_symptom += attri_vec
    if args.embedding_type == 'TransR':
        Mr = embeds['Mr_disease_symptom']
        have_symptom = np.matmul(have_symptom, Mr)
    elif args.embedding_type == 'TransH':
        r_hype = embeds['disease_symptom_hype']
        have_symptom = have_symptom - (np.matmul(have_symptom, r_hype.T) * r_hype).squeeze()
        have_disease = have_disease - (np.matmul(have_disease, r_hype.T) * r_hype).squeeze()

    score = np.dot(have_symptom + disease_symptom_vec, have_disease.T)
    return score


def get_atten_scores(have_symptom_embeds, attribute_embeds, disease_symptom_vec, have_disease_embeds,
                     WK_weight, WQ_weight, WV_weight, WK_bias, WQ_bias, WV_bias, path, syms):
    _, attr_id, attr_text = path[0][-3:]
    pid = path[-1][-1]
    have_symptom_vec = have_symptom_embeds[syms].mean(axis=0)
    attr_id.extend([0] * (attr_num - len(attr_id)))
    attribute_vec = attribute_embeds[attr_id]
    K = torch.matmul(attribute_vec, WK_weight) + WK_bias
    Q = torch.matmul(attribute_vec, WQ_weight) + WQ_bias
    V = torch.matmul(attribute_vec, WV_weight) + WV_bias
    score = torch.softmax(torch.matmul(Q, K.transpose(0, 1)), -1) / math.sqrt(attr_num)
    attri_vec = torch.sum(torch.matmul(score, V), dim=0).numpy()
    score = np.dot(have_symptom_vec + attri_vec + disease_symptom_vec, have_disease_embeds[pid].T)
    return score


def get_normal_scores(embeds, attribute_embed, path, syms):
    _, attr_id, attr_text = path[0][-3:]
    pid = path[-1][-1]
    have_symptom = embeds['have_symptom'][syms].mean(axis=0)
    # have_symptom = embeds['have_symptom'][syms]
    have_disease = embeds['have_disease'][pid]
    disease_symptom_vec = embeds['disease_symptom'][0]
    if args.embedding_type == 'TransR':
        Mr = embeds['Mr_disease_symptom']
        have_symptom = np.matmul(have_symptom, Mr)
        have_disease = np.matmul(have_disease, Mr)
    elif args.embedding_type == 'TransH':
        r_hype = embeds['disease_symptom_hype']
        have_symptom = have_symptom - (np.matmul(have_symptom, r_hype.T) * r_hype).squeeze()
        have_disease = have_disease - (np.matmul(have_disease, r_hype.T) * r_hype).squeeze()

    score = np.dot(have_symptom + disease_symptom_vec, have_disease.T)
    return score


def get_bert_scores(embeds, attribute_embeds, path, syms):
    _, attr_id, attr_text = path[0][-3:]
    pid = path[-1][-1]
    have_symptom = embeds['have_symptom'][syms].mean(axis=0)
    have_disease = embeds['have_disease'][pid]
    disease_symptom_vec = embeds['disease_symptom'][0]
    if args.embedding_type == 'TransR':
        Mr = embeds['Mr_disease_symptom']
        have_symptom = np.matmul(have_symptom, Mr)
        have_disease = np.matmul(have_disease, Mr)
    elif args.embedding_type == 'TransH':
        r_hype = embeds['disease_symptom_hype']
        have_symptom = have_symptom - (np.matmul(have_symptom, r_hype.T) * r_hype).squeeze()
        have_disease = have_disease - (np.matmul(have_disease, r_hype.T) * r_hype).squeeze()

    # have_symptom_vec = have_symptom_embeds[_]
    attr_id.extend([0] * (attr_num - len(attr_id)))
    attribute_vec = attribute_embeds[attr_id]
    attri_vec = torch.sum(attribute_vec, dim=0).numpy() / attr_num
    score = np.dot(have_symptom + attri_vec + disease_symptom_vec, have_disease.T)
    return score


def evaluate_paths(path_file, test_labels, args):
    embeds = load_embed(args.dataset)
    if args.model_type == 'atten':
        attribute_embed = torch.relu(torch.from_numpy(embeds['attribute']))
        WK_weight = torch.from_numpy(embeds['WK.weight'])
        WQ_weight = torch.from_numpy(embeds['WQ.weight'])
        WV_weight = torch.from_numpy(embeds['WV.weight'])
        WK_bias = torch.from_numpy(embeds['WK.bias'])
        WQ_bias = torch.from_numpy(embeds['WQ.bias'])
        WV_bias = torch.from_numpy(embeds['WV.bias'])
    elif args.model_type == 'bert':
        bert_attri_vec = torch.from_numpy(embeds['bert_attr_vec'])
    elif args.model_type == 'embedding' or args.model_type == 'w2v':
        attribute_embed = torch.from_numpy(embeds['attribute'])
    elif args.model_type == 'none':
        attribute_embed = np.zeros_like(embeds[HAVE_SYMPTOM][0])
    have_symptom_embeds = embeds[HAVE_SYMPTOM]
    disease_symptom_vec = embeds[DISEASE_SYMPTOM][0]
    have_disease_embeds = embeds[HAVE_DISEASE]

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    all_pid = []
    all_uid = []
    for item in test_labels:
        for i in item:
            if type(i) == int:
                all_pid.append(i)
                all_uid.append(item[i])
    pred_paths = []
    re = results['paths']
    for path, probs in zip(results['paths'], results['probs']):
        temp = []
        #     单独处理每个输入症状的预测结果，用index指示当前数据位置
        for p1, p2 in zip(path, probs):
            syms = []
            for i in p1:
                if i[0][2] not in syms:
                    syms.append(i[0][2])
            for item, prob in zip(p1, p2):
                if item[-1][1] != HAVE_DISEASE:
                    continue
                #   pid == 0,此时未找到疾病
                if item[-1][2] == 0:
                    continue
                if args.model_type == 'atten':
                    path_score = get_atten_scores(have_symptom_embeds, attribute_embed, disease_symptom_vec,
                                                  have_disease_embeds,
                                                  WK_weight, WQ_weight, WV_weight, WK_bias, WQ_bias, WV_bias,
                                                  item, syms)
                elif args.model_type == 'bert':
                    path_score = get_bert_scores(embeds, attribute_embed, item, syms)
                elif args.model_type == 'none':
                    path_score = get_normal_scores(embeds, attribute_embed, item, syms)
                elif args.model_type == 'embedding' or args.model_type == 'w2v':
                    path_score = get_embedding_scores(embeds, attribute_embed, item, syms)
                path_prob = reduce(lambda x, y: x * y, prob)
                #
                if item not in temp:
                    my_path = []
                    my_path.append((syms, item[0][4]))
                    my_path.extend(item[1:])
                    temp.append((path_score, path_prob, my_path))
        pred_paths.append(temp)

    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_path = []
    top5_pids = []
    top3_pids = []
    top1_pids = []
    sort_by = 'score'
    for i in pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(i, key=lambda x: x[0], reverse=True)
            if sorted_path:
                best_path.append(sorted_path[0][-1])
            else:
                best_path.append([])
        elif sort_by == 'prob':
            sorted_path = sorted(i, key=lambda x: x[1], reverse=True)
            best_path.append(sorted_path[0][-1])
        if len(i) == 0:
            top5_pids.append([-1])
        else:
            result = getTop(sorted_path)
            top5 = result[:10]
            top5_pids.append(top5)
    len_ = len(all_pid)
    hit = [0] * 10
    for i in range(len(all_pid)):
        pid = all_pid[i]
        top1_pids.append(top5_pids[i][0])
        if pid == top5_pids[i][0]:
            hit[0] += 1
        if pid in top5_pids[i][:2]:
            hit[1] += 1
        if pid in top5_pids[i][:3]:
            hit[2] += 1
        if pid in top5_pids[i][:4]:
            hit[3] += 1
        if pid in top5_pids[i][:5]:
            hit[4] += 1
        if pid in top5_pids[i][:6]:
            hit[5] += 1
        if pid in top5_pids[i][:7]:
            hit[6] += 1
        if pid in top5_pids[i][:8]:
            hit[7] += 1
        if pid in top5_pids[i][:9]:
            hit[8] += 1
        if pid in top5_pids[i]:
            hit[9] += 1

    hit = list(np.array(hit) / len_)
    precision = precision_score(all_pid, top1_pids, average='macro')
    recall = recall_score(all_pid, top1_pids, average='macro')
    f1 = f1_score(all_pid, top1_pids, average='macro')
    hit.append(f1)
    hit.append(recall)
    hit.append(precision)

    #   保存预测结果
    with open('预测结果.txt', 'wb') as f:
        pickle.dump(best_path, f)
    pd.DataFrame(best_path).to_csv('预测结果.csv')

    return hit


def getTop(sorted_path):
    temp = []
    for _, _, i in sorted_path:
        if i[-1][-1] not in temp:
            temp.append(i[-1][-1])
    return temp


def test(args, model):
    if not model:
        policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
        env = BatchKGEnvironment(args)
        pretrain_sd = torch.load(policy_file)
        model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
        model_sd = model.state_dict()
        model_sd.update(pretrain_sd)
        model.load_state_dict(model_sd)

    if args.dataset == Aier_EYE:
        with open('data/my_dataset/test_dataset.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
    elif args.dataset == Medical:
        with open('medical_data/tmp/medical_data/test_dataset.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
    path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)
    predict_paths(model, path_file, args)

    return evaluate_paths(path_file, test_dataset, args)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=Medical, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=120, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=300, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    parser.add_argument('--embedding_type', type=str, default='TransH')
    parser.add_argument('--model_type', type=str, default='none')
    parser.add_argument('--path_limit', type=int, default=1, help='limit path node type')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    accu = 0
    result = []
    for i in range(1):
        result.append(test(args, None))
    result = np.array(result)
    mean = np.mean(result, axis=0)
    std = np.std(result, axis=0)
    all_ = ['top-1', 'top-2', 'top-3', 'top-4', 'top-5',
            'top-6', 'top-7', 'top-8', 'top-9', 'top-10',
            'precision', 'recall', 'f1', ]
    print("*******************--result--********************")

    for i in range(len(mean)):
        print("{}:  ${}\\pm{}$".format(all_[i], round(mean[i], 4), round(std[i], 4)))

    print("*******************--latex--********************")
    result_str = ''
    for i in range(len(mean)):
        result_str += "&" + str(round(mean[i], 4))
    print(result_str)
