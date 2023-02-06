import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def draw_bar():
    svm = [0.07, 0.13, 0.19, 0.21, 0.24]
    xgboost = [0.16, 0.20, 0.25, 0.30, 0.32]
    GAMP = [0.21, 0.216, 0.263, 0.266, 0.28]
    BED = [0.1818, 0.270, 0.326, 0.334, 0.360]
    diaformer = [0.22, 0.303, 0.3441, 0.3446, 0.4436]
    TransE = [0.2582, 0.3522, 0.4109, 0.4423, 0.4688]

    TransR = [0.325, 0.425, 0.475, 0.4833, 0.5139]
    X = ['HITS@1', 'HITS@2', 'HITS@3', 'HITS@4', 'HITS@5']

    plt.figure(figsize=(8, 4), dpi=600)

    # 两组数据
    x = np.arange(len(X))  # x轴刻度标签位置
    width = 0.1  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width/2，x + width/2即每组数据在x轴上的位置
    plt.bar(x - 3 * width, svm, width, label='SVM', color='lightgrey')
    plt.bar(x - 2 * width, xgboost, width, label='XGBoost', color='silver')
    plt.bar(x - 1 * width, GAMP, width, label='GAMP', color='darkgray')
    plt.bar(x + 0 * width, BED, width, label='BED', color='gray')

    plt.bar(x + 1 * width, diaformer, width, label='Diaformer', color='dimgrey')
    # plt.bar(x + 2 * width, TransE, width, label='TransE', color='#696960')
    plt.bar(x + 2 * width, TransR, width, label='RDKG', color='k')

    plt.xticks(x, labels=X)
    plt.legend(fontsize=8)

    plt.savefig('baseline_bar.png', dpi=600, format='png')


def draw_c():
    # TransR
    none = [0.2917, 0.3556, 0.4111, 0.45015, 0.525]
    embedding = [0.3528, 0.4444, 0.4806, 0.5, 0.5333]
    w2v = [0.3028, 0.4194, 0.4667, 0.475, 0.5101]

    # TransE
    e_w2v = [0.252, 0.2813, 0.4111, 0.4694, 0.4819]
    e_none = [0.2611, 0.3361, 0.3917, 0.4361, 0.475]
    e_embedding = [0.2917, 0.3528, 0.425, 0.4528, 0.4917]

    #  TransH
    h_none = [0.2907, 0.3657, 0.3889, 0.4213, 0.4959]
    h_embedding = [0.3333, 0.463, 0.5078, 0.5141, 0.5226]
    h_w2v = [0.2963, 0.412, 0.4306, 0.4491, 0.463]

    plt.figure(figsize=(12, 5))

    ax = plt.subplot(121)
    ax.set_ylim(0.25, 0.6)

    ax2 = plt.subplot(122)
    # ax3 = plt.subplot(122)
    # ax3.set_ylim(0.25, 0.6)

    ax2.set_ylim(0.25, 0.6)
    ax.set_title("TranR", fontweight='bold')
    ax2.set_title("TransE", fontweight='bold')
    # ax3.set_title("TransH", fontweight='bold')

    x = ['HITS@1', 'HITS@2', 'HITS@3', 'HITS@4', 'HITS@5']

    ax.plot(x, none, marker='x', linestyle=':', lw=4, label='Base')  # ‘x’ : x号
    ax.plot(x, embedding, marker='p', linestyle='dashed', lw=4, label='NN')  # ‘p’ : 五角星
    ax.plot(x, w2v, marker='1', linestyle='dotted', lw=4, label='W2v')  # ‘1’ : 三脚架标记
    ax.tick_params(labelsize=13)
    ax.legend(loc='upper left')

    ax2.plot(x, e_none, marker='x', linestyle=':', lw=4, label='Base')  # ‘x’ : x号
    ax2.plot(x, e_embedding, marker='p', linestyle='dashed', lw=3, label='NN')  # ‘p’ : 五角星
    ax2.plot(x, e_w2v, marker='1', linestyle='dotted', lw=4, label='W2v')  # ‘1’ : 三脚架标记
    ax2.tick_params(labelsize=13)
    ax2.legend(loc='upper left')

    # ax3.plot(x, h_none, marker='x', linestyle=':', lw=4, label='Base')  # ‘x’ : x号
    # ax3.plot(x, h_embedding, marker='p', linestyle='dashed', lw=3, label='NN')  # ‘p’ : 五角星
    # ax3.plot(x, h_w2v, marker='1', linestyle='dotted', lw=4, label='W2v')  # ‘1’ : 三脚架标记
    # ax3.tick_params(labelsize=13)
    # ax3.legend(loc='upper left')

    plt.savefig('embedding_top.png', dpi=600, format='png', bbox_inches='tight', pad_inches=0)
    # ax2.savefig('embedding_top.png', dpi=600, format='png', )


def draw_k():
    # TransR
    k1 = [0.3361, 0.4444, 0.4694, 0.4861, 0.5111]
    k2 = [0.2778, 0.4028, 0.4583, 0.5417, 0.5694]
    k3 = [0.2861, 0.3889, 0.4556, 0.4556, 0.55]

    # TransE
    e_k3 = [0.2532, 0.3564, 0.4082, 0.4435, 0.4915]
    e_k2 = [0.2761, 0.3782, 0.4109, 0.4532, 0.5250]
    e_k1 = [0.3106, 0.3776, 0.425, 0.4892, 0.4997]

    plt.figure(figsize=(12, 5))

    ax = plt.subplot(121)
    ax.set_ylim(0.25, 0.6)
    ax2 = plt.subplot(122)
    ax2.set_ylim(0.25, 0.6)

    ax.set_title("TranR", fontweight='bold')
    ax2.set_title("TransE", fontweight='bold')

    x = ['HITS@1', 'HITS@2', 'HITS@3', 'HITS@4', 'HITS@5']

    ax.plot(x, k1, marker='x', linestyle=':', lw=4, label='L=1')  # ‘x’ : x号
    ax.plot(x, k2, marker='p', linestyle='dashed', lw=4, label='L=2')  # ‘p’ : 五角星
    ax.plot(x, k3, marker='1', linestyle='dotted', lw=4, label='L=3')  # ‘1’ : 三脚架标记
    ax.legend()
    ax.tick_params(labelsize=13)
    ax.legend(loc='upper left')

    ax2.plot(x, e_k1, marker='x', linestyle=':', lw=4, label='L=1')  # ‘x’ : x号
    ax2.plot(x, e_k2, marker='p', linestyle='dashed', lw=4, label='L=2')  # ‘p’ : 五角星
    ax2.plot(x, e_k3, marker='1', linestyle='dotted', lw=4, label='L=3')  # ‘1’ : 三脚架标记
    ax2.tick_params(labelsize=13)
    ax2.legend(loc='upper left')

    ax2.tick_params(labelsize=13)

    plt.savefig('top_with_ks.png', dpi=600, format='png', bbox_inches='tight', pad_inches=0)


def draw_one(data, model_name, fig_name, label_name):
    e_k1, e_k2, e_k3 = data[0], data[1], data[2]
    # TransE
    # e_k3 = [0.2532, 0.3564, 0.4082, 0.4435, 0.4915]
    # e_k2 = [0.2761, 0.3782, 0.4109, 0.4532, 0.5250]
    # e_k1 = [0.3106, 0.3776, 0.425, 0.4892, 0.4997]
    if label_name == "k":
        label = ['K=1', 'K=2', 'K=3']
    else:
        label = ['Base', 'NN', 'W2v']

    # plt.figure(figsize=(12, 5))
    plt.cla()
    plt.ylim(0.25, 0.6)
    plt.title(model_name, fontweight='bold')
    x = ['HITS@1', 'HITS@2', 'HITS@3', 'HITS@4', 'HITS@5']

    plt.plot(x, e_k1, marker='x', linestyle=':', lw=4, label=label[0])  # ‘x’ : x号
    plt.plot(x, e_k2, marker='p', linestyle='dashed', lw=4, label=label[1])  # ‘p’ : 五角星
    plt.plot(x, e_k3, marker='1', linestyle='dotted', lw=4, label=label[2])  # ‘1’ : 三脚架标记
    plt.tick_params(labelsize=13)
    plt.legend(loc='upper left')
    plt.tick_params(labelsize=13)

    plt.savefig('new_result/' + fig_name, dpi=600, format='png')


draw_k()
# # TransR
# k1 = [0.3361, 0.4444, 0.4694, 0.4861, 0.5111]
# k2 = [0.2778, 0.4028, 0.4583, 0.5417, 0.5694]
# k3 = [0.2861, 0.3889, 0.4556, 0.4556, 0.55]
# draw_one([k1, k2, k3], 'TransR', 'transR_ks.png', 'k')
# # TransE
# e_k3 = [0.2532, 0.3564, 0.4082, 0.4435, 0.4915]
# e_k2 = [0.2761, 0.3782, 0.4109, 0.4532, 0.5250]
# e_k1 = [0.3106, 0.3776, 0.425, 0.4892, 0.4997]
# draw_one([e_k1, e_k2, e_k3], 'TransE', 'transE_ks.png', 'k')
# #  TransH
# h_k1 = [0.2907, 0.3657, 0.3889, 0.4213, 0.4259]
# h_k2 = [0.3148, 0.463, 0.4861, 0.5, 0.5556]
# h_k3 = [0.3201, 0.4365, 0.4701, 0.5021, 0.5412]
# draw_one([h_k1, h_k2, h_k3], 'TransH', 'transH_ks.png', 'k')
#
# # TransR
# none = [0.2917, 0.3556, 0.4111, 0.4694, 0.525]
# embedding = [0.3528, 0.4444, 0.4806, 0.5, 0.5333]
# w2v = [0.3028, 0.4194, 0.4667, 0.475, 0.4917]
# draw_one([none, embedding, w2v], 'TransR', 'transR_embedding.png', 'e')
# # TransE
# e_none = [0.2611, 0.3361, 0.3917, 0.4361, 0.475]
# e_embedding = [0.2917, 0.3528, 0.425, 0.4528, 0.4917]
# e_w2v = [0.252, 0.2813, 0.4111, 0.4694, 0.4819]
# draw_one([e_none, e_embedding, e_w2v], 'TransE', 'transE_embedding.png', 'e')
# #  TransH
# h_none = [0.2907, 0.3657, 0.3889, 0.4213, 0.4259]
# h_embedding = [0.3333, 0.463, 0.5078, 0.5141, 0.5226]
# h_w2v = [0.2963, 0.412, 0.4306, 0.4491, 0.463]
# draw_one([h_none, h_embedding, h_w2v], 'TransH', 'transH_embedding.png', 'e')
draw_c()
