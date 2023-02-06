from __future__ import absolute_import, division, print_function

import os
import argparse

import jieba
from gensim.models.keyedvectors import KeyedVectors
import torch.optim
from transformers import AutoModel, AutoTokenizer
from utils import *
from data_utils import DataLoader
from transe_model import KnowledgeEmbedding

logger = None


def train(args):
    dataset = load_dataset(args.dataset)
    dataloader = DataLoader(dataset, args.batch_size, args.dataset)
    words_to_train = args.epochs * dataset.review.word_count + 1
    model = KnowledgeEmbedding(dataset, args).to(args.device)

    # for k, v in model.named_parameters():
    #       print(k)

    if args.model_type == 'bert':
        #     分层训练
        bert_params = list(map(id, model.bert.parameters()))
        unfreeze_layers = ['encoder.layer.11', 'pooler']
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        base_params = filter(lambda p: id(p) not in bert_params, model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params, 'lr': args.lr},
                                     {'params': model.bert.parameters()}], lr=1e-5)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    steps = 0
    smooth_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            # Set learning rate.
            lr = args.lr * max(1e-4, 1.0 - dataloader.finished_word_num / float(words_to_train))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Get training batch.

            batch_idxs = dataloader.get_batch
            # Train model.
            optimizer.zero_grad()
            train_loss = model(batch_idxs)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            smooth_loss += train_loss.item() / args.steps_per_checkpoint

            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                logger.info('Epoch: {:02d} | '.format(epoch) +
                            'Words: {:d}/{:d} | '.format(dataloader.finished_word_num, words_to_train) +
                            'Lr: {:.5f} | '.format(lr) +
                            'Smooth loss: {:.5f}'.format(smooth_loss))
                smooth_loss = 0.0
    torch.save(model.state_dict(), '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, args.epochs))


def extract_embeddings(args):
    """Note that last entity embedding is of size [vocab_size+1, d]."""
    model_file = '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, args.epochs)
    print('Load embeddings', model_file)
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)

    embeds = {
        HAVE_DISEASE: state_dict['have_disease.weight'].cpu().data.numpy(),
        HAVE_SYMPTOM: state_dict['have_symptom.weight'].cpu().data.numpy(),
        SURGERY: state_dict['surgery.weight'].cpu().data.numpy(),
        MEDICINE: state_dict['medicine.weight'].cpu().data.numpy(),
        WORD: state_dict['word.weight'].cpu().data.numpy(),
        DISEASE_SYMPTOM: [
            state_dict['disease_symptom'].cpu().data.numpy()[0],
            state_dict['disease_symptom_bias.weight'].cpu().data.numpy()
        ],
        MENTION: [
            state_dict['mentions'].cpu().data.numpy()[0],
            state_dict['mentions_bias.weight'].cpu().data.numpy()
        ],
        DESCRIBED_AS: [
            state_dict['described_as'].cpu().data.numpy()[0],
            state_dict['described_as_bias.weight'].cpu().data.numpy()
        ],
        DISEASE_SURGERY: [
            state_dict['disease_surgery'].cpu().data.numpy()[0],
            state_dict['disease_surgery_bias.weight'].cpu().data.numpy()
        ],
        DISEASE_DRUG: [
            state_dict['disease_drug'].cpu().data.numpy()[0],
            state_dict['disease_drug_bias.weight'].cpu().data.numpy()
        ],
        RELATED_DISEASE: [
            state_dict['related_disease'].cpu().data.numpy()[0],
            state_dict['related_disease_bias.weight'].cpu().data.numpy()
        ],
        RELATED_SYMPTOM: [
            state_dict['related_symptom'].cpu().data.numpy()[0],
            state_dict['related_symptom_bias.weight'].cpu().data.numpy()
        ],
    }
    if args.embedding_type == 'TransR':
        embeds.update({
            'Mr_disease_symptom': state_dict['Mr_disease_symptom'].cpu().data.numpy(),
            'Mr_mentions': state_dict['Mr_mentions'].cpu().data.numpy(),
            'Mr_described_as': state_dict['Mr_described_as'].cpu().data.numpy(),
            'Mr_disease_surgery': state_dict['Mr_disease_surgery'].cpu().data.numpy(),
            'Mr_disease_drug': state_dict['Mr_disease_drug'].cpu().data.numpy(),
            'Mr_related_disease': state_dict['Mr_related_disease'].cpu().data.numpy(),
            'Mr_related_symptom': state_dict['Mr_related_symptom'].cpu().data.numpy(),
        })
    elif args.embedding_type == 'TransH':
        embeds.update({
            'disease_symptom_hype': state_dict['disease_symptom_hype'].cpu().data.numpy(),
            'mentions_hype': state_dict['mentions_hype'].cpu().data.numpy(),
            'described_as_hype': state_dict['described_as_hype'].cpu().data.numpy(),
            'disease_surgery_hype': state_dict['disease_surgery_hype'].cpu().data.numpy(),
            'disease_drug_hype': state_dict['disease_drug_hype'].cpu().data.numpy(),
            'related_disease_hype': state_dict['related_disease_hype'].cpu().data.numpy(),
            'related_symptom_hype': state_dict['related_symptom_hype'].cpu().data.numpy(),
                       })

    if args.model_type == 'atten':
        embeds.update({
            'attribute': state_dict['attribute.weight'].cpu().data.numpy(),
            'WK.weight': state_dict['WK.weight'].cpu().data.numpy(),
            'WQ.weight': state_dict['WQ.weight'].cpu().data.numpy(),
            'WV.weight': state_dict['WV.weight'].cpu().data.numpy(),
            'WK.bias': state_dict['WK.bias'].cpu().data.numpy(),
            'WQ.bias': state_dict['WQ.bias'].cpu().data.numpy(),
            'WV.bias': state_dict['WV.bias'].cpu().data.numpy(), })
    elif args.model_type == 'bert':
        with open('data/dictionary/attribute.txt', 'rb') as f:
            attribute = pickle.load(f)
        #     保存bert生成的embedding
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained('cyclone/simcse-chinese-roberta-wwm-ext')
            bert = AutoModel.from_pretrained('cyclone/simcse-chinese-roberta-wwm-ext').to(args.device)
            pooler = state_dict['pooler.weight'].cpu().data.numpy()
            pooler_b = state_dict['pooler.bias'].cpu().data.numpy()
            x = bert.state_dict()
            for name, param in bert.named_parameters():
                t = state_dict['bert.' + name]
                x.update({name: t})
            bert.load_state_dict(x)
            #     idx为0代表无记录，提至最前
            attribute_text = list(attribute.keys())
            attribute_text = [attribute_text[-1]] + attribute_text[:-1]
            bert_input = tokenizer(attribute_text, padding=True, truncation=True, return_tensors='pt').to(
                args.device)
            out = bert(bert_input['input_ids'], bert_input['attention_mask'])[1].detach().cpu().numpy()
            attr_vec = np.dot(out, pooler.T) + pooler_b
            embeds.update({'bert_attr_vec': attr_vec, })
    elif args.model_type == 'embedding':
        embeds.update({'attribute': state_dict['attribute.weight'].cpu().data.numpy()})
    elif args.model_type == 'w2v':
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
        attr_vec = np.dot(np.array(embedding, dtype='float32'), state_dict['pooler.weight'].cpu().data.numpy().T)
        embeds.update({'attribute': attr_vec})
    save_embed(args.dataset, embeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=Medical)
    parser.add_argument('--name', type=str, default='train_transe_model', help='model name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='2', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--model_type', type=str, default='none', help='model type.bert,atten,none,embedding,w2v')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
    parser.add_argument('--embed_size', type=int, default=100, help='knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=128, help='number of negative samples.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=1000, help='Number of steps for checkpoint.')
    parser.add_argument('--embedding_type', type=str, default='TransH',
                        help='Embedding model type,TransR,TransE or TransH')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)
    set_random_seed(args.seed)
    train(args)
    extract_embeddings(args)


if __name__ == '__main__':
    main()
