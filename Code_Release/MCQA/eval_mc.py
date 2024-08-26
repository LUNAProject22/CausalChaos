#########################################################################
# We build upon the following work.
# Original file credits as follows. We thank the authors of this work and
# all the related work.
# If you find this work useful, please consider citing the following and 
# related work.
# File Name: main.sh
# Author: Xiao Junbin
# mail: junbin@comp.nus.edu.sg
# @InProceedings{xiao2021next,
#     author    = {Xiao, Junbin and Shang, Xindi and Yao, Angela and Chua, Tat-Seng},
#     title     = {NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#     month     = {June},
#     year      = {2021},
#     pages     = {9777-9786}
# }
# @article{causalchaos2024,
#   title={CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes},
#   author={Parmar, Paritosh and Peh, Eric and Chen, Ruirui and Lam, Ting En and Chen, Yuhan and Tan, Elston and Fernando, Basura},
#   journal={arXiv preprint arXiv:2404.01299},
#   year={2024}
# }
#########################################################################
import os.path as osp
from utils import load_file

map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}

def accuracy_metric(sample_list_file, result_file, dataset):

    sample_list = load_file(sample_list_file)
    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}

    #Added basic eval for our dataset 
    if dataset == 'causalchaos':
        qns_ids = list(sample_list['qid'].astype(str))
        preds = load_file(result_file)
        all_acc = 0
        all_cnt = 0
        for qid in qns_ids:
            all_cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']
            if answer == pred: 
                all_acc += 1
        print('')
        print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))
    
    else:
        for id, row in sample_list.iterrows():
            qns_id = str(row['video']) + '_' + str(row['qid'])
            qtype = str(row['type'])
            print(qns_id,qtype)
            #(combine temporal qns of previous and next as 'TN')
            if qtype == 'TP': qtype = 'TN'
            group[qtype].append(qns_id)

        preds = load_file(result_file)
        group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
        group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
        overall_acc = {'C':0, 'T':0, 'D':0}
        overall_cnt = {'C':0, 'T':0, 'D':0}
        all_acc = 0
        all_cnt = 0
        for qtype, qns_ids in group.items():
            cnt = 0
            acc = 0
            for qid in qns_ids:

                cnt += 1
                answer = preds[qid]['answer']
                pred = preds[qid]['prediction']

                if answer == pred: 
                    acc += 1

            group_cnt[qtype] = cnt
            group_acc[qtype] += acc
            overall_acc[qtype[0]] += acc
            overall_cnt[qtype[0]] += cnt
            all_acc += acc
            all_cnt += cnt


        for qtype, value in overall_acc.items():
            group_acc[qtype] = value
            group_cnt[qtype] = overall_cnt[qtype]

        for qtype in group_acc:
            print(map_name[qtype], end='\t')
        print('')
        for qtype, acc in group_acc.items():
            print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
        print('')
        print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))


def main(result_file, dataset, split, diff,mode='val'):
    if dataset == 'causalchaos':
        dataset_dir = f'dataset/causalchaos/{diff}/'
        data_set = mode
        sample_list_file = osp.join(dataset_dir, f'l{split}_multi_choice_{mode}.csv')

    else:
        dataset_dir = 'dataset/nextqa/'
        data_set = mode
        sample_list_file = osp.join(dataset_dir, data_set+'.csv')

    print('Evaluating {}'.format(result_file))

    accuracy_metric(sample_list_file, result_file, dataset)


if __name__ == "__main__":
    # please set the parameters according to your needs
    model_type = 'HGA'
    mode = 'test'
    model_prefix = 'bert-ft-h256-{}-example'.format(mode)
    #result_file = 'results/{}-{}-{}-{}.json'.format(model_type, model_prefix)
    dataset = 'next-causalchaos'
    split = 0
    diff = 0

    result_file = 'results/causalchaos-{}-{}-{}-glove-test.json'.format(diff,split,model_type)#, model_prefix)

    main(result_file, dataset, split, diff, mode)
