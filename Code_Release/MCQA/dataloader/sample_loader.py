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
import torch
from torch.utils.data import Dataset, DataLoader
from .util import load_file
import os.path as osp
import numpy as np
import nltk
import h5py
import pandas as pd

class VidQADataset(Dataset):
    """load the dataset in dataloader"""

    def __init__(self, video_feature_path, video_feature_cache, sample_list_path, vocab, use_bert, dataset, mode, split, diff):
        self.video_feature_path = video_feature_path
        self.vocab = vocab
        self.dataset = dataset
        self.mode = mode
        if self.dataset=='causalchaos':
            self.sample_list = pd.read_csv('./dataset/causalchaos/{}/{}_{}.csv'.format(split,diff,mode))
            print("Anno file loaded from: ./dataset/causalchaos/{}/{}_{}.csv ".format(split,diff,mode))

        self.max_qa_length = 50
        self.use_bert = use_bert
        self.use_frame = True # False for STVQA
        self.use_mot = True # False for STVQA
        self.use_spatial = False #True for STVQA
        

        if self.dataset=='nextqa':
            self.sample_list = load_file(sample_list_file)
            if self.use_bert:
                self.bert_file = osp.join(video_feature_path, 'qas_bert/bert_ft_{}.h5'.format(mode))
            if not self.use_spatial:
                vid_feat_file = osp.join(video_feature_path, 'vid_feat/app_mot_{}.h5'.format(mode))

                print('Load {}...'.format(vid_feat_file))
                self.frame_feats = {}
                self.mot_feats = {}
                with h5py.File(vid_feat_file, 'r') as fp:
                    vids = fp['ids']
                    feats = fp['feat']
                    for id, (vid, feat) in enumerate(zip(vids, feats)):
                        if self.use_frame:
                            self.frame_feats[str(vid)] = feat[:, :2048]  # (16, 2048)
                        if self.use_mot:
                            self.mot_feats[str(vid)] = feat[:, 2048:]  # (16, 2048)
            else:
                # if you don't have enough memory(>60G), you can read feature from hdf5 at each iteration
                vid_feat_file = osp.join(video_feature_path, 'spatial_feat/feat_maps_{}.h5'.format(mode))
                print('Load large file {}...'.format(vid_feat_file))
                self.spatial_feats = {}
                with h5py.File(vid_feat_file, 'r') as fp:
                    vids = fp['ids']
                    feats = fp['feat']
                    for id, (vid, feat) in enumerate(zip(vids, feats)):
                        self.spatial_feats[str(vid)] = feat  # (*, 4096, 7, 7) (obtained by np.concatenate((app, mot), axis=1)

        elif self.dataset=='causalchaos':
            if self.use_bert:
                self.bert_file = osp.join(video_feature_path, 'causalchaos/bert_ft_{}.h5'.format(mode))
            if not self.use_spatial:
                vid_feat_file = osp.join(video_feature_path, 'causalchaos/latest_app_mot_feat.h5')

                print('Load {}...'.format(vid_feat_file))
                self.frame_feats = {}
                self.mot_feats = {}
                with h5py.File(vid_feat_file, 'r') as fp:
                    vids = fp['ids']
                    feats = fp['feat']
                    for id, (vid, feat) in enumerate(zip(vids, feats)):
                        if self.use_frame:
                            self.frame_feats[str(vid)] = feat[:, :2048]  # (16, 2048)
                        if self.use_mot:
                            self.mot_feats[str(vid)] = feat[:, 2048:]  # (16, 2048)
            else:
                # if you don't have enough memory(>60G), you can read feature from hdf5 at each iteration
                vid_feat_file = osp.join(video_feature_path, 'spatial_feat/feat_maps_{}.h5'.format(mode))
                print('Load large file {}...'.format(vid_feat_file))
                self.spatial_feats = {}
                with h5py.File(vid_feat_file, 'r') as fp:
                    vids = fp['ids']
                    feats = fp['feat']
                    for id, (vid, feat) in enumerate(zip(vids, feats)):
                        self.spatial_feats[str(vid)] = feat  # (*, 4096, 7, 7) (obtained by np.concatenate((app, mot), axis=1)

        #TODO add combined dataoader for training on NExT-QA + causalchaos


    def __len__(self):
        return len(self.sample_list)

    def get_video_feature(self, video_name):
        """
        :param video_name:
        :return:
        """
        if self.use_spatial:
            video_feature = self.spatial_feats[video_name] #(16, 4096, 7 ,7)
        else:    
            if self.use_frame:
                app_feat = self.frame_feats[video_name]
                video_feature = app_feat # (16, 2048)
            if self.use_mot:
                mot_feat = self.mot_feats[video_name]
                video_feature = np.concatenate((video_feature, mot_feat), axis=1) #(16, 4096)

                #np.random.shuffle(video_feature)
                #np.apply_along_axis(np.random.shuffle,1,video_feature) 
                
        # print(video_feature.shape)
        return torch.from_numpy(video_feature).type(torch.float32)


    def get_word_idx(self, text):
        """
        """
        tokens = nltk.tokenize.word_tokenize(str(text).lower())
        token_ids = [self.vocab(token) for i, token in enumerate(tokens) if i < 25]

        return token_ids

    def get_trans_matrix(self, candidates):

        qa_lengths = [len(qa) for qa in candidates]
        candidates_matrix = torch.zeros([5, self.max_qa_length]).long()
        for k in range(5):
            sentence = candidates[k]
            candidates_matrix[k, :qa_lengths[k]] = torch.Tensor(sentence)

        return candidates_matrix, qa_lengths


    def __getitem__(self, idx):
        """
        """
        if self.dataset == 'causalchaos':
            cur_sample = self.sample_list.loc[idx]
            video_name, qns, ans, qid = str(cur_sample['vid']), str(cur_sample['question']),\
                                        int(cur_sample['answer']), str(cur_sample['qid'])
            
            candidate_qas = []
            qns2ids = [self.vocab('<start>')]+self.get_word_idx(qns)+[self.vocab('<end>')]
            for id in range(5):
                cand_ans = cur_sample['a'+str(id)]
                ans2id = self.get_word_idx(cand_ans) + [self.vocab('<end>')]
                candidate_qas.append(qns2ids+ans2id)

            candidate_qas, qa_lengths = self.get_trans_matrix(candidate_qas)

            if self.use_bert:
                with h5py.File(self.bert_file, 'r') as fp:
                    temp_feat = fp['feat'][idx]
                    candidate_qas = torch.from_numpy(temp_feat).type(torch.float32)
                for i in range(5):
                    valid_row = nozero_row(candidate_qas[i])
                    qa_lengths[i] = valid_row

            video_feature = self.get_video_feature(qid)
            qns_key = qid
            qa_lengths = torch.tensor(qa_lengths)

        else:
            cur_sample = self.sample_list.loc[idx]
            video_name, qns, ans, qid = str(cur_sample['vid']), str(cur_sample['question']),\
                                        int(cur_sample['answer']), str(cur_sample['qid'])
            candidate_qas = []
            qns2ids = [self.vocab('<start>')]+self.get_word_idx(qns)+[self.vocab('<end>')]
            for id in range(5):
                cand_ans = cur_sample['a'+str(id)]
                ans2id = self.get_word_idx(cand_ans) + [self.vocab('<end>')]
                candidate_qas.append(qns2ids+ans2id)

            candidate_qas, qa_lengths = self.get_trans_matrix(candidate_qas)
            if self.use_bert:
                with h5py.File(self.bert_file, 'r') as fp:
                    temp_feat = fp['feat'][idx]
                    candidate_qas = torch.from_numpy(temp_feat).type(torch.float32)
                for i in range(5):
                    valid_row = nozero_row(candidate_qas[i])
                    qa_lengths[i] = valid_row

            video_feature = self.get_video_feature(video_name)
            qns_key = video_name + '_' + qid
            qa_lengths = torch.tensor(qa_lengths)
        return video_feature, candidate_qas, qa_lengths, ans, qns_key


def nozero_row(A):
    i = 0
    for row in A:
        if row.sum()==0:
            break
        i += 1

    return i

class QALoader():
    def __init__(self, batch_size, num_worker, video_feature_path, video_feature_cache,
                 sample_list_path, split, vocab, use_bert, dataset, diff, train_shuffle=True, val_shuffle=False):
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.video_feature_path = video_feature_path
        self.video_feature_cache = video_feature_cache
        self.sample_list_path = sample_list_path
        self.vocab = vocab
        self.use_bert = use_bert
        self.dataset = dataset
        self.split =split
        self.diff = diff
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle

    def run(self, mode=''):
        if mode != 'train':
            train_loader = ''
            val_loader = self.validate(mode)
        else:
            train_loader = self.train('train')
            val_loader = self.validate('val')
        return train_loader, val_loader


    def train(self, mode):

        training_set = VidQADataset(self.video_feature_path, self.video_feature_cache, self.sample_list_path,
                                       self.vocab, self.use_bert, self.dataset,mode, self.split, self.diff)

        print('Eligible video-qa pairs for training : {}'.format(len(training_set)))

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_worker
            )

        return train_loader


    def validate(self, mode):

        validation_set = VidQADataset(self.video_feature_path, self.video_feature_cache, self.sample_list_path,
                                         self.vocab, self.use_bert, self.dataset, mode, self.split, self.diff)

        print('Eligible video-qa pairs for validation : {}'.format(len(validation_set)))

        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_worker
            )

        return val_loader

