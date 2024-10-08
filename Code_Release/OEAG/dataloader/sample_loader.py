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
from .util import load_file, pkdump, pkload
import os.path as osp
import numpy as np
import nltk
import h5py
import pandas as pd

class VidQADataset(Dataset):
    """load the dataset in dataloader"""

    def __init__(self, video_feature_path, video_feature_cache, sample_list_path, vocab_qns, vocab_ans, dataset, mode, split, version):

        self.video_feature_path = video_feature_path
        self.vocab_qns = vocab_qns
        self.vocab_ans = vocab_ans
        self.dataset = dataset
        sample_list_file = osp.join(sample_list_path, '{}.csv'.format(mode))
        self.video_feature_cache = video_feature_cache
        self.use_frame = True
        self.use_mot = True
        self.frame_feats = {}
        self.mot_feats = {}
        self.mode=mode
        
        if self.dataset == 'nextqa':
            self.sample_list = load_file(sample_list_file)
            vid_feat_file = osp.join(video_feature_path, 'vid_feat/app_mot_{}.h5'.format(mode))
            with h5py.File(vid_feat_file, 'r') as fp:
                vids = fp['ids']
                feats = fp['feat']
                for id, (vid, feat) in enumerate(zip(vids, feats)):
                    if self.use_frame:
                        self.frame_feats[str(vid)] = feat[:,:2048]  # (16, 2048)
                    if self.use_mot:
                        self.mot_feats[str(vid)] = feat[:, 2048:]  # (16, 2048)\
        
        elif self.dataset == 'causalchaos':
            self.sample_list = pd.read_csv('./dataset/causalchaos/{}/{}_open_end_{}.csv'.format(split,version,mode))
            print('Loading Annotation from : ','./dataset/causalchaos/{}/{}_open_end_{}.csv'.format(split,version,mode))
            vid_feat_file = osp.join(video_feature_path, 'causalchaos/latest_app_mot_feat.h5')
            with h5py.File(vid_feat_file, 'r') as fp:
                vids = fp['ids']
                feats = fp['feat']
                for id, (vid, feat) in enumerate(zip(vids, feats)):
                    if self.use_frame:
                        self.frame_feats[vid] = feat[:,:2048]  # (16, 2048)
                    if self.use_mot:
                        self.mot_feats[vid] = feat[:, 2048:]  # (16, 2048)\
        
        # TODO: add dataloader for combined training of NExT-QA + causalchaos
    def __len__(self):
        return len(self.sample_list)

    def get_video_feature(self, video_name):
        """

        """
        if self.use_frame:
            app_feat = self.frame_feats[video_name]
            video_feature = app_feat # (16, 2048)
        if self.use_mot:
            mot_feat = self.mot_feats[video_name]
            video_feature = np.concatenate((video_feature, mot_feat), axis=1) #(16, 4096)

        return torch.from_numpy(video_feature).type(torch.float32)


    def get_word_idx(self, text, src='qns'):
        """
        convert relation to index sequence
        :param relation:
        :return:
        """
        if src=='qns': vocab = self.vocab_qns
        elif src=='ans': vocab = self.vocab_ans
        tokens = nltk.tokenize.word_tokenize(str(text).lower())
        text = []
        text.append(vocab('<start>'))
        text.extend([vocab(token) for i,token in enumerate(tokens) if i < 23])
        #text.append(vocab('<end>'))
        target = torch.Tensor(text)

        return target


    def __getitem__(self, idx):
        """

        """
        if self.dataset == 'causalchaos':
            qtype=""
            sample = self.sample_list.loc[idx]
            video_name, qns, ans = sample['vid'], sample['question'], sample['answer']
            qid = sample['qid']
            #print(qid)
            video_name = str(video_name)
            qns, ans, qid = str(qns), str(ans), qid
            #video_feature = torch.tensor([0])
            video_feature = self.get_video_feature(qid)
            qns2idx = self.get_word_idx(qns, 'qns')
            ans2idx = self.get_word_idx(ans, 'ans')

        else:
            sample = self.sample_list.loc[idx]
            video_name, qns, ans = sample['video'], sample['question'], sample['answer']
            qid, qtype = sample['qid'], sample['type']
            video_name = str(video_name)
            qns, ans, qid, qtype = str(qns), str(ans), str(qid), str(qtype)


            #video_feature = torch.tensor([0])
            video_feature = self.get_video_feature(video_name)

            qns2idx = self.get_word_idx(qns, 'qns')
            ans2idx = self.get_word_idx(ans, 'ans')

        return video_feature, qns2idx, ans2idx, video_name, qid, qtype


class QALoader():
    def __init__(self, batch_size, num_worker, video_feature_path, video_feature_cache,
                 sample_list_path, vocab_qns, vocab_ans, dataset, split, version, train_shuffle=True, val_shuffle=False):
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.video_feature_path = video_feature_path
        self.video_feature_cache = video_feature_cache
        self.sample_list_path = sample_list_path
        self.vocab_qns = vocab_qns
        self.vocab_ans = vocab_ans
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.dataset = dataset
        self.split = split
        self.version = version


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
                                       self.vocab_qns, self.vocab_ans, self.dataset, mode, self.split , self.version)

        print('Eligible QA pairs for training : {}'.format(len(training_set)))
        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_worker,
            collate_fn=collate_fn)

        return train_loader

    def validate(self, mode):

        validation_set = VidQADataset(self.video_feature_path, self.video_feature_cache, self.sample_list_path,
                                         self.vocab_qns, self.vocab_ans, self.dataset, mode, self.split , self.version)

        print('Eligible QA pairs for validation : {}'.format(len(validation_set)))
        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_worker,
            collate_fn=collate_fn)

        return val_loader


def collate_fn (data):
    """
    """
    data.sort(key=lambda x : len(x[1]), reverse=True)
    videos, qns2idx, ans2idx, video_names, qids, qtypes = zip(*data)

    #merge videos
    videos = torch.stack(videos, 0)

    #merge relations
    qns_lengths = [len(qns) for qns in qns2idx]
    targets_qns = torch.zeros(len(qns2idx), max(qns_lengths)).long()
    for i, qns in enumerate(qns2idx):
        end = qns_lengths[i]
        targets_qns[i, :end] = qns[:end]

    ans_lengths = [len(ans) for ans in ans2idx]
    targets_ans = torch.zeros(len(ans2idx), 34).long()
    for i, ans in enumerate(ans2idx):
        end = ans_lengths[i]
        targets_ans[i, :end] = ans[:end]

    return videos, targets_qns, qns_lengths, targets_ans, ans_lengths, video_names, qids, qtypes
