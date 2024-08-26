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
import nltk
#nltk.download('punkt') #uncomment it if you are run the first fime
import pickle
import argparse
from utils import load_file, save_file
from collections import Counter
import string
import pandas as pd
import numpy as np

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(anno_file, threshold):
    """Build a simple vocabulary wrapper."""

    annos = load_file(anno_file)
    #annos=pd.read_csv(anno_file)
    print('total QA pairs', len(annos))
    counter = Counter()

    for (qns,ans0,ans1,ans2,ans3,ans4) in zip(annos['question'], annos['a0'],annos['a1'],annos['a2'],annos['a3'],annos['a4']):
        # qns, ans = vqa['question'], vqa['answer']
        # text = qns # qns +' ' +ans
        text = str(qns) + ' '+ str(ans0)+' '+ str(ans1)+' '+ str(ans2)+' '+ str(ans3)+' '+ str(ans4)
        tokens = nltk.tokenize.word_tokenize(text.lower())
        counter.update(tokens)

    counter = sorted(counter.items(), key=lambda item:item[1], reverse=True)
    #save_file(dict(counter), 'dataset/causalchaos/word_count.json')
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [item[0] for item in counter if item[1] >= threshold]
    #print(len(words))
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
        
    return vocab

def main(args):
    vocab = build_vocab(args.anno_path, args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path', type=str, 
                        default='',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=1,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
