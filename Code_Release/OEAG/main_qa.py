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
from videoqa import *
import dataloader
from build_vocab import Vocabulary
from utils import *
import argparse
import eval_oe


def main(args):

    mode = args.mode
    if mode == 'train':
        batch_size = 64
        num_worker = 8
    else:
        batch_size = 64
        num_worker = 8
    spatial = False
    if spatial:
        #for STVQA
        video_feature_path = '../data/feats/spatial/'
        video_feature_cache = '../data/feats/cache_spatial/'
    else:
        video_feature_path = '../data/feats/'
        video_feature_cache = '../data/feats/cache/'

    dataset = args.dataset
    split = args.split
    version = args.version
    sample_list_path = 'dataset/{}/'.format(dataset)

    # We separate the dicts for qns and ans, in case one wants to use different word-dicts for them.
    vocab_qns = pkload('dataset/{}/vocab.pkl'.format(dataset))
    vocab_ans = pkload('dataset/{}/vocab.pkl'.format(dataset))

    word_type = 'glove'
    glove_embed_qns = 'dataset/{}/{}_embed.npy'.format(dataset, word_type)
    glove_embed_ans = 'dataset/{}/{}_embed.npy'.format(dataset, word_type)
    checkpoint_path = 'models'
    model_type = 'HGA'

    model_prefix = 'same-att-qns23ans7'
    vis_step = 116
    lr_rate = 5e-5
    epoch_num = 100

    data_loader = dataloader.QALoader(batch_size, num_worker, video_feature_path, video_feature_cache,
                                      sample_list_path, vocab_qns, vocab_ans, dataset, split, version, True, False)

    train_loader, val_loader = data_loader.run(mode=mode)

    vqa = VideoQA(vocab_qns, vocab_ans, train_loader, val_loader, glove_embed_qns, glove_embed_ans,
                  checkpoint_path, model_type, model_prefix, vis_step,lr_rate, batch_size, epoch_num , dataset, split, version)
    ep = 47
    acc = 0.2413

    model_file = f'{version}-{dataset}-{split}-{model_type}-{model_prefix}-{ep}-{acc:.4f}.ckpt'

    if mode != 'train':
        result_file = f'{version}-{dataset}-{split}-{model_type}-{model_prefix}-{mode}.json'
        vqa.predict(model_file, result_file)
        eval_oe.main(result_file, mode, dataset, split, version)
    else:
        model_file = f'{version}-{dataset}--{split}--{model_type}-{model_prefix}-44-0.2140.ckpt'
        vqa.run(model_file, pre_trained=False)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', type=int,
                        default=0, help='gpu device id')
    parser.add_argument('--mode', dest='mode', type=str,
                        default='train', help='train or val')
    parser.add_argument('--dataset', dest='dataset', type=str,
                        default='nextqa', help='nextqa or causalchaos')
    parser.add_argument('--split', dest='split', type=str,
                        default='UD', help='UD, PS or UN')
    parser.add_argument('--version', dest='version', type=str,
                        default='A', help='A or E')
    args = parser.parse_args()
    set_gpu_devices(args.gpu)
    main(args)
