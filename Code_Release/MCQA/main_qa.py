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
import eval_mc


def main(args):

    mode = args.mode
    if mode == 'train':
        batch_size = 64
        num_worker = 8
    else:
        batch_size = 4
        num_worker = 4
        
    video_feature_cache = '../data/feats/cache/'
    video_feature_path = '../data/feats/'

    dataset = args.dataset
    split = args.split
    diff = args.diff
    sample_list_path = 'dataset/{}'.format(dataset)
    #print('dataset/{}/vocab.pkl'.format(dataset))
    vocab = pkload('dataset/{}/vocab.pkl'.format(dataset))
    glove_embed = 'dataset/{}/glove_embed.npy'.format(dataset)


    use_bert = False #Otherwise GloVe
    #checkpoint_path = 'models'
    checkpoint_path = f'models/{dataset}'
    model_type = 'BlindQA' #(EVQA, STVQA, CoMem, HME, HGA)
    #model_prefix= 'bert-ft-h256'
    model_prefix= 'glove'

    print('Dataset: {} | Diff:{} |Levels: {} | Bert? : {}'.format(dataset, diff, split, use_bert))

    vis_step = 106
    lr_rate = 5e-5 if use_bert else 1e-4
    epoch_num = 50

    data_loader = dataloader.QALoader(batch_size, num_worker, video_feature_path, video_feature_cache,
                                      sample_list_path, split, vocab, use_bert, dataset, diff, True, False)
    
    train_loader, val_loader = data_loader.run(mode=mode)

    vqa = VideoQA(vocab, train_loader, val_loader, glove_embed, use_bert, checkpoint_path, model_type, model_prefix,
                  vis_step,lr_rate, batch_size, epoch_num, dataset, split, diff)

    ep = 41
    acc = 30.80
    model_file = f'{dataset}-{diff}-{split}-{model_type}-{model_prefix}-{ep}-{acc:.2f}.ckpt'

    if mode != 'train':
        result_file = f'results/{dataset}-{diff}-{split}-{model_type}-{model_prefix}-{mode}-shuffle.json'
        if args.dataset == 'causalchaos':
            vqa.predict(model_file, result_file)
            eval_mc.main(result_file, dataset, split, diff, mode)
        else: 
            vqa.predict_next(model_file, result_file)
            eval_mc.main(result_file, dataset, split, diff, mode)
    else:
        #Model for resume-training.
        model_file = f'{dataset}-{diff}-{split}-{model_type}-{model_prefix}-0-00.00.ckpt'
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
    parser.add_argument('--diff', dest='diff', type=str,
                        default='A', help='A or E')
    args = parser.parse_args()

    main(args)
