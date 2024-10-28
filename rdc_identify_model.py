import torch
import pickle
import time 
import logging
import argparse
import pprint
import os
import yaml

from model_def import  build_model_class
from data_utils import CustomDataset
from model_train import train_a_model


def retrain_with_stop_words(custom_args, snp_token_str_rm):
    logging.info('add stop_words to retrain')
    logging.info('--------------------------------')
    train_conf = custom_args.conf['train_param']
    data_conf = custom_args.conf['data']
    logging.info(pprint.pformat(train_conf))

    device = torch.device(train_conf['cuda'] if torch.cuda.is_available() else "cpu")
    
    
    dataset_with_stop_words = CustomDataset(data_conf['train_data']
                                            , stop_words_list=snp_token_str_rm
                                            , mit_flag = data_conf['mit_flag'])
    
    model_class = build_model_class(train_conf['model_flag'])
    model = train_a_model(device, model_class, dataset_with_stop_words, train_conf)
    model = model.to(torch.device('cpu'))
    logging.info('save final model and vect')
    
    utils_conf = custom_args.conf['utils']
    torch.save({
        'model_def' : model
        ,'model_param' : model.state_dict()
        ,'model_def_name' :  type(model).__name__
        }
        , os.path.join(utils_conf['checkpoint'], 'model.pt'))
    
    dataset_with_stop_words.save_vocabulary(os.path.join( utils_conf['checkpoint'], 'token_vec.pickle'))


def main_proc(custom_args):
    conf = custom_args.conf
    # 
    logging.info('------\n-----begin a new model train----\n---------')   
    
    def train_with_stop_snp(custom_args):
        stop_words_info = pickle.load(open(conf['data']['stop_snps'], 'rb'))
        retrain_with_stop_words(custom_args, stop_words_info['snp_token_str_rm'])

    method_dict = {
        'train_with_stop_snp': train_with_stop_snp
        ,'train': lambda custom_args : retrain_with_stop_words(custom_args, [])
    }

    func = method_dict.get(custom_args.method, lambda x: print('not implement'))
    func(custom_args)

    logging.info('------\n-----end a new model train----\n---------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='dg_classify')

    parser.add_argument('--method', type=str, default='train_with_stop_snp')
    parser.add_argument('--conf', type=str, default='dg_rdg/config_test.yml')
    parser.add_argument('--test_flag', type=int, default=0)

    args = parser.parse_args()

    with open(args.conf, 'r') as _file:
        conf = yaml.safe_load(_file)
    print(conf)
    args.conf = conf

    model_dir = conf['utils']['model_dir']
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    checkpoint = conf['utils']['checkpoint']
    if not os.path.exists(checkpoint): os.makedirs(checkpoint)

    logging.basicConfig(filename=conf['utils']['logger_file']
                        , level=conf['utils']['logging_level']
                        , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main_proc(args)

