'''
Created on May 3, 2018
public version for Deep Fusion Model. I merge all the functions into one single file so that readers can follow the whole logics directly.

Version 8 

update message:  make linear part into attention ensemble
update message:  add deep interest network

update message:
add valid file logic
att_feature idx @ 11,12,13 indicate position,dayOfWeek,hour. Add normalization for them

update message: add DNN, add in low-level

'''

import tensorflow as tf
import math
from time import clock
import numpy as np
import os
import pickle
import sys
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from datetime import datetime
import logging
import platform
import random
import codecs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers.python.layers import utils
from myutils import *


'''
before running this model, the following numbers should be set according to your dataset:
'''
DOMAIN_FIELD_COUNT = [4,7,4,3,5] #[4,7,4,3,4]
ATTENTION_FEATURE_LEN = 14 #96  auxilliary features
FEATURE_COUNT = 700000  
FIELD_COUNT=24




params = {
        'predict_only':False,
        'model_load_path':'model_output/AttentiveUserEmb/model_save/attEmbDNN-350',
        'output_predictions':False,
        'save_model':True,
        'log_path': 'model_output/DFM/normal/',
        'model_path': 'model_output/DFM/normal',
        'graph_summary_path': 'model_output/AttentiveUserEmb/' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S'),
        'reg_w_linear': 0.002,  'reg_w_nn': 0.001,  #0.001
        'reg_w_l1': 0.0002, 'reg_w_emb':0.002,
        'reg_w_nn_MF':0.001,'reg_w_emb_MF':0.002,
        'init_value': 0.1, ## 0.001
        'init_value_w':0.01,
        'init_value_b':0.01,
        'layer_sizes': [[32,32],[]] ,#[[32,32],[32],[]]   [[32]]   [[32,32,32],[32],[]]  #depths for inception module
        'activations':['tanh','tanh'],#
        'dnn_layer_sizes':[32,32],
        'fusion_method': 'linear_tran',
        'dropout_probs':1.0,
        'eta': 0.1,
        'att_tao': 2,
        'n_epoch': 200,  # 500
        'batch_size': 1024,
        'inside_train_ite_per_test':50,
        'dim': 32,
        # 'log_path': 'logs/' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S'),
        #'warm_up_file':'data/AttentiveUserEmb/filtered/sample0.2_news_0.txt',
        'train_file':  'data/AttentiveUserEmb/filtered/DFM/sample0.2_news_train.txt', 
        'valid_file':    'data/AttentiveUserEmb/filtered/DFM/sample0.2_news_valid.txt',   
        'test_file':    'data/AttentiveUserEmb/filtered/DFM/sample0.2_news_test.txt', 
        'is_use_linear_part':False,
        'is_use_dnn_part':False,
        'is_use_MF_part':True,
        'is_use_userbasic_part':True,
        'is_use_news_part':True,
        'is_use_browsing_part':True,
        'is_use_search_part':True,
        'is_use_userall_part':True,
        'is_attention_enabled':False,
        'is_use_itembasic_part':True, 
        'is_multi_level':True,
        'learning_rate':0.001, # 0.0005
        'loss': 'log_loss', # [focal_log_loss, cross_entropy_loss, square_loss, log_loss]
        'optimizer':'adam', # [adam, ftrl, sgd]
        'clean_cache':False,
        'att_layer_size':[],  # [32,32] set attention network
        'activations_att':['tanh','tanh'],
        'enable_batch_norm':False,
        'test_mode':'inside_train_epoch',
        'focal_gamma':1,
        'focal_alpha':1,
        'is_ADIN_ensebled':False, # deep interest network https://arxiv.org/abs/1706.06978
        'metrics': [
             {'name': 'auc_ind'},
             {'name': 'auc'}
             ,{'name': 'log_loss'}
            #, {'name': 'precision', 'k': 1}
            #, {'name': 'map', 'k': 1}
            #, {'name': 'map', 'k': 2}
            #, {'name': 'map', 'k': 3}
            #, {'name': 'recall', 'k': 1}
            #, {'name': 'recall', 'k': 2}
            #, {'name': 'recall', 'k': 3}
            #, {'name': 'hit', 'k': 1}
            #, {'name': 'hit', 'k': 2}
            #, {'name': 'hit', 'k': 3}
            #, {'name': 'ndcg', 'k': 1}
            #, {'name': 'ndcg', 'k': 2}
            #, {'name': 'ndcg', 'k': 3}
        ]
    }






class PlaceHolderGather:
    def __init__(self):
        self._indices = tf.placeholder(tf.int64, shape=[None, 2])
        self._values = tf.placeholder(tf.float32, shape=[None])
        self._shape = tf.placeholder(tf.int64, shape=[2])
        self._field2feature_indices = tf.placeholder(tf.int64, shape=[None, 2])
        self._field2feature_values = tf.placeholder(tf.int64, shape=[None])
        self._field2feature_weights = tf.placeholder(tf.float32, shape=[None])
        self._field2feature_shape = tf.placeholder(tf.int64, shape=[2])
        
        self._itememb_lookup_indices = tf.placeholder(tf.int64, shape=[None])

        

          

def init_input_data_holder(res):
    res['attention_features']=[]
    res['comments']=[]
    res['labels']=[]
    res['userbasic'] = []
    res['news']=[]
    res['browsing']=[]
    res['search']=[]      
    res['itembasic']=[]  
    res['qids']=[]
    res['dnnfeature']=[]
    res['userfeatureall']=[]

# format of input data: label a1,a2,a3,...an(for attention network input) basic_user_featture(like 0:1:2:0.5) user_domain1_feature user_domain2_feature ... basic_item_feautre
# basic_user_featture: idx = 0  
# user news domain : idx = 1 
# user_browsing_domain : idx = 2 
# user_search_domain : idx = 3
# basic_item_feature : idx = 4
def load_data_from_file_batching(file, batch_size):

    cnt = 0
    res = {}
    
    init_input_data_holder(res)

    
    with codecs.open(file, 'r', 'utf-8') as rd:
        while True:
            line = rd.readline()#.strip()
            if not line:
                break
            cnt += 1
            if '#' in line:
                punc_idx = line.index('#')
            else:
                punc_idx = len(line)

            before_comment_line = line[:punc_idx].strip()
            after_comment_line = line[punc_idx + 1:].strip()
            res['comments'].append(after_comment_line)

            cols = before_comment_line.split(' ')
            label = float(cols[0])
            if label > 0:
                label = 1
            else:
                label = 0
            
            res['labels'].append([label])  
            

            cur_feature_list = [[],[],[],[],[]]
            
            dnn_features = []
            userall_features = []
            att_idx = 0  
            cur_attention_features = []
            for col in cols[1].split(','):
                cur_value = float(col)
                cur_attention_features.append(cur_value)
                dnn_features.append([0, att_idx, cur_value])
                att_idx +=1

            res['attention_features'].append(cur_attention_features)
            
            
            for col in cols[2:]:                
                if col.startswith('qid:'):
                    res['qids'].append(col)
                    continue
                    
                tokens = col.split(':') 
                cur_field_id = int(tokens[1])-1
                cur_domain_id = int(tokens[0])
                cur_feature_id = int(tokens[2]) -1 
                cur_tuple = ( cur_field_id, cur_feature_id , float(tokens[3]) )
                
                cur_feature_list[cur_domain_id].append(cur_tuple)
                
                field_idx = 0 
                for i in range(cur_domain_id):
                    field_idx+=DOMAIN_FIELD_COUNT[i]
                field_idx += int(tokens[1])
                dnn_features.append([field_idx, cur_feature_id, float(tokens[3])])
                if cur_domain_id != 4:
                    userall_features.append([field_idx - 1, cur_feature_id, float(tokens[3])])


            res['userbasic'].append(cur_feature_list[0]) 
            res['news'].append(cur_feature_list[1])
            res['browsing'].append(cur_feature_list[2])
            res['search'].append(cur_feature_list[3])
            res['itembasic'].append(cur_feature_list[4])
            res['dnnfeature'].append(dnn_features)
            res['userfeatureall'].append(userall_features)

                
            if cnt == batch_size:
                yield res 
                init_input_data_holder(res)
                cnt=0
    if cnt > 0:
        yield res

def prepare_data_4_sp_domainly(domain_features,  instance_cnt, field_cnt, name): 
    indices = []
    values = [] 
    shape = [instance_cnt, FEATURE_COUNT]
    field2feature_indices = []
    field2feature_values = []
    field2feature_weights = []
    filed2feature_shape = [instance_cnt * field_cnt, -1]

    lastidx = 0 
    for i in range(instance_cnt):
        m = len(domain_features[i])
        field2features_dic = {}
        for j in range(m):
            
            ##-- debugging only
            if name =='itembasic' and (domain_features[i][j][0]==4 ):
                domain_features[i][j]=(domain_features[i][j][0],999,0)
            
            indices.append([i, domain_features[i][j][1]])
            values.append(domain_features[i][j][2])

            #feature_indices.append(features[i][j][1])
            if domain_features[i][j][0] not in field2features_dic:
                field2features_dic[domain_features[i][j][0]] = 0
            else:
                field2features_dic[domain_features[i][j][0]] += 1
            cur_idx = i * field_cnt + domain_features[i][j][0] 
            if lastidx<cur_idx-1 or lastidx>cur_idx:
                print(name, ' lastidx ',lastidx, ' curidx ',cur_idx, ' fieldidx ',domain_features[i][j][0], 'features ', domain_features[i] )
            if lastidx<cur_idx:
                lastidx = cur_idx
            field2feature_indices.append([i * field_cnt + domain_features[i][j][0], field2features_dic[domain_features[i][j][0]]])
            field2feature_values.append(domain_features[i][j][1])
            field2feature_weights.append(domain_features[i][j][2] ) 
            if filed2feature_shape[1] < field2features_dic[domain_features[i][j][0]]:
                filed2feature_shape[1] = field2features_dic[domain_features[i][j][0]]
    filed2feature_shape[1] += 1

    sorted_index = sorted(range(len(field2feature_indices)), key=lambda k: (field2feature_indices[k][0],field2feature_indices[k][1]))



    res = {}
    res['indices'] = np.asarray(indices, dtype=np.int64)
    res['values'] = np.asarray(values, dtype=np.float32)

    res['shape'] = np.asarray(shape, dtype=np.int64)

    res['field2feature_indices'] = np.asarray(field2feature_indices, dtype=np.int64)[sorted_index]
    res['field2feature_values'] = np.asarray(field2feature_values, dtype=np.int64)[sorted_index]
    res['field2feature_weights'] = np.asarray(field2feature_weights, dtype=np.float32)[sorted_index]
    res['filed2feature_shape'] = np.asarray(filed2feature_shape, dtype=np.int64)

    return res

def prepare_data_4_sp(raw_input_dict):    
    instance_cnt = len(raw_input_dict['labels'])
    res = {}
    res['userbasic'] = prepare_data_4_sp_domainly(raw_input_dict['userbasic'],   instance_cnt, DOMAIN_FIELD_COUNT[0], 'userbasic')
    res['news'] = prepare_data_4_sp_domainly(raw_input_dict['news'],   instance_cnt, DOMAIN_FIELD_COUNT[1], 'news')
    res['browsing'] = prepare_data_4_sp_domainly(raw_input_dict['browsing'],   instance_cnt, DOMAIN_FIELD_COUNT[2], 'browsing')
    res['search'] = prepare_data_4_sp_domainly(raw_input_dict['search'],   instance_cnt, DOMAIN_FIELD_COUNT[3], 'search')
    res['itembasic'] = prepare_data_4_sp_domainly(raw_input_dict['itembasic'],  instance_cnt, DOMAIN_FIELD_COUNT[4], 'itembasic')    
    res['labels'] = np.asarray(raw_input_dict['labels'], dtype=np.float32)    
    res['att_features'] = np.asarray(raw_input_dict['attention_features'], dtype=np.float32)
    res['qids']=raw_input_dict['qids']
    res['dnn_features'] = prepare_data_4_sp_dnn(raw_input_dict['dnnfeature'], instance_cnt)
    res['userfeatureall'] = prepare_data_4_sp_domainly(raw_input_dict['userfeatureall'], instance_cnt,  DOMAIN_FIELD_COUNT[0] + DOMAIN_FIELD_COUNT[1] + DOMAIN_FIELD_COUNT[2] + DOMAIN_FIELD_COUNT[3], 'userfeatureall')
    return res


def prepare_data_4_sp_dnn( features, instance_cnt): 
    
    indices = []
    values = []
    values_2 = []
    shape = [instance_cnt, FEATURE_COUNT]  
    field2feature_indices = []
    field2feature_values = []
    field2feature_weights = []
    filed2feature_shape = [instance_cnt * FIELD_COUNT, -1]  ##--

    lastidx = 0 
    for i in range(instance_cnt):
        m = len(features[i])
        field2features_dic = {}
        for j in range(m):
            indices.append([i, features[i][j][1]])
            values.append(features[i][j][2])
            values_2.append(features[i][j][2] * features[i][j][2])
            #feature_indices.append(features[i][j][1])
            if features[i][j][0] not in field2features_dic:
                field2features_dic[features[i][j][0]] = 0
            else:
                field2features_dic[features[i][j][0]] += 1
            cur_idx = i * FIELD_COUNT + features[i][j][0]  
            if lastidx<cur_idx:
                lastidx = cur_idx
            field2feature_indices.append([i * FIELD_COUNT + features[i][j][0], field2features_dic[features[i][j][0]]])
            field2feature_values.append(features[i][j][1])
            field2feature_weights.append(features[i][j][2] ) 
            if filed2feature_shape[1] < field2features_dic[features[i][j][0]]:
                filed2feature_shape[1] = field2features_dic[features[i][j][0]]
    filed2feature_shape[1] += 1

    sorted_index = sorted(range(len(field2feature_indices)), key=lambda k: (field2feature_indices[k][0],field2feature_indices[k][1]))
 

    res = {}
    res['indices'] = np.asarray(indices, dtype=np.int64)
    res['values'] = np.asarray(values, dtype=np.float32)
    res['values2'] = np.asarray(values_2, dtype=np.float32)
    res['shape'] = np.asarray(shape, dtype=np.int64)
    res['field2feature_indices'] = np.asarray(field2feature_indices, dtype=np.int64)[sorted_index]
    res['field2feature_values'] = np.asarray(field2feature_values, dtype=np.int64)[sorted_index]
    res['field2feature_weights'] = np.asarray(field2feature_weights, dtype=np.float32)[sorted_index]
    res['filed2feature_shape'] = np.asarray(filed2feature_shape, dtype=np.int64)

    return res



def update_feature_4_ADIN(domain_dict):
    domain_dict['itememb_lookup_indices'] = np.asarray( [t[0] for t in domain_dict['indices']], dtype=np.int32)

def load_data_cache(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def pre_build_data_cache(infile, outfile, batch_size):
    wt = open(outfile, 'wb') 
    for raw_input_data in load_data_from_file_batching(infile, batch_size): 
        input_in_sp = prepare_data_4_sp(raw_input_data)
        pickle.dump(input_in_sp, wt)
    wt.close()


 
def single_run():
    
    logging_filename = params['log_path'] + platform.node() + '__' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S') + '.log'
    check_dir_of_file(logging_filename)
    logger = logging.getLogger(__name__)
    for hander in logger.handlers[:]:
        logger.removeHandler(hander)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logging_filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    handler02 = logging.StreamHandler()
    handler02.setLevel(logging.INFO)
    handler02.setFormatter(formatter)
    logger.addHandler(handler02)

    logger.info('\n\n')
    logger.info(params)
    logger.info('\n\n')


    pre_build_data_cache_if_need(params['train_file'], params['batch_size'], try_get_params(params, 'clean_cache', False))
    pre_build_data_cache_if_need(params['valid_file'], params['batch_size'], try_get_params(params, 'clean_cache', False))
    pre_build_data_cache_if_need(params['test_file'], params['batch_size'], try_get_params(params, 'clean_cache', False))

    params['train_file'] = params['train_file'].replace('.csv','.pkl').replace('.txt','.pkl')
    params['valid_file'] = params['valid_file'].replace('.csv','.pkl').replace('.txt','.pkl')
    params['test_file'] = params['test_file'].replace('.csv','.pkl').replace('.txt','.pkl')
    
    # a small file for initializing models (the same format with training file, but different instances)
    if 'warm_up_file' in params:
        pre_build_data_cache_if_need(params['warm_up_file'], params['batch_size'], try_get_params(params, 'clean_cache', False))
        params['warm_up_file'] = params['warm_up_file'].replace('.csv','.pkl').replace('.txt','.pkl')
    
  
    print('start single_run')
    
    tf.reset_default_graph()

    n_epoch = params['n_epoch']
    batch_size = params['batch_size']
    
    params['num_attentional_features'] = ATTENTION_FEATURE_LEN
    _att_features = tf.placeholder(tf.float32, shape=[None, ATTENTION_FEATURE_LEN], name='attention_features')
 
    _y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

    place_holders = {}
    place_holders['userbasic'] = PlaceHolderGather()
    place_holders['itembasic'] = PlaceHolderGather()
    place_holders['news'] = PlaceHolderGather()
    place_holders['browsing'] = PlaceHolderGather()
    place_holders['search'] = PlaceHolderGather()
    place_holders['dnn'] = PlaceHolderGather()
    place_holders['userfeatureall'] = PlaceHolderGather()
    
    _train_phase = tf.placeholder(tf.bool, name = 'train_phase')
    _train_init_done = tf.placeholder(tf.bool, name='train_init_done')
    _dropout_value = tf.placeholder(tf.float32, name = 'dropout_value')

    train_step, loss, error, preds, merged_summary  = build_model(_att_features, _y, place_holders, _dropout_value,  _train_phase, _train_init_done)


    if params['save_model']:
        saver = tf.train.Saver(max_to_keep=50)
        
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    summary_writer = tf.summary.FileWriter(params['graph_summary_path'], graph=sess.graph)

    glo_ite = 0

    last_best_auc = None
    max_stop_grow_torrelence = 12
    stop_grow_cnt = 0

    if params['predict_only']:
        saver.restore(sess, params['model_load_path'])
        query2res_test = predict_test_file(preds, sess, params['test_file'],
                                                    _att_features, place_holders, _y, _train_phase ,  _train_init_done , 
                                                    0, batch_size, 'test', params['model_path'], params['output_predictions'], _dropout_value
                                                    )  
        metrics_test = compute_metric(query2res_test)
        metrics_strs=[]
        for metric_name in sorted(metrics_test):
            metrics_strs.append('{0} is {1:.4f}'.format(metric_name, metrics_test[metric_name]))
        print(' '.join(metrics_strs))
        return 
        
    start = clock()
    for eopch in range(n_epoch):
        #iteration = -1
        
        time_load_data, time_sess = 0, 0
        time_cp02 = clock()
        
        train_loss_per_epoch = 0
        train_error_per_epoch = 0 
        
        if stop_grow_cnt>max_stop_grow_torrelence:
            break 
       
        print(' new eopch idx: ' + str(eopch))
        for training_input_in_sp in load_data_cache(params['train_file'] if eopch>2 else (params['warm_up_file'] if 'warm_up_file' in params else params['train_file'])):
            
            if try_get_params(params,'is_ADIN_ensebled', False):
                update_feature_4_ADIN(training_input_in_sp['news'])
                update_feature_4_ADIN(training_input_in_sp['browsing'])
                update_feature_4_ADIN(training_input_in_sp['search'])
                update_feature_4_ADIN(training_input_in_sp['userfeatureall'])
            
            if training_input_in_sp['userbasic']['shape'][0]< params['batch_size']:
                continue
            
            #if 'warm_up_file' in params and eopch<=2 and random.random()<0.3:
            #   continue
            
            #print('training_input_in_sp=',training_input_in_sp)
            #sys.exit()
            time_cp01 = clock()
            time_load_data += time_cp01 - time_cp02
            #iteration += 1
            glo_ite += 1
            _,  cur_loss, cur_error, summary  = sess.run([train_step,  loss, error, merged_summary], feed_dict={
                _att_features:   training_input_in_sp['att_features'],
                _y: training_input_in_sp['labels'],                         
                place_holders['userbasic']._indices: training_input_in_sp['userbasic']['indices'], 
                place_holders['userbasic']._values: training_input_in_sp['userbasic']['values'],
                place_holders['userbasic']._shape: training_input_in_sp['userbasic']['shape'],                   
                place_holders['userbasic']. _field2feature_indices: training_input_in_sp['userbasic']['field2feature_indices'],
                place_holders['userbasic']._field2feature_values: training_input_in_sp['userbasic']['field2feature_values'],
                place_holders['userbasic']. _field2feature_weights: training_input_in_sp['userbasic']['field2feature_weights'],
                place_holders['userbasic']. _field2feature_shape: training_input_in_sp['userbasic']['filed2feature_shape'],
                
                place_holders['itembasic']._indices: training_input_in_sp['itembasic']['indices'], 
                place_holders['itembasic']._values: training_input_in_sp['itembasic']['values'],
                place_holders['itembasic']._shape: training_input_in_sp['itembasic']['shape'],                   
                place_holders['itembasic']. _field2feature_indices: training_input_in_sp['itembasic']['field2feature_indices'],
                place_holders['itembasic']._field2feature_values: training_input_in_sp['itembasic']['field2feature_values'],
                place_holders['itembasic']. _field2feature_weights: training_input_in_sp['itembasic']['field2feature_weights'],
                place_holders['itembasic']. _field2feature_shape: training_input_in_sp['itembasic']['filed2feature_shape'],
                
                place_holders['news']._indices: training_input_in_sp['news']['indices'], 
                place_holders['news']._values: training_input_in_sp['news']['values'],
                place_holders['news']._shape: training_input_in_sp['news']['shape'],                   
                place_holders['news']. _field2feature_indices: training_input_in_sp['news']['field2feature_indices'],
                place_holders['news']._field2feature_values: training_input_in_sp['news']['field2feature_values'],
                place_holders['news']. _field2feature_weights: training_input_in_sp['news']['field2feature_weights'],
                place_holders['news']. _field2feature_shape: training_input_in_sp['news']['filed2feature_shape'],
                place_holders['news']._itememb_lookup_indices: training_input_in_sp['news']['itememb_lookup_indices'] if try_get_params(params,'is_ADIN_ensebled',False) else [],
                
                place_holders['browsing']._indices: training_input_in_sp['browsing']['indices'], 
                place_holders['browsing']._values: training_input_in_sp['browsing']['values'],
                place_holders['browsing']._shape: training_input_in_sp['browsing']['shape'],                   
                place_holders['browsing']. _field2feature_indices: training_input_in_sp['browsing']['field2feature_indices'],
                place_holders['browsing']._field2feature_values: training_input_in_sp['browsing']['field2feature_values'],
                place_holders['browsing']. _field2feature_weights: training_input_in_sp['browsing']['field2feature_weights'],
                place_holders['browsing']. _field2feature_shape: training_input_in_sp['browsing']['filed2feature_shape'],
                place_holders['browsing']._itememb_lookup_indices: training_input_in_sp['browsing']['itememb_lookup_indices'] if try_get_params(params,'is_ADIN_ensebled',False) else [],
                
                place_holders['search']._indices: training_input_in_sp['search']['indices'], 
                place_holders['search']._values: training_input_in_sp['search']['values'],
                place_holders['search']._shape: training_input_in_sp['search']['shape'],                   
                place_holders['search']. _field2feature_indices: training_input_in_sp['search']['field2feature_indices'],
                place_holders['search']._field2feature_values: training_input_in_sp['search']['field2feature_values'],
                place_holders['search']. _field2feature_weights: training_input_in_sp['search']['field2feature_weights'],
                place_holders['search']. _field2feature_shape: training_input_in_sp['search']['filed2feature_shape'],    
                place_holders['search']._itememb_lookup_indices: training_input_in_sp['search']['itememb_lookup_indices'] if try_get_params(params,'is_ADIN_ensebled',False) else [],
                         
                place_holders['dnn']._indices: training_input_in_sp['dnn_features']['indices'], 
                place_holders['dnn']._values: training_input_in_sp['dnn_features']['values'],
                place_holders['dnn']._shape: training_input_in_sp['dnn_features']['shape'],                   
                place_holders['dnn']. _field2feature_indices: training_input_in_sp['dnn_features']['field2feature_indices'],
                place_holders['dnn']._field2feature_values: training_input_in_sp['dnn_features']['field2feature_values'],
                place_holders['dnn']. _field2feature_weights: training_input_in_sp['dnn_features']['field2feature_weights'],
                place_holders['dnn']. _field2feature_shape: training_input_in_sp['dnn_features']['filed2feature_shape'],  

                place_holders['userfeatureall']._indices: training_input_in_sp['userfeatureall']['indices'], 
                place_holders['userfeatureall']._values: training_input_in_sp['userfeatureall']['values'],
                place_holders['userfeatureall']._shape: training_input_in_sp['userfeatureall']['shape'],                   
                place_holders['userfeatureall']. _field2feature_indices: training_input_in_sp['userfeatureall']['field2feature_indices'],
                place_holders['userfeatureall']._field2feature_values: training_input_in_sp['userfeatureall']['field2feature_values'],
                place_holders['userfeatureall']. _field2feature_weights: training_input_in_sp['userfeatureall']['field2feature_weights'],
                place_holders['userfeatureall']. _field2feature_shape: training_input_in_sp['userfeatureall']['filed2feature_shape'],    
                place_holders['userfeatureall']._itememb_lookup_indices: training_input_in_sp['userfeatureall']['itememb_lookup_indices'] if try_get_params(params,'is_ADIN_ensebled',False) else [],
                                         
                _train_phase : True, _train_init_done: True, _dropout_value: params['dropout_probs']
            })


            time_cp02 = clock()

            time_sess += time_cp02 - time_cp01

            train_loss_per_epoch += cur_loss
            train_error_per_epoch += cur_error
            

            summary_writer.add_summary(summary, glo_ite)
            end = clock()
  
            if try_get_params(params, 'test_mode', 'inside_train_epoch') == 'inside_train_epoch': # since this branch works better, I remove the code for the other branch 
                if glo_ite % try_get_params(params, 'inside_train_ite_per_test', 50 ) == 0:
                    model_path = params['model_path'] + "/attEmbDNN"# + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S')                    
                    if params['save_model']:
                        os.makedirs(params['model_path'], exist_ok=True)
                        saver.save(sess, model_path, global_step=glo_ite)     
                               
                    query2res_valid = predict_test_file(preds, sess, params['valid_file'],
                                                    _att_features, place_holders, _y, _train_phase ,   _train_init_done , 
                                                    eopch, batch_size, 'test', model_path, params['output_predictions'],_dropout_value
                                          )
                    
                    query2res_test = predict_test_file(preds, sess, params['test_file'],
                                                    _att_features, place_holders, _y, _train_phase ,   _train_init_done , 
                                                    eopch, batch_size, 'test', model_path, params['output_predictions'],_dropout_value
                                                    )
                    test_end = clock()
                    
                    metrics_valid = compute_metric(query2res_valid)
                    metrics_test = compute_metric(query2res_test)
                    
                    eva_end = clock()
                    
                    metrics_strs = []
                    metrics_strs.append('valid: ')
                    auc = 0
                    for metric_name in sorted(metrics_valid):
                        metrics_strs.append('{0} is {1:.4f}'.format(metric_name, metrics_valid[metric_name]))
                        if metric_name == 'auc':
                            auc = metrics_valid['auc']
                    metrics_strs.append('\ttest: ')
                    for metric_name in sorted(metrics_test):
                        metrics_strs.append('{0} is {1:.4f}'.format(metric_name, metrics_test[metric_name]))

                        
                    if last_best_auc is None or auc>last_best_auc:
                        last_best_auc = auc 
                        stop_grow_cnt = 0 
                    else:
                        stop_grow_cnt+=1
                        
                    res_str = ' ,'.join(metrics_strs) + ', at epoch {0:d}, train_time: {1:.4f}min, train_loss is {2:.2f} test_time: {3:.4f}min eva_time: {4:.4f}min'.format(glo_ite, (end -start) / 60.0, train_loss_per_epoch, (test_end -end) / 60.0, (eva_end-test_end)/60.0)
   
                    logger.info(res_str)
                    start = clock()
                    
                    if stop_grow_cnt>max_stop_grow_torrelence:
                        break 
        
         

    summary_writer.close()

def nonattentional_ops(_att_features, nn_output_userbasic, nn_output_news, nn_output_browsing, nn_output_search, nn_output_userall,
                   init_value, _train_phase, model_w_nn_params, model_linear_params, linear_part, bias, dnn_output, att_output):
    merge_user_embedding = tf.add_n([nn_output_userbasic,nn_output_news,nn_output_browsing,nn_output_search,nn_output_userall], name = 'avg_user_embedding')
    return merge_user_embedding, att_output, linear_part, bias, dnn_output 
    
def attentional_ops(_att_features, nn_output_userbasic, nn_output_news, nn_output_browsing, nn_output_search,nn_output_userall,
                   init_value, _train_phase, model_w_nn_params, model_linear_params, linear_part, bias, dnn_output, dnn_output_raw,  att_output):
    att_nn_input = tf.concat([_att_features, nn_output_userbasic, nn_output_news, nn_output_browsing, nn_output_search, nn_output_userall, linear_part, dnn_output_raw], 1, name = 'attention/concated_emb_input')

    att_hidden_nn_layers = []
    att_hidden_nn_layers.append(att_nn_input)
    att_layer_idx = 0
    att_last_layer_size = ATTENTION_FEATURE_LEN + 5 * get_final_emb_dim() + 1 + params['dnn_layer_sizes'][-1]  ##-- actually, it should be the size of last hidden layer instead of params['dim']
    for layer_size in params['att_layer_size']:
        cur_w_att_layer = tf.Variable(
            tf.truncated_normal([att_last_layer_size, layer_size], stddev= init_value/math.sqrt(float(att_last_layer_size)), mean=0),
            name = 'attention/w_nn_layer_'+str(att_layer_idx), dtype=tf.float32                  
                                )
    
        cur_b_att_layer = tf.Variable(tf.truncated_normal([layer_size], stddev=params['init_value_b'], mean = 0 ), name = 'attention/b_nn_layer_' + str(att_layer_idx) )
        
        if False: #_train_phase ==True and params['att_dropout_probs'][att_layer_idx]<1.0:
            cur_hidden_att_input = tf.nn.dropout(att_hidden_nn_layers[att_layer_idx], params['att_dropout_probs'][att_layer_idx], name = 'attention/hidden_input_'+ str(att_layer_idx))
        else:
            cur_hidden_att_input = att_hidden_nn_layers[att_layer_idx]
        
        if (try_get_params(params, 'enable_batch_norm', False)):
            cur_hidden_att_input = tf.contrib.layers.batch_norm(cur_hidden_att_input, 
                                      center=True, scale=True, 
                                      is_training=_train_phase)    
    
        cur_hidden_att_layer = tf.nn.xw_plus_b(cur_hidden_att_input, cur_w_att_layer, cur_b_att_layer, name = 'attention/hidden_raw_activation_'+ str(att_layer_idx))
    
        if params['activations_att'][att_layer_idx]=='tanh':
            cur_hidden_att_layer = tf.nn.tanh(cur_hidden_att_layer, name = 'attention/hidden_tanh_activation_'+ str(att_layer_idx))
        elif params['activations_att'][att_layer_idx]=='sigmoid':
            cur_hidden_att_layer = tf.nn.sigmoid(cur_hidden_att_layer, name = 'attention/hidden_sig_activation_'+ str(att_layer_idx))
        elif params['activations_att'][att_layer_idx]=='relu':
            cur_hidden_att_layer = tf.nn.relu(cur_hidden_att_layer, name = 'attention/hidden_relu_activation_'+ str(att_layer_idx))
        
        att_hidden_nn_layers.append(cur_hidden_att_layer)    
    
        att_layer_idx +=1 
        att_last_layer_size = layer_size
    
        model_w_nn_params.append(cur_w_att_layer)
        model_linear_params.append(cur_b_att_layer)
    

    w_att_output = tf.Variable(tf.truncated_normal([att_last_layer_size, 8], stddev=init_value, mean=0), name='attention/w_output', dtype=tf.float32)
    b_att_output =  tf.Variable(tf.truncated_normal([8], stddev=params['init_value_b'], mean=0), name='attention/b_output', dtype=tf.float32)
    att_output = tf.nn.xw_plus_b(att_hidden_nn_layers[-1], w_att_output, b_att_output, name='attention/raw_output')
    
    att_tao = try_get_params(params, 'att_tao', 1.0)        
    att_output = tf.nn.softmax(att_tao * att_output, name='attention/softmax_output')
    
    att_output_userbasic = tf.multiply(nn_output_userbasic, tf.reshape(att_output[:,0],[-1,1]), name = 'attention/userbasic_emb_output')
    att_output_news = tf.multiply(nn_output_news, tf.reshape(att_output[:,1],[-1,1]), name = 'attention/news_emb_output')
    att_output_browsing = tf.multiply(nn_output_browsing, tf.reshape(att_output[:,2],[-1,1]), name = 'attention/browsing_emb_output')
    att_output_search = tf.multiply(nn_output_search, tf.reshape(att_output[:,3],[-1,1]), name = 'attention/search_emb_output')
    att_output_userall = tf.multiply(nn_output_userall, tf.reshape(att_output[:,4],[-1,1]), name = 'attention/userall_emb_output') 
    
    ##-- testing only
    ## in the paper, linear part and dnn part do not have attentive weights
    linear_part = tf.multiply(linear_part, tf.reshape(att_output[:,5],[-1,1]), name = 'attention/linear_part_output')
    bias = tf.multiply(bias, tf.reshape(att_output[:,6],[-1,1]), name = 'attention/bias_part_output')
    dnn_output = tf.multiply(dnn_output, tf.reshape(att_output[:,7],[-1,1]), name = 'attention/dnn_part_output')
    
    merge_user_embedding = tf.add_n([att_output_userbasic,att_output_news,att_output_browsing,att_output_search,att_output_userall], name = 'attentive_user_embedding')
    
    return merge_user_embedding, att_output, linear_part, bias, dnn_output

def build_dnn_model(_train_phase , _place_holders, dropout_value, b_nn_params, w_nn_params, w_linear, w_fm):

    with tf.name_scope('DNN_submodule'):
        _x = tf.SparseTensor(_place_holders._indices, _place_holders._values, _place_holders._shape)  # m * FEATURE_COUNT sparse tensor
        
        nn_linear_output = tf.sparse_tensor_dense_matmul(_x, w_linear, name='contr_from_linear')
    
    
        init_value = params['init_value_w']
        dim = params['dim']
        layer_sizes = params['dnn_layer_sizes']
    
        
        w_fm_sparseIndexs = tf.SparseTensor(_place_holders._field2feature_indices, _place_holders._field2feature_values, _place_holders._field2feature_shape)
        w_fm_sparseWeights = tf.SparseTensor(_place_holders._field2feature_indices, _place_holders._field2feature_weights, _place_holders._field2feature_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(w_fm, w_fm_sparseIndexs,w_fm_sparseWeights,combiner="sum")
        w_fm_nn_input = tf.reshape(w_fm_nn_input_orgin, [-1, dim * FIELD_COUNT])
        
           
        
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        last_layer_size = FIELD_COUNT * dim
        layer_idx = 0
        
        
        for layer_size in layer_sizes:
            cur_w_nn_layer = tf.Variable(
                tf.truncated_normal([last_layer_size, layer_size], stddev=init_value / math.sqrt(float(10)), mean=0),
                    name='w_nn_layer' + str(layer_idx), dtype=tf.float32)
    
            cur_b_nn_layer = tf.Variable(tf.truncated_normal([layer_size], stddev=init_value, mean=0), name='b_nn_layer' + str(layer_idx))  
    
            cur_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx], cur_w_nn_layer, cur_b_nn_layer)
           
            if params['activations'][layer_idx]=='tanh':
                cur_hidden_nn_layer = tf.nn.tanh(cur_hidden_nn_layer)
            elif params['activations'][layer_idx]=='sigmoid':
                cur_hidden_nn_layer = tf.nn.sigmoid(cur_hidden_nn_layer)
            elif params['activations'][layer_idx]=='relu':
                cur_hidden_nn_layer = tf.nn.relu(cur_hidden_nn_layer)
           
    
            hidden_nn_layers.append(cur_hidden_nn_layer)
    
            layer_idx += 1
            last_layer_size = layer_size
    
    
            w_nn_params.append(cur_w_nn_layer)
            b_nn_params.append(cur_b_nn_layer)
        
        
        w_nn_output = tf.Variable(tf.truncated_normal([last_layer_size, 1], stddev=init_value, mean=0), name='dnn_output',
                                     dtype=tf.float32)
        nn_output = tf.matmul(hidden_nn_layers[-1], w_nn_output) 
        w_nn_params.append(w_nn_output)

    
    return nn_output, nn_linear_output, hidden_nn_layers[-1]


def build_model(_att_features, _y, place_holders, dropout_value,  _train_phase, _train_init_done):
    
    init_value =try_get_params( params, 'init_value', 0.01)
    dim = try_get_params( params, 'dim', 8)
    w_linear = tf.Variable(tf.truncated_normal([FEATURE_COUNT, 1], stddev=init_value, mean=0),  #tf.random_uniform([FEATURE_COUNT, 1], minval=-0.05, maxval=0.05),
                        name='w_linear', dtype=tf.float32)
    w_emb = tf.Variable(tf.truncated_normal([FEATURE_COUNT, dim], stddev=init_value / math.sqrt(float(dim)), mean=0),
                           name='w_emb', dtype=tf.float32)
    bias = tf.Variable(tf.truncated_normal([1], stddev=params['init_value_b'], mean=0), name='bias')
    

    model_latent_params = []
    model_linear_params = [] 
    model_w_nn_params = []
    model_linear_params.append(bias)
    model_linear_params.append(w_linear)
    model_latent_params.append(w_emb)
    
    w_emb_MF = tf.Variable(tf.truncated_normal([FEATURE_COUNT, dim], stddev=init_value / math.sqrt(float(dim)), mean=0),
                           name='w_emb_MF', dtype=tf.float32)
    model_latent_params_MF = []
    model_latent_params_MF.append(w_emb_MF)
    model_w_nn_params_MF = []    
    
    
    nn_output_userbasic, linear_pred_userbasic = build_domain_model(_train_phase ,place_holders['userbasic'], dropout_value,  DOMAIN_FIELD_COUNT[0], 'userbasic',  model_w_nn_params_MF, model_w_nn_params_MF , w_linear, w_emb_MF)
    nn_output_itembasic, linear_pred_itembasic = build_domain_model(_train_phase ,place_holders['itembasic'], dropout_value,  DOMAIN_FIELD_COUNT[4], 'itembasic',  model_w_nn_params_MF, model_w_nn_params_MF , w_linear, w_emb_MF)
    
    nn_output_news, linear_pred_news = build_domain_model(_train_phase ,place_holders['news'], dropout_value,  DOMAIN_FIELD_COUNT[1], 'news',  model_w_nn_params_MF, model_w_nn_params_MF , w_linear, w_emb_MF, item_emb = nn_output_itembasic)
    nn_output_browsing, linear_pred_browsing = build_domain_model(_train_phase ,place_holders['browsing'], dropout_value,  DOMAIN_FIELD_COUNT[2], 'browsing',  model_w_nn_params_MF, model_w_nn_params_MF , w_linear, w_emb_MF, item_emb = nn_output_itembasic)
    nn_output_search, linear_pred_search = build_domain_model(_train_phase ,place_holders['search'], dropout_value,  DOMAIN_FIELD_COUNT[3], 'search',  model_w_nn_params_MF, model_w_nn_params_MF, w_linear, w_emb_MF, item_emb = nn_output_itembasic)
    nn_output_userall, linear_pred_userall = build_domain_model(_train_phase ,place_holders['userfeatureall'], dropout_value,  DOMAIN_FIELD_COUNT[0] + DOMAIN_FIELD_COUNT[1] +DOMAIN_FIELD_COUNT[2] +DOMAIN_FIELD_COUNT[3], 'userfeatureall',  model_w_nn_params_MF, model_w_nn_params_MF, w_linear, w_emb_MF, item_emb = nn_output_itembasic)
    
    dnn_output, dnn_linear_output, dnn_output_raw = build_dnn_model(_train_phase , place_holders['dnn'], dropout_value, model_linear_params, model_w_nn_params, w_linear, w_emb)


    if try_get_params(params, 'is_use_linear_part', True):
        att_w_linear = tf.Variable(tf.truncated_normal([ATTENTION_FEATURE_LEN, 1], stddev=init_value, mean=0),  #tf.random_uniform([FEATURE_COUNT, 1], minval=-0.05, maxval=0.05),
                        name='attention/att_w_linear', dtype=tf.float32)
        att_linear_res = tf.matmul(_att_features, att_w_linear, name = 'attention/linear_result')
        model_linear_params.append(att_w_linear)
        linear_part = tf.add_n([att_linear_res, linear_pred_userbasic,linear_pred_itembasic,linear_pred_news,
                            linear_pred_browsing,linear_pred_search], name='linear_predictions')
        ##-- TEST
        linear_part = dnn_linear_output
    else:
        linear_part = tf.zeros([try_get_params(params, 'batch_size', 1024), 1], tf.float32, name='attention/linear_zeros')

    
    att_output = tf.zeros([try_get_params(params, 'batch_size', 1024), 6])
    
    att_ope_enable = tf.logical_and(tf.constant(try_get_params(params, 'is_attention_enabled', True), tf.bool) , tf.logical_or(_train_phase , _train_init_done))
    # dynamic graphing unsupport
    merge_user_embedding, att_output, linear_part, bias, dnn_output = \
        utils.smart_cond( try_get_params(params, 'is_attention_enabled', True), #att_ope_enable,
                lambda: attentional_ops(_att_features, nn_output_userbasic, nn_output_news, nn_output_browsing, nn_output_search,nn_output_userall,
                   init_value, _train_phase, model_w_nn_params, model_linear_params, linear_part, bias, dnn_output, dnn_output_raw, att_output),
                lambda: nonattentional_ops(_att_features, nn_output_userbasic, nn_output_news, nn_output_browsing, nn_output_search,nn_output_userall,
                   init_value, _train_phase, model_w_nn_params, model_linear_params, linear_part, bias, dnn_output, att_output)
                )
    
 
    
    user_item_prod =  tf.reduce_sum(tf.multiply( merge_user_embedding,nn_output_itembasic, name='user_item_product_raw') , 1, keep_dims=True, name = 'user_item_product')
    
    r'''  ##--
    if try_get_params(params,'is_use_linear_part',True):
        final_preds_raw = tf.add(bias, tf.add_n([linear_part,user_item_prod, dnn_output], name='final_predictions_raw00') 
                             ,name='final_predictions_raw')
    else:
        final_preds_raw = bias + user_item_prod + dnn_output
    '''  
    final_preds_raw = 0 
    if try_get_params(params,'is_use_linear_part',True):
        final_preds_raw += bias + linear_part
    if try_get_params(params, 'is_use_dnn_part', False):
        final_preds_raw += dnn_output
    if try_get_params(params, 'is_use_MF_part', True):
        final_preds_raw += user_item_prod
    
        
    final_preds = tf.sigmoid(final_preds_raw, name='final_predictions_sig')
    
    type_of_loss = try_get_params(params, 'loss', 'rmse')
    
    if type_of_loss == 'cross_entropy_loss':
        error = tf.reduce_mean(
                               tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(final_preds_raw, [-1]), labels=tf.reshape(_y, [-1])),
                               name='error/cross_entropy_loss'
                               )
    elif type_of_loss == 'square_loss' or type_of_loss == 'rmse':
        error = tf.reduce_mean(tf.squared_difference(final_preds, _y, name='error/squared_diff'), name='error/mean_squared_diff')
    elif type_of_loss == 'log_loss':
        error = tf.reduce_mean(tf.losses.log_loss(predictions=final_preds, labels=_y), name='error/mean_log_loss')
    elif type_of_loss == 'focal_log_loss'   :  # ICCV2017  focal loss
        error = my_focal_log_loss(_y, final_preds) 

    lambda_w_linear = tf.constant(try_get_params(params, 'reg_w_linear', 0.01) , name='lambda_w_linear')
    lambda_w_latent = tf.constant(try_get_params(params, 'reg_w_emb', 0.01) , name='lambda_w_emb')
    lambda_w_nn = tf.constant(try_get_params(params, 'reg_w_nn', 0.01), name='lambda_w_nn')
    #lambda_w_l1 = tf.constant(try_get_params(params, 'reg_w_l1', 0.01), name='lambda_w_l1')
     
    with tf.name_scope('error'): 
        l2_norm = 0
        for par in model_latent_params:
            ##--l2_norm +=  tf.multiply(lambda_w_l1, tf.reduce_sum(tf.abs(par)))  + tf.multiply(lambda_w_latent, tf.reduce_sum(tf.pow(par, 2)))
            l2_norm += tf.nn.l2_loss(par)* lambda_w_latent
        for par in model_linear_params:
            ##--l2_norm +=  tf.multiply(lambda_w_l1, tf.reduce_sum(tf.abs(par)))   + tf.multiply(lambda_w_linear, tf.reduce_sum(tf.pow(par, 2)))
            l2_norm += tf.nn.l2_loss(par)* lambda_w_linear 
        for par in model_w_nn_params:
            ##--l2_norm +=  tf.multiply(lambda_w_l1, tf.reduce_sum(tf.abs(par)))   + tf.multiply(lambda_w_nn, tf.reduce_sum(tf.pow(par, 2)))
            l2_norm += tf.nn.l2_loss(par) * lambda_w_nn  
        for par in model_w_nn_params_MF:
            l2_norm += tf.nn.l2_loss(par) * params['reg_w_nn_MF']  
        for par in model_latent_params:
            l2_norm += tf.nn.l2_loss(par) * params['reg_w_emb_MF']  
            
            
        loss = error + l2_norm
    
    eta = tf.constant(try_get_params(params, 'eta', 0.1))
    
    model_params = []
    model_params.extend(model_latent_params)
    model_params.extend(model_linear_params)
    model_params.extend(model_w_nn_params)
    model_params.extend(model_w_nn_params_MF)
    model_params.extend(model_latent_params_MF)


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) ##--
    with tf.control_dependencies(update_ops):
        type_of_opt = try_get_params(params, 'optimizer', 'sgd')
        if type_of_opt == 'adadelta':  
            train_step = tf.train.AdadeltaOptimizer(eta).minimize(loss,var_list=model_params)#
        elif type_of_opt == 'sgd':
            train_step = tf.train.GradientDescentOptimizer(try_get_params(params, 'learning_rate', 0.0001)).minimize(loss,var_list=model_params)
        elif type_of_opt =='adam':
            train_step = tf.train.AdamOptimizer(try_get_params(params, 'learning_rate', 0.0001)).minimize(loss, var_list=model_params)
        elif type_of_opt =='ftrl':
            train_step = tf.train.FtrlOptimizer(try_get_params(params, 'learning_rate', 0.0001)).minimize(loss,var_list=model_params)
        else:
            train_step = tf.train.GradientDescentOptimizer(try_get_params(params, 'learning_rate', 0.0001)).minimize(loss,var_list=model_params)          

    tf.summary.histogram("att_output[0]",att_output[:,0] )
    tf.summary.histogram("att_output[1]",att_output[:,1] )
    tf.summary.histogram("att_output[2]",att_output[:,2] )
    tf.summary.histogram("att_output[3]",att_output[:,3] )
    tf.summary.histogram("att_output[4](linear)",att_output[:,4] )
    tf.summary.histogram("att_output[5](bias)",att_output[:,5] )
 
    tf.summary.scalar('RMSE', error)
    tf.summary.scalar('loss', loss)
    #tf.summary.histogram('linear_weights_hist', w_linear)
    #tf.summary.histogram('embedding_weights_hist', w_emb)
    for var in model_params:
        tf.summary.histogram(var.name, var)
    

    merged_summary = tf.summary.merge_all()
    
    return train_step, loss, error, final_preds , merged_summary 

def my_focal_log_loss(_y, final_preds) :
    # focal loss for dense object detection https://arxiv.org/abs/1708.02002
    # we refer to this implementation:  https://github.com/tensorflow/tensorflow/pull/12257/files
    gamma = try_get_params(params, 'focal_gamma', 2)
    alpha = try_get_params(params, 'focal_alpha', 1)
    labels =  math_ops.to_float(_y)
    preds  = array_ops.where(math_ops.equal(labels,1), final_preds, 1. - final_preds)
    losses = - alpha * (1. - preds) ** gamma * math_ops.log(preds + 1e-7)
    return tf.reduce_mean(losses)
    

def try_get_params(params, name, dvalue):
    if name in params:
        return params[name]
    else:
        #logger.info('missing param : ' + name)
        return dvalue

def VeryIfEmbEnable(domain_name):
    if domain_name == 'userbasic' and not try_get_params(params, 'is_use_userbasic_part', True):
        return False 
    if domain_name == 'itembasic' and not try_get_params(params, 'is_use_itembasic_part', True):
        return False
    if domain_name == 'news' and not try_get_params(params, 'is_use_news_part', True):
        return False 
    if domain_name == 'browsing' and not try_get_params(params, 'is_use_browsing_part', True):
        return False 
    if domain_name == 'search' and not try_get_params(params, 'is_use_search_part', True):
        return False 
    if domain_name == 'userfeatureall' and not try_get_params(params, 'is_use_userall_part', True):
        return False     
    return True
       
def get_final_emb_dim():
    layer_sizes = try_get_params(params, 'layer_sizes', [])
    dim =   try_get_params(params, 'dim', 8)  
    if not layer_sizes :
        return dim 
    else:
        if isinstance(layer_sizes[0], list):
            for subnetwork in layer_sizes:
                if subnetwork:
                    return subnetwork[-1]
            return dim
        else:
            return layer_sizes[-1]

def build_fusion_layer(w_nn_input, _field_cnt, dim, layer_sizes, sub_network_idx, init_value, _train_phase, 
                       model_linear_params, model_w_nn_params ,  w_linear, w_emb, dropout_value):
    with tf.name_scope('sub_network_'+str(sub_network_idx)) :  
        hidden_nn_layers = []
        hidden_nn_layers.append(w_nn_input)
        last_layer_size = _field_cnt * dim
        layer_idx = 0
            
        if not layer_sizes:
                return tf.add_n(tf.split(hidden_nn_layers[0], _field_cnt,  1)), dim 
        else:            
            for layer_size in layer_sizes:
                cur_w_nn_layer = tf.Variable(
                    tf.truncated_normal([last_layer_size, layer_size], stddev=init_value / math.sqrt(float(last_layer_size)), mean=0), ##-- 10
                    name='w_nn_layer' + str(layer_idx), dtype=tf.float32)
    
                cur_b_nn_layer = tf.Variable(tf.truncated_normal([layer_size], stddev=params['init_value_b'], mean=0), name='b_nn_layer' + str(layer_idx)) #tf.get_variable('b_nn_layer' + str(layer_idx), [layer_size], initializer=tf.constant_initializer(0.0)) 
    
                #
                if params['dropout_probs']<1.0 :
                    cur_hidden_nn_input = tf.nn.dropout(hidden_nn_layers[layer_idx], dropout_value, name='nn_input_' + str(layer_idx) )
                else:
                    cur_hidden_nn_input = hidden_nn_layers[layer_idx] 
                    
                if (try_get_params(params, 'enable_batch_norm', False)):
                    cur_hidden_nn_input = tf.contrib.layers.batch_norm(cur_hidden_nn_input, 
                                          center=True, scale=True, 
                                          is_training=_train_phase)    
            
                cur_hidden_nn_layer = tf.nn.xw_plus_b(cur_hidden_nn_input, cur_w_nn_layer, cur_b_nn_layer, name='nn_values_raw_' + str(layer_idx))
                
                cur_hidden_nn_layer = GoActivationFunc(params['activations'][layer_idx], cur_hidden_nn_layer, 'nn_values_act_' + str(layer_idx))
                
                
                #cur_hidden_nn_layer = tf.matmul(hidden_nn_layers[layer_idx], cur_w_nn_layer)
                #w_nn_layers.append(cur_w_nn_layer)
                hidden_nn_layers.append(cur_hidden_nn_layer)
    
                layer_idx += 1
                last_layer_size = layer_size
    
                model_w_nn_params.append(cur_w_nn_layer)
                model_linear_params.append(cur_b_nn_layer)
                
            return hidden_nn_layers[-1], layer_sizes[-1]
            
                
def GoActivationFunc(act_type, input, name):
    output = 0
    if act_type == 'tanh':
        output = tf.nn.tanh(input, name=name, )
    elif act_type == 'sigmoid':
        output = tf.nn.sigmoid(input, name=name )
    elif act_type == 'relu':
        output = tf.nn.relu(input, name = name)
    return output              
             
def build_domain_model(_train_phase , place_holders, dropout_value, 
                         _field_cnt,  domain_name, 
                         model_linear_params, model_w_nn_params , 
                         w_linear, w_emb, item_emb = None):
    
    with tf.name_scope(domain_name) :    
        
        _x = tf.SparseTensor(place_holders._indices, place_holders._values, place_holders._shape)  # m * FEATURE_COUNT sparse tensor

        init_value = try_get_params(params, 'init_value', 0.001) 
        dim = try_get_params(params, 'dim', 8)  
        #layer_sizes = params['layer_sizes'] #params['layer_sizes_' + domain_name]
        
        output_dim = get_final_emb_dim()
     
        # linear part
        if try_get_params(params, 'is_use_linear_part', True):            
            linear_preds = tf.sparse_tensor_dense_matmul(_x, w_linear, name='contr_from_linear')
        else:
            linear_preds = tf.zeros([try_get_params(params, 'batch_size', 1024), 1], tf.float32, name='zeros_linear')
            
        if VeryIfEmbEnable(domain_name):  
            w_nn_sparseIndexs = tf.SparseTensor(place_holders._field2feature_indices, place_holders._field2feature_values, place_holders._field2feature_shape)
            w_nn_sparseWeights = tf.SparseTensor(place_holders._field2feature_indices, place_holders._field2feature_weights, place_holders._field2feature_shape)
            
            
            if try_get_params(params, 'is_ADIN_ensebled', False) and item_emb is not None:  # testing only. We didn't use it in the final model
                din_layer_size = 16
                w_din_feature = tf.gather(w_emb, place_holders._field2feature_values)
                w_din_feature = w_din_feature * tf.reshape( place_holders._field2feature_weights, [-1,1])
                
                w_din_item = tf.gather(item_emb, place_holders._itememb_lookup_indices)
                
                w1_din_layer = tf.Variable(
                    tf.truncated_normal([params['dim'], din_layer_size], stddev=init_value / math.sqrt(float(din_layer_size)), mean=0), ##-- 10
                    name='w1_din_layer' , dtype=tf.float32) 
                model_w_nn_params.append(w1_din_layer)
                w2_din_layer = tf.Variable(
                    tf.truncated_normal([params['dim'], din_layer_size], stddev=init_value / math.sqrt(float(din_layer_size)), mean=0), ##-- 10
                    name='w2_din_layer' , dtype=tf.float32)    
                model_w_nn_params.append(w2_din_layer)
                b1_din_layer = tf.Variable(tf.truncated_normal([din_layer_size], stddev=params['init_value_b'], mean=0), name='b1_din_layer' ) #tf.get_variable('b_nn_layer' + str(layer_idx), [layer_size], initializer=tf.constant_initializer(0.0)) 
                model_linear_params.append(b1_din_layer)
                
                hidden_din_layer = tf.matmul(w_din_feature, w1_din_layer) + tf.matmul(w_din_item, w2_din_layer) + b1_din_layer
                hidden_din_layer = tf.nn.relu(hidden_din_layer)
    
                w3_din_layer = tf.Variable(
                    tf.truncated_normal([din_layer_size, 1], stddev=init_value / math.sqrt(float(din_layer_size)), mean=0), ##-- 10
                    name='w3_din_layer' , dtype=tf.float32)  
                model_w_nn_params.append(w3_din_layer)
                b2_din_layer = tf.Variable(tf.truncated_normal([1], stddev=params['init_value_b'], mean=0), name='b2_din_layer' )
                model_linear_params.append(b2_din_layer)
                
                output_din_layer = tf.nn.xw_plus_b(hidden_din_layer, w3_din_layer, b2_din_layer)
                
                w_nn_sparseWeights = tf.SparseTensor(place_holders._field2feature_indices, tf.reshape(output_din_layer, [-1]), place_holders._field2feature_shape)
                w_nn_sparseWeights = tf.sparse_softmax(w_nn_sparseWeights)
                
                
            w_nn_input_orgin = tf.nn.embedding_lookup_sparse(w_emb, w_nn_sparseIndexs, w_nn_sparseWeights, combiner="sum", name='w_nn_input_orgin' )
            w_nn_input = tf.reshape(w_nn_input_orgin, [-1, dim * _field_cnt], name='w_nn_input')  
            
            if not params['layer_sizes']:
                emb_res = tf.add_n(tf.split(w_nn_input, _field_cnt,  1))
            else:
                #last_pooling_size = params['layer_sizes'][0]
                before_pooling_size = 0 
                layer_before_pooling = []
                sub_network_idx = 0
                for layer_sizes in  params['layer_sizes']:
                    cur_fusion_layer, cur_fusion_layer_size = build_fusion_layer(w_nn_input, _field_cnt, dim, layer_sizes, sub_network_idx, init_value, _train_phase, 
                                                                                 model_linear_params, model_w_nn_params ,  w_linear, w_emb, dropout_value)
                    layer_before_pooling.append(cur_fusion_layer)
                    before_pooling_size += cur_fusion_layer_size
                    sub_network_idx +=1 
                
                if (try_get_params(params, 'fusion_method', 'add')) == 'linear_tran' :#'fusion_method': 'linear_tran', 
                    trans_w = tf.Variable(
                            tf.truncated_normal([before_pooling_size, output_dim], stddev=init_value / math.sqrt(float(before_pooling_size)), mean=0), ##-- 10
                            name='fusion_trans_w', dtype=tf.float32)                
                    model_w_nn_params.append(trans_w)                
                    emb_res = tf.matmul( tf.concat(layer_before_pooling,1), trans_w)            
                else:
                    emb_res = tf.add_n(layer_before_pooling) 
   
        else:
            emb_res = tf.zeros([try_get_params(params, 'batch_size', 1024), output_dim ], tf.float32, name='zeros_emb') ##--
            
            r'''
            hidden_nn_layers = [] 
            last_layer_size = _field_cnt * dim
            layer_idx = 0
            hidden_nn_layers.append(tf.zeros([try_get_params(params, 'batch_size', 1024), dim * _field_cnt], tf.float32))
            
            if not layer_sizes:
                hidden_nn_layers[0] = tf.accumulate_n(tf.split(hidden_nn_layers[0], _field_cnt,  1))
     
            for layer_size in layer_sizes:
                cur_w_nn_layer = tf.Variable(
                    tf.truncated_normal([last_layer_size, layer_size], stddev=init_value / math.sqrt(float(last_layer_size)), mean=0), ##-- 10
                    name='w_nn_layer' + str(layer_idx), dtype=tf.float32)
    
                cur_b_nn_layer = tf.Variable(tf.truncated_normal([layer_size], stddev=init_value, mean=0), name='b_nn_layer' + str(layer_idx)) #tf.get_variable('b_nn_layer' + str(layer_idx), [layer_size], initializer=tf.constant_initializer(0.0)) 
    
                #
                if _train_phase ==True and params['dropout_probs'][layer_idx]<1.0:
                    cur_hidden_nn_input = tf.nn.dropout(hidden_nn_layers[layer_idx], params['dropout_probs'][layer_idx], name='nn_input_' + str(layer_idx) )
                else:
                    cur_hidden_nn_input = hidden_nn_layers[layer_idx] 
                cur_hidden_nn_layer = tf.nn.xw_plus_b(cur_hidden_nn_input, cur_w_nn_layer, cur_b_nn_layer, name='nn_values_raw_' + str(layer_idx))
                
                if params['activations'][layer_idx]=='tanh':
                    cur_hidden_nn_layer = tf.nn.tanh(cur_hidden_nn_layer, name='nn_values_tanh_' + str(layer_idx))
                elif params['activations'][layer_idx]=='sigmoid':
                    cur_hidden_nn_layer = tf.nn.sigmoid(cur_hidden_nn_layer, name='nn_values_sig_' + str(layer_idx))
                elif params['activations'][layer_idx]=='relu':
                    cur_hidden_nn_layer = tf.nn.relu(cur_hidden_nn_layer, name='nn_values_relu_' + str(layer_idx))
                
                #cur_hidden_nn_layer = tf.matmul(hidden_nn_layers[layer_idx], cur_w_nn_layer)
                #w_nn_layers.append(cur_w_nn_layer)
                hidden_nn_layers.append(cur_hidden_nn_layer)
    
                layer_idx += 1
                last_layer_size = layer_size
    
                model_w_nn_params.append(cur_w_nn_layer)
                model_linear_params.append(cur_b_nn_layer)
                
            emb_res = hidden_nn_layers[-1]
            '''
    return emb_res, linear_preds



def predict_test_file(preds, sess, test_file,
                      _att_features, place_holders, _y, _train_phase, _train_init_done,
                      epoch, batch_size, tag, path, output_prediction, _dropout_value):
    if output_prediction:
        wt = open(path + '/attentiveUserEmb_pred_' + tag + str(epoch) + '.txt', 'w')

    query2res = {}

    for training_input_in_sp in load_data_cache(test_file):
        
        if training_input_in_sp['userbasic']['shape'][0]< params['batch_size']:
            continue
        
                    
        if try_get_params(params,'is_ADIN_ensebled', False):
            update_feature_4_ADIN(training_input_in_sp['news'])
            update_feature_4_ADIN(training_input_in_sp['browsing'])
            update_feature_4_ADIN(training_input_in_sp['search'])
        
        
        #print('field2feature_values ',test_input_in_sp['field2feature_values'])
        #print('field2feature_weights ',test_input_in_sp['field2feature_weights'])
        #break
        predictios = sess.run(preds, feed_dict={
                _att_features:   training_input_in_sp['att_features'],
                _y: training_input_in_sp['labels'],                         
                place_holders['userbasic']._indices: training_input_in_sp['userbasic']['indices'], 
                place_holders['userbasic']._values: training_input_in_sp['userbasic']['values'],
                place_holders['userbasic']._shape: training_input_in_sp['userbasic']['shape'],                   
                place_holders['userbasic']. _field2feature_indices: training_input_in_sp['userbasic']['field2feature_indices'],
                place_holders['userbasic']._field2feature_values: training_input_in_sp['userbasic']['field2feature_values'],
                place_holders['userbasic']. _field2feature_weights: training_input_in_sp['userbasic']['field2feature_weights'],
                place_holders['userbasic']. _field2feature_shape: training_input_in_sp['userbasic']['filed2feature_shape'],
                
                place_holders['itembasic']._indices: training_input_in_sp['itembasic']['indices'], 
                place_holders['itembasic']._values: training_input_in_sp['itembasic']['values'],
                place_holders['itembasic']._shape: training_input_in_sp['itembasic']['shape'],                   
                place_holders['itembasic']. _field2feature_indices: training_input_in_sp['itembasic']['field2feature_indices'],
                place_holders['itembasic']._field2feature_values: training_input_in_sp['itembasic']['field2feature_values'],
                place_holders['itembasic']. _field2feature_weights: training_input_in_sp['itembasic']['field2feature_weights'],
                place_holders['itembasic']. _field2feature_shape: training_input_in_sp['itembasic']['filed2feature_shape'],
                
                place_holders['news']._indices: training_input_in_sp['news']['indices'], 
                place_holders['news']._values: training_input_in_sp['news']['values'],
                place_holders['news']._shape: training_input_in_sp['news']['shape'],                   
                place_holders['news']. _field2feature_indices: training_input_in_sp['news']['field2feature_indices'],
                place_holders['news']._field2feature_values: training_input_in_sp['news']['field2feature_values'],
                place_holders['news']. _field2feature_weights: training_input_in_sp['news']['field2feature_weights'],
                place_holders['news']. _field2feature_shape: training_input_in_sp['news']['filed2feature_shape'],
                place_holders['news']._itememb_lookup_indices: training_input_in_sp['news']['itememb_lookup_indices'] if try_get_params(params,'is_ADIN_ensebled',False) else [],
                
                place_holders['browsing']._indices: training_input_in_sp['browsing']['indices'], 
                place_holders['browsing']._values: training_input_in_sp['browsing']['values'],
                place_holders['browsing']._shape: training_input_in_sp['browsing']['shape'],                   
                place_holders['browsing']. _field2feature_indices: training_input_in_sp['browsing']['field2feature_indices'],
                place_holders['browsing']._field2feature_values: training_input_in_sp['browsing']['field2feature_values'],
                place_holders['browsing']. _field2feature_weights: training_input_in_sp['browsing']['field2feature_weights'],
                place_holders['browsing']. _field2feature_shape: training_input_in_sp['browsing']['filed2feature_shape'],
                place_holders['browsing']._itememb_lookup_indices: training_input_in_sp['browsing']['itememb_lookup_indices'] if try_get_params(params,'is_ADIN_ensebled',False) else [],
                
                place_holders['search']._indices: training_input_in_sp['search']['indices'], 
                place_holders['search']._values: training_input_in_sp['search']['values'],
                place_holders['search']._shape: training_input_in_sp['search']['shape'],                   
                place_holders['search']. _field2feature_indices: training_input_in_sp['search']['field2feature_indices'],
                place_holders['search']._field2feature_values: training_input_in_sp['search']['field2feature_values'],
                place_holders['search']. _field2feature_weights: training_input_in_sp['search']['field2feature_weights'],
                place_holders['search']. _field2feature_shape: training_input_in_sp['search']['filed2feature_shape'],
                place_holders['search']._itememb_lookup_indices: training_input_in_sp['search']['itememb_lookup_indices'] if try_get_params(params,'is_ADIN_ensebled',False) else [],
                
                place_holders['dnn']._indices: training_input_in_sp['dnn_features']['indices'], 
                place_holders['dnn']._values: training_input_in_sp['dnn_features']['values'],
                place_holders['dnn']._shape: training_input_in_sp['dnn_features']['shape'],                   
                place_holders['dnn']. _field2feature_indices: training_input_in_sp['dnn_features']['field2feature_indices'],
                place_holders['dnn']._field2feature_values: training_input_in_sp['dnn_features']['field2feature_values'],
                place_holders['dnn']. _field2feature_weights: training_input_in_sp['dnn_features']['field2feature_weights'],
                place_holders['dnn']. _field2feature_shape: training_input_in_sp['dnn_features']['filed2feature_shape'],  

                place_holders['userfeatureall']._indices: training_input_in_sp['userfeatureall']['indices'], 
                place_holders['userfeatureall']._values: training_input_in_sp['userfeatureall']['values'],
                place_holders['userfeatureall']._shape: training_input_in_sp['userfeatureall']['shape'],                   
                place_holders['userfeatureall']. _field2feature_indices: training_input_in_sp['userfeatureall']['field2feature_indices'],
                place_holders['userfeatureall']._field2feature_values: training_input_in_sp['userfeatureall']['field2feature_values'],
                place_holders['userfeatureall']. _field2feature_weights: training_input_in_sp['userfeatureall']['field2feature_weights'],
                place_holders['userfeatureall']. _field2feature_shape: training_input_in_sp['userfeatureall']['filed2feature_shape'],    
                place_holders['userfeatureall']._itememb_lookup_indices: training_input_in_sp['userfeatureall']['itememb_lookup_indices'] if try_get_params(params,'is_ADIN_ensebled',False) else [],
                                                                
                _train_phase:False, _train_init_done: True, _dropout_value: 1.0
        }).reshape(-1).tolist()
        
        
        for (gt, preded, qid) in zip(training_input_in_sp['labels'].reshape(-1).tolist(), predictios, training_input_in_sp['qids']):
            if output_prediction:
                wt.write('{0:d},{1:f},{2:s}\n'.format(int(gt), preded,qid))
            if qid not in query2res:
                query2res[qid] = []
            query2res[qid].append([gt, preded])
            
    if output_prediction:
        wt.close()
    return query2res

    
 

def compute_metric(query2res):
    result = {}

    for m in params['metrics']:
        if 'auc' == m['name'].lower():
            gt_scores = []
            pred_scores = []
            for qid in query2res:
                gt_scores.extend([x[0] for x in query2res[qid]] )
                pred_scores.extend([x[1] for x in query2res[qid]] )
            #print('gt_scores ',gt_scores) 
            #print(query2res)
            result['auc'] = roc_auc_score(np.asarray(gt_scores), np.asarray(pred_scores))
        elif 'auc_ind'  ==  m['name'].lower():  # mean individual AUC
            aucs = []
            for qid in query2res:
                gt_scores = np.asarray([x[0] for x in query2res[qid]])
                if gt_scores.min() > 0 or gt_scores.max() < 1:
                    continue
                pred_scores = np.asarray([x[1] for x in query2res[qid]])
                aucs.append(roc_auc_score(gt_scores, pred_scores))
            result['auc_ind'] = np.asarray(aucs).mean()
        elif 'log_loss' == m['name'].lower():
            gt_scores = []
            pred_scores = []
            for qid in query2res:
                gt_scores.extend([x[0] for x in query2res[qid]] )
                pred_scores.extend([x[1] for x in query2res[qid]] ) 
            result['log_loss'] = log_loss(np.asarray(gt_scores), np.asarray(pred_scores))
        elif 'precision' in m['name']:  # definition of precision and recall for recommendation systems: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
            precisions = []
            for qid in query2res:
                k = min(m['k'], len(query2res[qid]))
                if k<=0:
                    continue
                gt_scores = np.asarray([x[0] for x in query2res[qid]])
                pred_scores = np.asarray([x[1] for x in query2res[qid]])
                if gt_scores.min() > 0 or gt_scores.max() < 1:
                    continue
                precision = gt_scores[np.argsort(pred_scores)[::-1][:k]].mean()
                precisions.append(precision)
            result['precision@' + str(m['k'])] = np.asarray(precisions).mean()
        elif 'map' == m['name'].lower(): # we use the map defined by http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html  , different from this : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html 
            precisions = []
            for qid in query2res:
                k = min(m['k'], len(query2res[qid]))
                if k<=0:
                    continue
                gt_scores = np.asarray([x[0] for x in query2res[qid]])
                pred_scores = np.asarray([x[1] for x in query2res[qid]])
                if gt_scores.min() > 0 or gt_scores.max() < 1:
                    continue
                sorted_idx = np.argsort(pred_scores)[::-1]
                precision = np.asarray([gt_scores[sorted_idx[:i+1]].mean() for i in range(k)]).mean()
                precisions.append(precision)
            result['map@' + str(m['k'])] = np.asarray(precisions).mean()
        elif 'ndcg' == m['name'].lower():
            precisions = []
            for qid in query2res:
                k = min(m['k'], len(query2res[qid]))
                if k<=0:
                    continue
                gt_scores = np.asarray([x[0] for x in query2res[qid]])
                pred_scores = np.asarray([x[1] for x in query2res[qid]])
                if gt_scores.min() > 0 or gt_scores.max() < 1:
                    continue
                sorted_idx = np.argsort(pred_scores)[::-1]
                ideal_idx = np.argsort(gt_scores)[::-1]                
                dcg = dcg_at_k(gt_scores[sorted_idx],k)
                idcg = dcg_at_k(gt_scores[ideal_idx], k)
                precision = dcg / idcg
                precisions.append(precision)
            result['ndcg@' + str(m['k'])] = np.asarray(precisions).mean()
        elif 'hit' == m['name'].lower() or 'recall' == m['name']: # hit ratio: http://chbrown.github.io/kdd-2013-usb/kdd/p892.pdf
            recalls = []
            hits = []
            for qid in query2res:
                k = min(m['k'], len(query2res[qid]))
                if k<=0:
                    continue
                gt_scores = np.asarray([x[0] for x in query2res[qid]])
                pred_scores = np.asarray([x[1] for x in query2res[qid]])
                if gt_scores.min() > 0 or gt_scores.max() < 1:
                    continue
                sorted_idx = np.argsort(pred_scores)[::-1]
                hit = gt_scores[sorted_idx][:k].sum()
                total = gt_scores.sum()
                recalls.append(hit / total)
                if hit>0:
                    hits.append(1.0)
                else:
                    hits.append(0.0)
            result['hit@' + str(m['k'])] = np.asarray(hits).mean()
            result['recall@' + str(m['k'])] = np.asarray(recalls).mean()
            

    return result

def dcg_at_k(rels, k):
    if rels.size<=0:
        return 0 
    return np.sum(np.asfarray(rels[:k])/np.log2(np.arange(2,k+2)))

def pre_build_data_cache_if_need(infile, batch_size, rebuild_cache):
    outfile = infile.replace('.csv','.pkl').replace('.txt','.pkl')
    if not os.path.isfile(outfile) or rebuild_cache:
        print('pre_build_data_cache for ', infile)
        pre_build_data_cache(infile, outfile, batch_size)
        print('pre_build_data_cache finished.' )


def check_dir_of_file(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
  
 





def grid_search():
 
     
    r'''
    print("DFM")
    params['log_path'] = 'model_output/DFM/normal/'
    params['model_path'] = 'model_output/DFM/normal'
    params['layer_sizes'] = [[32,32,32],[32],[]]    
    params['is_attention_enabled'] = True
    params['reg_w_emb_MF'] = 0.002
    params['reg_w_nn_MF'] = 0.001 # 0.001
    params['reg_w_linear'] = 0.005 # 0.001
    params['reg_w_nn'] = 0.001 # 0.001
    params['att_layer_size'] = [32]
    single_run() 
    '''

    
    params[ 'metrics']  = [
             {'name': 'auc_ind'},
             {'name': 'auc'}
            ,{'name': 'log_loss'}
            #, {'name': 'precision', 'k': 1}
            #, {'name': 'precision', 'k': 2}
            #, {'name': 'precision', 'k': 3}
            #, {'name': 'precision', 'k': 4}
            #, {'name': 'precision', 'k': 5}
            #, {'name': 'precision', 'k': 10}
            #, {'name': 'precision', 'k': 20}
            , {'name': 'map', 'k': 2}
            , {'name': 'map', 'k': 4}
            , {'name': 'map', 'k': 6}
            , {'name': 'map', 'k': 8}
            , {'name': 'map', 'k': 10}
            , {'name': 'map', 'k': 20}
            #, {'name': 'recall', 'k': 1}
            #, {'name': 'recall', 'k': 2}
            #, {'name': 'recall', 'k': 3}
            #, {'name': 'recall', 'k': 4}
            #, {'name': 'recall', 'k': 5}
            #, {'name': 'recall', 'k': 10}
            #, {'name': 'recall', 'k': 20}
            , {'name': 'hit', 'k': 2}
            , {'name': 'hit', 'k': 4}
            , {'name': 'hit', 'k': 6}
            , {'name': 'hit', 'k': 8}
            , {'name': 'hit', 'k': 10}
            #, {'name': 'hit', 'k': 10}
            #, {'name': 'hit', 'k': 20}
            , {'name': 'ndcg', 'k': 2}
            , {'name': 'ndcg', 'k': 4}
            , {'name': 'ndcg', 'k': 6}
            , {'name': 'ndcg', 'k': 8}
            , {'name': 'ndcg', 'k': 10}
            #, {'name': 'ndcg', 'k': 10}
            #, {'name': 'ndcg', 'k': 20}            
        ]
    params['test_file'] = 'data/AttentiveUserEmb/newsuserfeature_withposition/filtered/DFM/news_test.txt'
    

     
    r'''
    params['log_path'] = 'model_output/DFM/normal/testing/'
    params['model_path'] = 'model_output/DFM/normal'
    params['layer_sizes'] = [[32,32,32],[32],[]]     
    params['is_attention_enabled'] = True
    params['reg_w_emb_MF'] = 0.002
    params['reg_w_nn_MF'] = 0.001 # 0.001
    params['reg_w_linear'] = 0.005 # 0.001
    params['reg_w_nn'] = 0.001 # 0.001
    params['att_layer_size'] = [32]
    params['predict_only']=True

    params['model_load_path']= 'model_output/DFM/normal/attEmbDNN-700'
    single_run()

    '''
      
if __name__ == '__main__':
    grid_search()