import pickle
import pandas as pd
from recbole.utils import init_seed
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import NGCF
from recbole.trainer import Trainer
import torch
from recbole.data.interaction import Interaction
import numpy as np


''' Reading the data '''
with open('general_posts_b_tr.pickle', 'rb') as handle:
    general_posts_b_tr = pickle.load(handle)

with open('general_sources_b_tr.pickle', 'rb') as handle:
    general_sources_b_tr = pickle.load(handle)

with open('courses_follows_b_tr.pickle', 'rb') as handle:
    courses_follows_b_tr = pickle.load(handle)

with open('users_follows_b_tr.pickle', 'rb') as handle:
    users_follows_b_tr = pickle.load(handle)

with open('ratings_b_tr.pickle', 'rb') as handle:
    ratings_b_tr = pickle.load(handle)

with open('post_likes_b_tr.pickle', 'rb') as handle:
    post_likes_b_tr = pickle.load(handle)

with open('post_comments_b_tr.pickle', 'rb') as handle:
    post_comments_b_tr = pickle.load(handle)

with open('post_comment_likes_b_tr.pickle', 'rb') as handle:
    post_comment_likes_b_tr = pickle.load(handle)
    

''' Interactions. '''
interactions = pd.concat([general_posts_b_tr.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          general_sources_b_tr.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          courses_follows_b_tr.rename(columns = {'id_course':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          users_follows_b_tr.rename(columns = {'destination_id_user': 'id_content', 'origin_id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          ratings_b_tr.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          post_likes_b_tr.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          post_comments_b_tr.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          post_comments_b_tr.rename(columns = {'id_comment':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          post_comment_likes_b_tr.rename(columns = {'id_comment':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']]
                          ])

interactions['item_id:token'] = interactions[['id_content']].apply(tuple,axis=1).rank(method='dense').astype(int)
interactions['user_id:token'] = interactions['user_id:token'].str.slice(start=1).astype('int64')

item_id = interactions[['id_content', 'item_id:token']].drop_duplicates()

# !mkdir \educational\interactions_train

# interactions[['user_id:token', 'item_id:token']].to_csv('\educational\interactions_train\interactions_train.inter', index=False, sep='\t')


''' Model. '''
def NGCF_model(train_set, content_set, filter_dict, hidden_size_list, node_dropout, message_dropout, reg_weight):
 
    parameter_dict = {
        'data_path': '\educational',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'load_col': {'inter': ['user_id', 'item_id']},
        
        'hidden_size_list': hidden_size_list,
        'node_dropout': node_dropout,
        'message_dropout': message_dropout,
        'reg_weight': reg_weight,
        
        'eval_args': {
          'split': {'RS': [10,0,0]}
          }
    }
    
    config = Config(model='NGCF', dataset='interactions_train', config_dict=parameter_dict)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = NGCF(config, train_data.dataset).to(config['device'])
    
    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data)
    
    iid_set = dataset.token2id(dataset.iid_field, list(map(str, content_set['item_id:token'])))
    iid = np.vstack((iid_set, list(content_set['item_id:token']))).T
    iid = pd.DataFrame(iid, columns=['iid', 'item_id:token'])
    
    mrrs_10 = []
    for user in np.unique(train_set['user_id:token']):    
        uid = dataset.token2id(dataset.uid_field, [str(user)]).item()
        content = list(train_set[train_set['user_id:token']==user]['item_id:token'])
        
        input_inter = Interaction({
            'user_id': torch.tensor([uid])
        })
        input_inter = dataset.join(input_inter)
        input_inter = input_inter.to(config['device'])

        scores = model.full_sort_predict(input_inter)
        scores = scores.view(-1, dataset.item_num)

        top_k = torch.topk(scores, scores.shape[1])
        top_k = np.concatenate((top_k[1].detach().numpy().T, top_k[0].detach().numpy().T), axis=1)
        top_k = top_k[np.isin(top_k[:,0], iid_set)]

        predictions = pd.DataFrame(top_k, columns=['iid', 'prediction'])
        predictions = predictions.merge(iid, how='left', on='iid')
        predictions = predictions[~predictions['item_id:token'].isin(filter_dict[user])]

        recommendation_list_10 = list(predictions.nlargest(10, 'prediction')['item_id:token'].reset_index(drop=True))
        
        mrr_10 = 0
        if any(i in recommendation_list_10 for i in content):
            mrr_10 = 1.0/(1+min(np.where(np.isin(recommendation_list_10,content))[0]))
        mrrs_10.append(mrr_10)
    
    return np.mean(mrrs_10)



train_file_path = 'courses_follows_tr.pickle'
content_collumn = 'item_id:token'
content_type = courses_follows_b_tr.rename(columns = {'id_course':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']]
content_type = content_type.merge(item_id, how='left', on='id_content')
content_type['user_id:token'] = content_type['user_id:token'].str.slice(start=1).astype('int64')

content_file = item_id[item_id['id_content'].str.startswith('c')]

with open(train_file_path, 'rb') as handle:
    train_file = pickle.load(handle)

train_file = train_file.rename(columns = {'id_course':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']]
train_file = train_file.merge(item_id, how='left', on='id_content')
train_file['item_id:token'] = train_file['item_id:token'].fillna(item_id['item_id:token'].max()+train_file[['id_content']].apply(tuple,axis=1).rank(method='dense').astype(int))
train_file['item_id:token'] = train_file['item_id:token'].astype(int)
train_file['user_id:token'] = train_file['user_id:token'].str.slice(start=1).astype('int64')

train_file = train_file[train_file['user_id:token'].isin(interactions['user_id:token'])]

train_filter = content_type[content_type['user_id:token'].isin(train_file['user_id:token'])]
train_filter = train_filter.groupby('user_id:token', sort=False)[content_collumn].agg(set).to_dict()


hsl = [64,64,64]
# hsl = [128,128,128]
# hsl = [256,256,256]
# hsl = [512,512,512]
nd_list = [0.0,0.1,0.2]
md_list = [0.0,0.1,0.2,0.3]
rw_list = [1e-5,1e-4]

mrr_list = []
for nd in nd_list:
    for md in md_list:
        for rw in rw_list:
            mrr_list.append(NGCF_model(train_file, content_file, train_filter, hsl, nd, md, rw))
