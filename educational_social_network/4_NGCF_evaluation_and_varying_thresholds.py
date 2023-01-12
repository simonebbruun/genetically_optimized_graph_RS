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
with open('general_posts_b_t.pickle', 'rb') as handle:
    general_posts_b_t = pickle.load(handle)

with open('general_sources_b_t.pickle', 'rb') as handle:
    general_sources_b_t = pickle.load(handle)

with open('courses_follows_b_t.pickle', 'rb') as handle:
    courses_follows_b_t = pickle.load(handle)

with open('users_follows_b_t.pickle', 'rb') as handle:
    users_follows_b_t = pickle.load(handle)

with open('ratings_b_t.pickle', 'rb') as handle:
    ratings_b_t = pickle.load(handle)

with open('post_likes_b_t.pickle', 'rb') as handle:
    post_likes_b_t = pickle.load(handle)

with open('post_comments_b_t.pickle', 'rb') as handle:
    post_comments_b_t = pickle.load(handle)

with open('post_comment_likes_b_t.pickle', 'rb') as handle:
    post_comment_likes_b_t = pickle.load(handle)
    

''' Interactions. '''
interactions = pd.concat([general_posts_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          general_sources_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          courses_follows_b_t.rename(columns = {'id_course':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          users_follows_b_t.rename(columns = {'destination_id_user': 'id_content', 'origin_id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          ratings_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          post_likes_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          post_comments_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          post_comments_b_t.rename(columns = {'id_comment':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']],
                          post_comment_likes_b_t.rename(columns = {'id_comment':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']]
                          ])

interactions['item_id:token'] = interactions[['id_content']].apply(tuple,axis=1).rank(method='dense').astype(int)
interactions['user_id:token'] = interactions['user_id:token'].str.slice(start=1).astype('int64')

item_id = interactions[['id_content', 'item_id:token']].drop_duplicates()

# !mkdir \educational\interactions_test

# interactions[['user_id:token', 'item_id:token']].to_csv('\educational\interactions_test\interactions_test.inter', index=False, sep='\t')


''' Model. '''
def NGCF_model(test_set, content_set, filter_dict, hidden_size_list, node_dropout, message_dropout, reg_weight):
 
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
    
    config = Config(model='NGCF', dataset='interactions_test', config_dict=parameter_dict)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = NGCF(config, train_data.dataset).to(config['device'])
    
    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data)
    
    iid_set = dataset.token2id(dataset.iid_field, list(map(str, content_set['item_id:token'])))
    iid = np.vstack((iid_set, list(content_set['item_id:token']))).T
    iid = pd.DataFrame(iid, columns=['iid', 'item_id:token'])
    
    mrrs = []
    hrs = []
    users = []
    for user in np.unique(test_set['user_id:token']):    
        uid = dataset.token2id(dataset.uid_field, [str(user)]).item()
        content = list(test_set[test_set['user_id:token']==user]['item_id:token'])
        
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
        
        mrr_t = []
        hr_t = []
        for t in range(20):
            recommendation_list = list(predictions.nlargest(t+1, 'prediction')['item_id:token'].reset_index(drop=True))
        
            mrr = 0
            hr = 0
            if any(i in recommendation_list for i in content):
                mrr = 1.0/(1+min(np.where(np.isin(recommendation_list,content))[0]))
                hr = 1
            mrr_t.append(mrr)
            hr_t.append(hr)
        mrrs.append(mrr_t)
        hrs.append(hr_t)        
        users.append(user)
    mrrs = np.stack(mrrs, axis=0)
    hrs = np.stack(hrs, axis=0)
    
    print("MRR5 %f" % np.mean(mrrs[:,4]))
    print("MRR10 %f" % np.mean(mrrs[:,9]))
    print("HR5 %f" % np.mean(hrs[:,4]))    
    print("HR10 %f" % np.mean(hrs[:,9]))
    
    return (np.mean(mrrs, axis=0), np.mean(hrs, axis=0)), (mrrs[:,4], mrrs[:,9], hrs[:,4], hrs[:,9], users)


test_file_path ='courses_follows_t.pickle'
content_collumn = 'item_id:token'
content_type = courses_follows_b_t.rename(columns = {'id_course':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']]
content_type = content_type.merge(item_id, how='left', on='id_content')
content_type['user_id:token'] = content_type['user_id:token'].str.slice(start=1).astype('int64')

content_file = item_id[item_id['id_content'].str.startswith('c')]

with open(test_file_path, 'rb') as handle:
    test_file = pickle.load(handle)

test_file = test_file.rename(columns = {'id_course':'id_content', 'id_user':'user_id:token'})[['user_id:token', 'id_content']]
test_file = test_file.merge(item_id, how='left', on='id_content')
test_file['item_id:token'] = test_file['item_id:token'].fillna(item_id['item_id:token'].max()+test_file[['id_content']].apply(tuple,axis=1).rank(method='dense').astype(int))
test_file['item_id:token'] = test_file['item_id:token'].astype(int)
test_file['user_id:token'] = test_file['user_id:token'].str.slice(start=1).astype('int64')

test_file = test_file[test_file['user_id:token'].isin(interactions['user_id:token'])]

test_filter = content_type[content_type['user_id:token'].isin(test_file['user_id:token'])]
test_filter = test_filter.groupby('user_id:token', sort=False)[content_collumn].agg(set).to_dict()



hsl = [512, 512, 512]
nd = 0.2
md = 0.0
rw = 1e-5

varying_thresholds, statistical_significans = NGCF_model(test_file, content_file, test_filter, hsl, nd, md, rw)


with open('varying_thresholds_NGCF.pickle', 'wb') as handle:
    pickle.dump(varying_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('statistical_significans_NGCF.pickle', 'wb') as handle:
    pickle.dump(statistical_significans, handle, protocol=pickle.HIGHEST_PROTOCOL)