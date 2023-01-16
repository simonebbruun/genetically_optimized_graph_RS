import pandas as pd
from recbole.utils import init_seed
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import NGCF
from recbole.trainer import Trainer
import torch
from recbole.data.interaction import Interaction
import numpy as np


''' Import. '''
ecommerce_items_b = pd.read_csv('ecommerce_items_b.csv')
ecommerce_services_b = pd.read_csv('ecommerce_services_b.csv')
claim_items_b = pd.read_csv('claim_items_b.csv')
claim_services_b = pd.read_csv('claim_services_b.csv')
info_items_b = pd.read_csv('info_items_b.csv')
info_services_b = pd.read_csv('info_services_b.csv')
account_items_b = pd.read_csv('account_items_b.csv')
account_services_b = pd.read_csv('account_services_b.csv')

purchase_items_b = pd.read_csv('purchase_items_b.csv')

purchase_items_tr = pd.read_csv('purchase_items_tr.csv')

purchase_items_f = pd.read_csv('purchase_items_f.csv')


''' Preprocessing. '''
# Add number to ids that signify their type.
ecommerce_items_b['item_id'] = ecommerce_items_b['item_id'].map(lambda x: '2' + str(x))
ecommerce_services_b['service_id'] = ecommerce_services_b['service_id'].map(lambda x: '3' + str(x))
claim_items_b['item_id'] = claim_items_b['item_id'].map(lambda x: '2' + str(x))
claim_services_b['service_id'] = claim_services_b['service_id'].map(lambda x: '3' + str(x))
info_items_b['item_id'] = info_items_b['item_id'].map(lambda x: '2' + str(x))
info_services_b['service_id'] = info_services_b['service_id'].map(lambda x: '3' + str(x))
account_items_b['item_id'] = account_items_b['item_id'].map(lambda x: '2' + str(x))
account_services_b['service_id'] = account_services_b['service_id'].map(lambda x: '3' + str(x))

purchase_items_b['item_id'] = purchase_items_b['item_id'].map(lambda x: '1' + str(x))

purchase_items_tr['item_id'] = purchase_items_tr['item_id'].map(lambda x: '1' + str(x))

purchase_items_f['item_id'] = purchase_items_f['item_id'].map(lambda x: '1' + str(x))


''' Interactions. '''
interactions = pd.concat([ecommerce_items_b.rename(columns={'item_id':'item_id:token', 'user_id':'user_id:token'}),
                          ecommerce_services_b.rename(columns={'service_id':'item_id:token', 'user_id':'user_id:token'}),
                          claim_items_b.rename(columns={'item_id':'item_id:token', 'user_id':'user_id:token'}),
                          claim_services_b.rename(columns={'service_id':'item_id:token', 'user_id':'user_id:token'}),
                          info_items_b.rename(columns={'item_id':'item_id:token', 'user_id':'user_id:token'}),
                          info_services_b.rename(columns={'service_id':'item_id:token', 'user_id':'user_id:token'}),
                          account_items_b.rename(columns={'item_id':'item_id:token', 'user_id':'user_id:token'}),
                          account_services_b.rename(columns={'service_id':'item_id:token', 'user_id':'user_id:token'}),
                          purchase_items_b.rename(columns={'item_id':'item_id:token', 'user_id':'user_id:token'})])

interactions['item_id:token'] = interactions['item_id:token'].astype(int)


# !mkdir \insurance\interactions_train

# interactions[['user_id:token', 'item_id:token']].to_csv('\insurance\interactions_train\interactions_train.inter', index=False, sep='\t')



''' NGCF. '''
def NGCF_model(train_set, filter_dict, hidden_size_list, node_dropout, message_dropout, reg_weight):
     
    parameter_dict = {
        'data_path': '\insurance',
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

    iid_set = dataset.token2id(dataset.iid_field, list(map(str, np.unique(train_set['item_id:token']))))
    iid = np.vstack((iid_set, list(np.unique(train_set['item_id:token'])))).T
    iid = pd.DataFrame(iid, columns=['iid', 'item_id:token'])

    mrrs_3 = []
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
        predictions = predictions[predictions['item_id:token'].isin(filter_dict[user])]

        recommendation_list_3 = list(predictions.nlargest(3, 'prediction')['item_id:token'].reset_index(drop=True))
        
        mrr_3 = 0
        if any(i in recommendation_list_3 for i in content):
            mrr_3 = 1.0/(1+min(np.where(np.isin(recommendation_list_3,content))[0]))
        mrrs_3.append(mrr_3)
    
    return np.mean(mrrs_3)



purchase_items_tr = purchase_items_tr.rename(columns={'item_id':'item_id:token', 'user_id':'user_id:token'})
purchase_items_tr['item_id:token'] = purchase_items_tr['item_id:token'].astype(int)
purchase_items_f = purchase_items_f.rename(columns={'item_id':'item_id:token', 'user_id':'user_id:token'})
purchase_items_f['item_id:token'] = purchase_items_f['item_id:token'].astype(int)

id_user_collumn = 'user_id:token'
content_collumn = 'item_id:token'
train_filter = purchase_items_f.groupby('user_id:token', sort=False)['item_id:token'].agg(set).to_dict()
train_file = purchase_items_tr
train_file = train_file[[content_collumn, id_user_collumn]]



hsl = [64, 64, 64]
nd = 0.0
md = 0.0
rw = 1e-5

mrr = NGCF_model(train_file, train_filter, hsl, nd, md, rw)
