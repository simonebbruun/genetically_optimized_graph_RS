import pandas as pd
from recbole.utils import init_seed
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import NGCF
from recbole.trainer import Trainer
import torch
from recbole.data.interaction import Interaction
import numpy as np
import pickle


''' Import. '''
ecommerce_items_b = pd.concat([pd.read_csv('ecommerce_items_b.csv'), pd.read_csv('ecommerce_items_t.csv')])
ecommerce_services_b = pd.concat([pd.read_csv('ecommerce_services_b.csv'),  pd.read_csv('ecommerce_services_t.csv')])
claim_items_b = pd.concat([pd.read_csv('claim_items_b.csv'), pd.read_csv('claim_items_t.csv')])
claim_services_b = pd.concat([pd.read_csv('claim_services_b.csv'), pd.read_csv('claim_services_t.csv')])
info_items_b = pd.concat([pd.read_csv('info_items_b.csv'), pd.read_csv('info_items_t.csv')])
info_services_b = pd.concat([pd.read_csv('info_services_b.csv'), pd.read_csv('info_services_t.csv')])
account_items_b = pd.concat([pd.read_csv('account_items_b.csv'), pd.read_csv('account_items_t.csv')])
account_services_b = pd.concat([pd.read_csv('account_services_b.csv'), pd.read_csv('account_services_t.csv')])
purchase_items_b = pd.concat([pd.read_csv('purchase_items_b.csv'), pd.read_csv('purchase_items_tr.csv')])

purchase_items_t = pd.read_csv('purchase_items_t.csv')

purchase_items_f = pd.read_csv('purchase_items_f_t.csv')


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

purchase_items_t['item_id'] = purchase_items_t['item_id'].map(lambda x: '1' + str(x))

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


# !mkdir \insurance\interactions_test

# interactions[['user_id:token', 'item_id:token']].to_csv('\insurance\interactions_test\interactions_test.inter', index=False, sep='\t')



''' NGCF. '''
def NGCF_model(test_set, filter_dict, hidden_size_list, node_dropout, message_dropout, reg_weight):
     
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
    
    config = Config(model='NGCF', dataset='interactions_test', config_dict=parameter_dict)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = NGCF(config, train_data.dataset).to(config['device'])
    
    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data)

    iid_set = dataset.token2id(dataset.iid_field, list(map(str, np.unique(test_set['item_id:token']))))
    iid = np.vstack((iid_set, list(np.unique(test_set['item_id:token'])))).T
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
        predictions = predictions[predictions['item_id:token'].isin(filter_dict[user])]
        
        mrr_t = []
        hr_t = []
        for t in range(5):
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
    
    return (np.mean(mrrs, axis=0), np.mean(hrs, axis=0)), (mrrs[:,2], hrs[:,2], users)


purchase_items_t = purchase_items_t.rename(columns={'item_id':'item_id:token', 'user_id':'user_id:token'})
purchase_items_t['item_id:token'] = purchase_items_t['item_id:token'].astype(int)
purchase_items_f = purchase_items_f.rename(columns={'item_id':'item_id:token', 'user_id':'user_id:token'})
purchase_items_f['item_id:token'] = purchase_items_f['item_id:token'].astype(int)

id_user_collumn = 'user_id:token'
content_collumn = 'item_id:token'
test_filter = purchase_items_f.groupby('user_id:token', sort=False)['item_id:token'].agg(set).to_dict()
test_file = purchase_items_t
test_file = test_file[[content_collumn, id_user_collumn]]



hsl = [64, 64, 64]
nd = 0.2
md = 0.2
rw = 1e-4


varying_thresholds, statistical_significans = NGCF_model(test_file, test_filter, hsl, nd, md, rw)



with open(r'R:\Skriv\Simone\Recommender System\Data\varying_thresholds_NGCF.pickle', 'wb') as handle:
    pickle.dump(varying_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(r'R:\Skriv\Simone\Recommender System\Data\statistical_significans_NGCF.pickle', 'wb') as handle:
    pickle.dump(statistical_significans, handle, protocol=pickle.HIGHEST_PROTOCOL)