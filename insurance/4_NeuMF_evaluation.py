import pandas as pd
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.models.ncf.ncf_singlenode import NCF
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
interactions = pd.concat([ecommerce_items_b.rename(columns={'item_id':'itemID', 'user_id':'userID'}),
                          ecommerce_services_b.rename(columns={'service_id':'itemID', 'user_id':'userID'}),
                          claim_items_b.rename(columns={'item_id':'itemID', 'user_id':'userID'}),
                          claim_services_b.rename(columns={'service_id':'itemID', 'user_id':'userID'}),
                          info_items_b.rename(columns={'item_id':'itemID', 'user_id':'userID'}),
                          info_services_b.rename(columns={'service_id':'itemID', 'user_id':'userID'}),
                          account_items_b.rename(columns={'item_id':'itemID', 'user_id':'userID'}),
                          account_services_b.rename(columns={'service_id':'itemID', 'user_id':'userID'}),
                          purchase_items_b.rename(columns={'item_id':'itemID', 'user_id':'userID'})])

interactions['itemID'] = interactions['itemID'].astype(int)
interactions['rating'] = 1


interactions = interactions.sort_values(by='userID', ignore_index=True)
interaction_file = "./interactions.csv"
interactions.to_csv(interaction_file, index=False)
s = 42

interaction_data = NCFDataset(train_file=interaction_file, seed=s)


''' NeuMF. '''
def NeuMF_model(data, test_set, filter_dict, latent_factors, EPOCHS, BATCH_SIZE, SEED):
 
    model = NCF(
        n_users=data.n_users, 
        n_items=data.n_items,
        model_type="NeuMF",
        n_factors=latent_factors,
        layer_sizes=[16,8,4],
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        verbose=1,
        seed=SEED
    )
    
    
    model.fit(data)
    
    predict = pd.DataFrame(np.unique(test_set['itemID']))
    predict.columns = ['itemID']

    mrrs = []
    hrs = []
    users = []
    for user in np.unique(test_set['userID']):    
        content = list(test_set[test_set['userID']==user]['itemID'])
            
        predict['userID'] = user
        predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
                       for (_, row) in predict.iterrows()]
        predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
        predictions = predictions[predictions['itemID'].isin(filter_dict[user])]

        mrr_t = []
        hr_t = []
        for t in range(5):
            recommendation_list = list(predictions.nlargest(t+1, 'prediction')['itemID'].reset_index(drop=True))
        
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





purchase_items_t = purchase_items_t.rename(columns={'item_id':'itemID', 'user_id':'userID'})
purchase_items_t['itemID'] = purchase_items_t['itemID'].astype(int)
purchase_items_f = purchase_items_f.rename(columns={'item_id':'itemID', 'user_id':'userID'})
purchase_items_f['itemID'] = purchase_items_f['itemID'].astype(int)

id_user_collumn = 'userID'
content_collumn = 'itemID'
test_filter = purchase_items_f.groupby('userID', sort=False)['itemID'].agg(set).to_dict()
test_file = purchase_items_t
test_file = test_file[[content_collumn, id_user_collumn]]




lf = 10
e = 5
bs = 64

varying_thresholds, statistical_significans = NeuMF_model(interaction_data, test_file, test_filter, lf, e, bs, s)



with open('varying_thresholds_NeuMF.pickle', 'wb') as handle:
    pickle.dump(varying_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('statistical_significans_NeuMF.pickle', 'wb') as handle:
    pickle.dump(statistical_significans, handle, protocol=pickle.HIGHEST_PROTOCOL)