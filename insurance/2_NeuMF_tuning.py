import pandas as pd
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.models.ncf.ncf_singlenode import NCF
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
def NeuMF_model(data, train_set, filter_dict, latent_factors, EPOCHS, BATCH_SIZE, SEED):
 
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
    
    predict = pd.DataFrame(np.unique(train_set['itemID']))
    predict.columns = ['itemID']

    mrrs_3 = []
    for user in np.unique(train_set['userID']):    
        content = list(train_set[train_set['userID']==user]['itemID'])
            
        predict['userID'] = user
        predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
                       for (_, row) in predict.iterrows()]
        predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
        predictions = predictions[predictions['itemID'].isin(filter_dict[user])]

        recommendation_list_3 = list(predictions.nlargest(3, 'prediction')['itemID'].reset_index(drop=True))
        
        mrr_3 = 0
        if any(i in recommendation_list_3 for i in content):
            mrr_3 = 1.0/(1+min(np.where(np.isin(recommendation_list_3,content))[0]))
        mrrs_3.append(mrr_3)
    
    return np.mean(mrrs_3)



purchase_items_tr = purchase_items_tr.rename(columns={'item_id':'itemID', 'user_id':'userID'})
purchase_items_tr['itemID'] = purchase_items_tr['itemID'].astype(int)
purchase_items_f = purchase_items_f.rename(columns={'item_id':'itemID', 'user_id':'userID'})
purchase_items_f['itemID'] = purchase_items_f['itemID'].astype(int)

id_user_collumn = 'userID'
content_collumn = 'itemID'
train_filter = purchase_items_f.groupby('userID', sort=False)['itemID'].agg(set).to_dict()
train_file = purchase_items_tr
train_file = train_file[[content_collumn, id_user_collumn]]



lf = 15
e = 5
bs = 1024


mrr = NeuMF_model(interaction_data, lf, train_file, train_filter, e, bs, s)



