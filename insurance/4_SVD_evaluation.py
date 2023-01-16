import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse.linalg import svds
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
# Add letter to ids that signify their type.
ecommerce_items_b['item_id'] = ecommerce_items_b['item_id'].map(lambda x: 'i' + str(x))
ecommerce_items_b['user_id'] = ecommerce_items_b['user_id'].map(lambda x: 'u' + str(x))
ecommerce_services_b['service_id'] = ecommerce_services_b['service_id'].map(lambda x: 's' + str(x))
ecommerce_services_b['user_id'] = ecommerce_services_b['user_id'].map(lambda x: 'u' + str(x))
claim_items_b['item_id'] = claim_items_b['item_id'].map(lambda x: 'i' + str(x))
claim_items_b['user_id'] = claim_items_b['user_id'].map(lambda x: 'u' + str(x))
claim_services_b['service_id'] = claim_services_b['service_id'].map(lambda x: 's' + str(x))
claim_services_b['user_id'] = claim_services_b['user_id'].map(lambda x: 'u' + str(x))
info_items_b['item_id'] = info_items_b['item_id'].map(lambda x: 'i' + str(x))
info_items_b['user_id'] = info_items_b['user_id'].map(lambda x: 'u' + str(x))
info_services_b['service_id'] = info_services_b['service_id'].map(lambda x: 's' + str(x))
info_services_b['user_id'] = info_services_b['user_id'].map(lambda x: 'u' + str(x))
account_items_b['item_id'] = account_items_b['item_id'].map(lambda x: 'i' + str(x))
account_items_b['user_id'] = account_items_b['user_id'].map(lambda x: 'u' + str(x))
account_services_b['service_id'] = account_services_b['service_id'].map(lambda x: 's' + str(x))
account_services_b['user_id'] = account_services_b['user_id'].map(lambda x: 'u' + str(x))

purchase_items_b['item_id'] = purchase_items_b['item_id'].map(lambda x: 'p' + str(x))
purchase_items_b['user_id'] = purchase_items_b['user_id'].map(lambda x: 'u' + str(x))

purchase_items_t['item_id'] = purchase_items_t['item_id'].map(lambda x: 'p' + str(x))
purchase_items_t['user_id'] = purchase_items_t['user_id'].map(lambda x: 'u' + str(x))

purchase_items_f['item_id'] = purchase_items_f['item_id'].map(lambda x: 'p' + str(x))
purchase_items_f['user_id'] = purchase_items_f['user_id'].map(lambda x: 'u' + str(x))


''' User-item matrix. '''
interactions = pd.concat([ecommerce_items_b.rename(columns={'item_id':'id'}),
                          ecommerce_services_b.rename(columns={'service_id':'id'}),
                          claim_items_b.rename(columns={'item_id':'id'}),
                          claim_services_b.rename(columns={'service_id':'id'}),
                          info_items_b.rename(columns={'item_id':'id'}),
                          info_services_b.rename(columns={'service_id':'id'}),
                          account_items_b.rename(columns={'item_id':'id'}),
                          account_services_b.rename(columns={'service_id':'id'}),
                          purchase_items_b.rename(columns={'item_id':'id'})])


matrix_file = interactions.set_index('user_id')
encoder = MultiLabelBinarizer()
matrix_file = pd.DataFrame(encoder.fit_transform(matrix_file['id'].str.split(',')), index=matrix_file.index, columns=encoder.classes_)
matrix_file = matrix_file.groupby(['user_id']).max()
matrix_file = matrix_file.astype(float)


''' SVD. '''
def svd(user_item_matrix, latent_factors, test_set, filter_dict):
    
    mrrs = []
    hrs = []
    users = []
    for user in np.unique(test_set[:,1]):
        user_item_submatrix = pd.concat([user_item_matrix[user_item_matrix.index==user],
                                          user_item_matrix[~user_item_matrix.index.isin(np.unique(test_set[:,1]))]])
        
        U, sigma, Vt = svds(user_item_submatrix, k = latent_factors)
        sigma = np.diag(sigma)
        R_hat = U@sigma@Vt
        scores = pd.DataFrame(R_hat, columns=user_item_submatrix.columns, index=user_item_submatrix.index)
        
        content = test_set[test_set[:,1]==user][:,0]
            
        score = scores[scores.index==user].T
        score.loc[score.index.difference(filter_dict[user])] = np.nan

        recommendation_list_dropna = score.dropna()
        
        mrr_t = []
        hr_t = []
        for t in range(5):
            recommendation_list = recommendation_list_dropna.nlargest(t+1,user).index
        
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


id_user_collumn = 'user_id'
content_collumn = 'item_id'
test_filter = purchase_items_f.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
test_file = purchase_items_t
test_file = test_file[[content_collumn, id_user_collumn]].to_numpy()

lf = 10


varying_thresholds, statistical_significans = svd(matrix_file, lf, test_file, test_filter) 



with open('varying_thresholds_SVD.pickle', 'wb') as handle:
    pickle.dump(varying_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('statistical_significans_SVD.pickle', 'wb') as handle:
    pickle.dump(statistical_significans, handle, protocol=pickle.HIGHEST_PROTOCOL)
