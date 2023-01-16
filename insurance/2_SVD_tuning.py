import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from scipy.sparse.linalg import svds


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

purchase_items_tr['item_id'] = purchase_items_tr['item_id'].map(lambda x: 'p' + str(x))
purchase_items_tr['user_id'] = purchase_items_tr['user_id'].map(lambda x: 'u' + str(x))

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
def svd(user_item_matrix, latent_factors, train_set, filter_dict):
    
    mrrs_3 = []
    for user in np.unique(train_set[:,1]):
        user_item_submatrix = pd.concat([user_item_matrix[user_item_matrix.index==user],
                                          user_item_matrix[~user_item_matrix.index.isin(np.unique(train_set[:,1]))]])
        
        U, sigma, Vt = svds(user_item_submatrix, k = latent_factors)
        sigma = np.diag(sigma)
        R_hat = U@sigma@Vt
        scores = pd.DataFrame(R_hat, columns=user_item_submatrix.columns, index=user_item_submatrix.index)
        
        content = train_set[train_set[:,1]==user][:,0]
            
        score = scores[scores.index==user].T
        score.loc[score.index.difference(filter_dict[user])] = np.nan

        recommendation_list_dropna = score.dropna()
        recommendation_list_3 = recommendation_list_dropna.nlargest(3,user).index
        
        mrr_3 = 0
        if any(i in recommendation_list_3 for i in content):
            mrr_3 = 1.0/(1+min(np.where(np.isin(recommendation_list_3,content))[0]))
        mrrs_3.append(mrr_3)
    
    return np.mean(mrrs_3)


id_user_collumn = 'user_id'
content_collumn = 'item_id'
train_filter = purchase_items_f.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
train_file = purchase_items_tr
train_file = train_file[[content_collumn, id_user_collumn]].to_numpy()


lf = 10


mrr_3 = svd(matrix_file, lf, train_file, train_filter) 
