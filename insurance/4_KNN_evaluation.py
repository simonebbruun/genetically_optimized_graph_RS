import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
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


''' User-item matrix. '''
interactions = pd.concat([ecommerce_items_b,
                          ecommerce_services_b,
                          claim_items_b,
                          claim_services_b,
                          info_items_b,
                          info_services_b,
                          account_items_b,
                          account_services_b,
                          purchase_items_b])

# One-hot encoding
encoder = OneHotEncoder()
dummies = encoder.fit_transform(interactions[['item_id', 'service_id']]).toarray()
dummy_names = encoder.get_feature_names(['item_id', 'service_id'])
interactions1 = pd.concat([interactions.reset_index(drop=True), pd.DataFrame(dummies, columns = dummy_names)], axis=1)
interactions1 = interactions1.drop(['item_id', 'service_id', 'item_id_nan', 'service_id_nan'], axis=1)

matrix_file = interactions1.groupby(['user_id']).max()


''' KNN. '''
def generate_neighbours(user_item_matrix, user, k_neighbours, content_set):
    similarity = cosine_similarity(user_item_matrix, user_item_matrix[user_item_matrix.index==user])
    similarity = pd.DataFrame(similarity, index=user_item_matrix.index)
    similarity_neighbours = similarity[similarity.index.isin(content_set.index)]
    similarity_neighbours_k = similarity_neighbours.nlargest(k_neighbours,0)
    
    return similarity_neighbours_k

def generate_scores(user_item_matrix, user, k_neighbours, content_set, filter_dict):
    similarity_neighbours_k = generate_neighbours(user_item_matrix, user, k_neighbours, content_set)
    score = content_set.mul(similarity_neighbours_k.iloc[:,0], axis=0).dropna().sum()
    score.loc[score.index.difference(filter_dict[user])] = 0.0
        
    return score


def gen_evaluate_with_set(user_item_matrix, test_set, k_neighbours, content_set, filter_dict):
    
    mrrs = []
    hrs = []
    users = []
    for user in np.unique(test_set[:,1]):
        content = test_set[test_set[:,1]==user][:,0]

        score = generate_scores(user_item_matrix, user, k_neighbours, content_set, filter_dict)
        recommendation_list_pos = score[score>0.0]
        
        mrr_t = []
        hr_t = []
        for t in range(5):
            recommendation_list = recommendation_list_pos.nlargest(t+1).index

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


content_file = purchase_items_b.set_index('user_id')
encoder = OneHotEncoder()
dummies = encoder.fit_transform(content_file[['item_id']]).toarray()
content_file = pd.DataFrame(dummies, index=content_file.index, columns=range(1,17))
content_file = content_file.groupby(['user_id']).max()

id_user_collumn = 'user_id'
content_collumn = 'item_id'
test_filter = purchase_items_f.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()

test_file = purchase_items_t
test_file = test_file[[content_collumn, id_user_collumn]].to_numpy()

k = 80

varying_thresholds, statistical_significans = gen_evaluate_with_set(matrix_file, test_file, k, content_file, test_filter)



with open('varying_thresholds_KNN.pickle', 'wb') as handle:
    pickle.dump(varying_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('statistical_significans_KNN.pickle', 'wb') as handle:
    pickle.dump(statistical_significans, handle, protocol=pickle.HIGHEST_PROTOCOL)
