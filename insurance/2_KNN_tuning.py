import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
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

def gen_evaluate_with_set(user_item_matrix, train_set, k_neighbours, content_set, filter_dict):
    mrrs_3 = []

    for user in np.unique(train_set[:,1]):
        content = train_set[train_set[:,1]==user][:,0]
    # generate N recommendations for this user
        score = generate_scores(user_item_matrix, user, k_neighbours, content_set, filter_dict)
        recommendation_list_pos = score[score>0.0]
        recommendation_list_3 = recommendation_list_pos.nlargest(3).index
    
    # calculate MRR and hit rate
        mrr_3 = 0
        if any(i in recommendation_list_3 for i in content):
            mrr_3 = 1.0/(1+min(np.where(np.isin(recommendation_list_3,content))[0]))
        mrrs_3.append(mrr_3)

        
    return np.mean(mrrs_3)


content_file = purchase_items_b.set_index('user_id')
encoder = OneHotEncoder()
dummies = encoder.fit_transform(content_file[['item_id']]).toarray()
content_file = pd.DataFrame(dummies, index=content_file.index, columns=range(1,17))
content_file = content_file.groupby(['user_id']).max()

id_user_collumn = 'user_id'
content_collumn = 'item_id'
train_filter = purchase_items_f.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()

train_file = purchase_items_tr
train_file = train_file[[content_collumn, id_user_collumn]].to_numpy()

k = 80


mrr_3 = gen_evaluate_with_set(matrix_file, train_file, k, content_file, train_filter)

