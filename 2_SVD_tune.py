import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.sparse.linalg import svds


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
    

''' User-item matrix. '''
interactions = pd.concat([general_posts_b_tr.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          general_sources_b_tr.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          courses_follows_b_tr.rename(columns = {'id_course':'id_content'})[['id_content', 'id_user']],
                          users_follows_b_tr.rename(columns = {'destination_id_user': 'id_content', 'origin_id_user':'id_user'})[['id_user', 'id_content']],
                          ratings_b_tr.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          post_likes_b_tr.rename(columns = {'id_general_posted':'id_content'})[['id_user', 'id_content']],
                          post_comments_b_tr.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          post_comments_b_tr.rename(columns = {'id_comment':'id_content'})[['id_content', 'id_user']],
                          post_comment_likes_b_tr.rename(columns = {'id_comment':'id_content'})[['id_user', 'id_content']]
                          ])

# One-hot encoding
encoder = OneHotEncoder()
dummies = encoder.fit_transform(interactions[['id_content']]).toarray()
dummy_names = np.char.replace(encoder.get_feature_names(['id_content']).astype(str), 'id_content_', '')
interactions1 = pd.concat([interactions.reset_index(drop=True), pd.DataFrame(dummies, columns = dummy_names)], axis=1)
interactions1 = interactions1.drop(['id_content'], axis=1)

matrix_file = interactions1.groupby(['id_user']).max()


''' SVD. '''
def svd(user_item_matrix, latent_factors, train_set, content_set, filter_dict):
    U, sigma, Vt = svds(user_item_matrix, k = latent_factors)
    sigma = np.diag(sigma)
    R_hat = U@sigma@Vt
    scores = pd.DataFrame(R_hat, columns=user_item_matrix.columns, index=user_item_matrix.index)
    
    mrrs_10 = []
    for user in np.unique(train_set[:,1]):    
        content = train_set[train_set[:,1]==user][:,0]
            
        score = scores[scores.index==user].T
        score = score[score.index.isin(content_set)]
        score = score[~score.index.isin(filter_dict[user])]

        recommendation_list_10 = score.nlargest(10,user).index
        
        mrr_10 = 0
        if any(i in recommendation_list_10 for i in content):
            mrr_10 = 1.0/(1+min(np.where(np.isin(recommendation_list_10,content))[0]))
        mrrs_10.append(mrr_10)
    
    return np.mean(mrrs_10)


train_file_path = 'courses_follows_tr.pickle'
content_collumn = 'id_course'
content_type = courses_follows_b_tr[[content_collumn, 'id_user']]

content_file = content_type[content_collumn].unique()

with open(train_file_path, 'rb') as handle:
    train_file = pickle.load(handle)
train_file = train_file[train_file['id_user'].isin(matrix_file.index)]
train_file = train_file[[content_collumn, 'id_user']].to_numpy()

train_filter = content_type[content_type['id_user'].isin(train_file[:,1])]
train_filter = train_filter.groupby('id_user', sort=False)[content_collumn].agg(set).to_dict()


lf_list = [10, 20, 30, 40, 50]
mrr_list = []
for lf in lf_list:
    mrr_list.append(svd(matrix_file, lf, train_file, content_file, train_filter))