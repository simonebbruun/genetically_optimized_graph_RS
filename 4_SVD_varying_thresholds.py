import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.sparse.linalg import svds


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
    

''' User-item matrix. '''
interactions = pd.concat([general_posts_b_t.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          general_sources_b_t.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          courses_follows_b_t.rename(columns = {'id_course':'id_content'})[['id_content', 'id_user']],
                          users_follows_b_t.rename(columns = {'destination_id_user': 'id_content', 'origin_id_user':'id_user'})[['id_user', 'id_content']],
                          ratings_b_t.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          post_likes_b_t.rename(columns = {'id_general_posted':'id_content'})[['id_user', 'id_content']],
                          post_comments_b_t.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          post_comments_b_t.rename(columns = {'id_comment':'id_content'})[['id_content', 'id_user']],
                          post_comment_likes_b_t.rename(columns = {'id_comment':'id_content'})[['id_user', 'id_content']]
                          ])

# One-hot encoding
encoder = OneHotEncoder()
dummies = encoder.fit_transform(interactions[['id_content']]).toarray()
dummy_names = np.char.replace(encoder.get_feature_names(['id_content']).astype(str), 'id_content_', '')
interactions1 = pd.concat([interactions.reset_index(drop=True), pd.DataFrame(dummies, columns = dummy_names)], axis=1)
interactions1 = interactions1.drop(['id_content'], axis=1)

matrix_file = interactions1.groupby(['id_user']).max()


''' SVD. '''
def svd(user_item_matrix, latent_factors, test_set, content_set, filter_dict):
    U, sigma, Vt = svds(user_item_matrix, k = latent_factors)
    sigma = np.diag(sigma)
    R_hat = U@sigma@Vt
    scores = pd.DataFrame(R_hat, columns=user_item_matrix.columns, index=user_item_matrix.index)
    
    mrrs = []
    hrs = []
    for user in np.unique(test_set[:,1]):    
        content = test_set[test_set[:,1]==user][:,0]
            
        score = scores[scores.index==user].T
        score = score[score.index.isin(content_set)]
        score = score[~score.index.isin(filter_dict[user])]
        
        mrr_t = []
        hr_t = []
        for t in range(20):
            recommendation_list = score.nlargest(t+1,user).index
            
            mrr = 0
            hr = 0
            if any(i in recommendation_list for i in content):
                mrr = 1.0/(1+min(np.where(np.isin(recommendation_list,content))[0]))
                hr = 1
            mrr_t.append(mrr)
            hr_t.append(hr)
        mrrs.append(mrr_t)
        hrs.append(hr_t)        
    mrrs = np.stack(mrrs, axis=0)
    hrs = np.stack(hrs, axis=0)
    
    return (np.mean(mrrs, axis=0), np.mean(hrs, axis=0))


test_file_path = 'courses_follows_t.pickle'
content_collumn = 'id_course'
content_type = courses_follows_b_t[[content_collumn, 'id_user']]

content_file = content_type[content_collumn].unique()

with open(test_file_path, 'rb') as handle:
    test_file = pickle.load(handle)
test_file = test_file[test_file['id_user'].isin(matrix_file.index)]
test_file = test_file[[content_collumn, 'id_user']].to_numpy()

test_filter = content_type[content_type['id_user'].isin(test_file[:,1])]
test_filter = test_filter.groupby('id_user', sort=False)[content_collumn].agg(set).to_dict()


lf = 30
evaluation = svd(matrix_file, lf, test_file, content_file, test_filter)



with open('varying_thresholds_SVD.pickle', 'wb') as handle:
    pickle.dump(evaluation, handle, protocol=pickle.HIGHEST_PROTOCOL)