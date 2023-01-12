import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
    
with open('users_b.pickle', 'rb') as handle:
    users_b = pickle.load(handle)
    

''' User-item matrix. '''
interactions = pd.concat([general_posts_b_tr[['id_general_posted', 'id_user']],
                          general_sources_b_tr[['id_general_posted', 'id_user']],
                          courses_follows_b_tr[['id_course', 'id_user']],
                          users_follows_b_tr.rename(columns = {'origin_id_user':'id_user'})[['id_user', 'destination_id_user']],
                          ratings_b_tr[['id_general_posted', 'id_user']],
                          post_likes_b_tr[['id_user', 'id_general_posted']],
                          post_comments_b_tr[['id_comment', 'id_general_posted', 'id_user']],
                          post_comment_likes_b_tr[['id_user', 'id_comment']],
                          users_b[['id_user', 'city_id', 'curr_ori_id']]])

# One-hot encoding
encoder = OneHotEncoder()
dummies = encoder.fit_transform(interactions[['id_general_posted', 'id_course', 'destination_id_user', 'id_comment', 'city_id', 'curr_ori_id']]).toarray()
dummy_names = encoder.get_feature_names(['id_general_posted', 'id_course', 'destination_id_user', 'id_comment', 'city_id', 'curr_ori_id'])
interactions1 = pd.concat([interactions.reset_index(drop=True), pd.DataFrame(dummies, columns = dummy_names)], axis=1)
interactions1 = interactions1.drop(['id_general_posted', 'id_course', 'destination_id_user', 'id_comment', 'city_id', 'curr_ori_id',
                                    'id_general_posted_nan', 'id_course_nan', 'destination_id_user_nan', 'id_comment_nan', 'city_id_nan', 'curr_ori_id_nan'], axis=1)

matrix_file = interactions1.groupby(['id_user']).max()


''' KNN. '''
def generate_neighbours(user_item_matrix, user, k_neighbours, content_set):
    similarity = cosine_similarity(user_item_matrix, user_item_matrix[user_item_matrix.index==user])
    similarity = pd.DataFrame(similarity, index=user_item_matrix.index)
    similarity = similarity[similarity.index!=user]
    similarity_neighbours = similarity[similarity.index.isin(content_set.index)]
    similarity_neighbours_k = similarity_neighbours.nlargest(k_neighbours,0)
    
    return similarity_neighbours_k

def generate_scores(user_item_matrix, user, k_neighbours, content_set, filter_dict):
    similarity_neighbours_k = generate_neighbours(user_item_matrix, user, k_neighbours, content_set)
    score = content_set.mul(similarity_neighbours_k.iloc[:,0], axis=0).dropna().sum()
    score[filter_dict[user]] = 0.0
      
    return score

def gen_evaluate_with_set(user_item_matrix, train_set, k_neighbours, content_set, filter_dict):
    mrrs_10 = []

    for user in np.unique(train_set[:,1]):
        content = train_set[train_set[:,1]==user][:,0]
    # generate N recommendations for this user
        score = generate_scores(user_item_matrix, user, k_neighbours, content_set, filter_dict)
        recommendation_list_pos = score[score>0.0]
        recommendation_list_10 = recommendation_list_pos.nlargest(10).index
    
    # calculate MRR and hit rate
        mrr_10 = 0
        if any(i in recommendation_list_10 for i in content):
            mrr_10 = 1.0/(1+min(np.where(np.isin(recommendation_list_10,content))[0]))
        mrrs_10.append(mrr_10)
        
        
    return np.mean(mrrs_10)


train_file_path = 'courses_follows_tr.pickle'
content_collumn = 'id_course'
content_type = courses_follows_b_tr[[content_collumn, 'id_user']]

content_file = content_type.set_index('id_user')
encoder = OneHotEncoder()
dummies = encoder.fit_transform(content_file[[content_collumn]]).toarray()
dummy_names = np.char.replace(encoder.get_feature_names([content_collumn]).astype(str), content_collumn+'_', '')
content_file = pd.DataFrame(dummies, index=content_file.index, columns=dummy_names)
content_file = content_file.groupby(['id_user']).max()

with open(train_file_path, 'rb') as handle:
    train_file = pickle.load(handle)
train_file = train_file[train_file['id_user'].isin(matrix_file.index)]
train_file = train_file[[content_collumn, 'id_user']].to_numpy()

train_filter = content_type[content_type['id_user'].isin(train_file[:,1])]
train_filter = train_filter.groupby('id_user', sort=False)[content_collumn].agg(set).to_dict()


k_list = [40, 50, 60, 70, 80]
mrr_list = []
for k in k_list:
    mrr_list.append(gen_evaluate_with_set(matrix_file, train_file, k, content_file, train_filter))
