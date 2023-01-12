import pickle
import pandas as pd
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.models.ncf.ncf_singlenode import NCF
import numpy as np


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
    
    
''' Interactions. '''
interactions = pd.concat([general_posts_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'userID'})[['userID', 'id_content']],
                          general_sources_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'userID'})[['userID', 'id_content']],
                          courses_follows_b_t.rename(columns = {'id_course':'id_content', 'id_user':'userID'})[['userID', 'id_content']],
                          users_follows_b_t.rename(columns = {'destination_id_user': 'id_content', 'origin_id_user':'userID'})[['userID', 'id_content']],
                          ratings_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'userID'})[['userID', 'id_content']],
                          post_likes_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'userID'})[['userID', 'id_content']],
                          post_comments_b_t.rename(columns = {'id_general_posted':'id_content', 'id_user':'userID'})[['userID', 'id_content']],
                          post_comments_b_t.rename(columns = {'id_comment':'id_content', 'id_user':'userID'})[['userID', 'id_content']],
                          post_comment_likes_b_t.rename(columns = {'id_comment':'id_content', 'id_user':'userID'})[['userID', 'id_content']]
                          ])

interactions['itemID'] = interactions[['id_content']].apply(tuple,axis=1).rank(method='dense').astype(int)
interactions['userID'] = interactions['userID'].str.slice(start=1).astype('int64')
interactions['rating'] = 1

itemID = interactions[['id_content', 'itemID']].drop_duplicates()


interactions = interactions.sort_values(by='userID', ignore_index=True)
interaction_file = "./interactions.csv"
interactions.to_csv(interaction_file, index=False)
s = 42

interaction_data = NCFDataset(train_file=interaction_file, seed=s)


def NeuMF_model(data, test_set, content_set, filter_dict, latent_factors, EPOCHS, BATCH_SIZE, SEED):
 
    model = NCF(
        n_users=data.n_users, 
        n_items=data.n_items,
        model_type="NeuMF",
        n_factors=latent_factors,
        layer_sizes=[16,8,4],
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        verbose=10,
        seed=SEED
    )
    
    
    model.fit(data)
    
    predict = pd.DataFrame(content_set['itemID'])
    
    mrrs = []
    hrs = []
    for user in np.unique(test_set['userID']):    
        content = list(test_set[test_set['userID']==user]['itemID'])
            
        predict['userID'] = user
        predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
                       for (_, row) in predict.iterrows()]
        predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
        predictions = predictions[~predictions['itemID'].isin(filter_dict[user])]
        
        mrr_t = []
        hr_t = []
        for t in range(20):
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
    mrrs = np.stack(mrrs, axis=0)
    hrs = np.stack(hrs, axis=0)
    
    return (np.mean(mrrs, axis=0), np.mean(hrs, axis=0))



test_file_path = 'courses_follows_t.pickle'
content_collumn = 'itemID'
content_type = courses_follows_b_t.rename(columns = {'id_course':'id_content', 'id_user':'userID'})[['userID', 'id_content']]
content_type = content_type.merge(itemID, how='left', on='id_content')
content_type['userID'] = content_type['userID'].str.slice(start=1).astype('int64')

content_file = itemID[itemID['id_content'].str.startswith('c')]


with open(test_file_path, 'rb') as handle:
    test_file = pickle.load(handle)

test_file = test_file.rename(columns = {'id_course':'id_content', 'id_user':'userID'})[['userID', 'id_content']]
test_file = test_file.merge(itemID, how='left', on='id_content')
test_file['itemID'] = test_file['itemID'].fillna(itemID['itemID'].max()+test_file[['id_content']].apply(tuple,axis=1).rank(method='dense').astype(int))
test_file['itemID'] = test_file['itemID'].astype(int)
test_file['userID'] = test_file['userID'].str.slice(start=1).astype('int64')
test_file['rating'] = 1

test_file = test_file[test_file['userID'].isin(interactions['userID'])]

train_filter = content_type[content_type['userID'].isin(test_file['userID'])]
train_filter = train_filter.groupby('userID', sort=False)[content_collumn].agg(set).to_dict()


lf = 15
e = 100
bs = 32

evaluation = NeuMF_model(interaction_data, test_file, content_file, train_filter, lf, e, bs, s)


with open('varying_thresholds_NeuMF.pickle', 'wb') as handle:
    pickle.dump(evaluation, handle, protocol=pickle.HIGHEST_PROTOCOL)