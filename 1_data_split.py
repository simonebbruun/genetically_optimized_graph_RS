import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
import pickle

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


courses_follows_df = pd.read_csv('courses_follow.csv')
users_follows_df = pd.read_csv('users_follow.csv')
general_sources_df = pd.read_csv('general_sources.csv')
ratings_df = pd.read_csv('general_sources_ratings.csv')
general_posts_df = pd.read_csv('general_posts.csv')
users_df = pd.read_csv('users.csv')
post_likes_df = pd.read_csv('general_posts_likes.csv')
post_comments_df = pd.read_csv('general_posted_objects_comments.csv')
post_comment_likes_df = pd.read_csv('general_posted_objects_comments_likes.csv')
origins_df = pd.read_csv('origins.csv')
users_df = pd.read_csv('users.csv')


courses_follows_df['id_course'] = courses_follows_df['id_course'].map(lambda x: 'c' + str(x))
courses_follows_df['id_user'] = courses_follows_df['id_user'].map(lambda x: 'u' + str(x))

users_follows_df['origin_id_user'] = users_follows_df['origin_id_user'].map(lambda x: 'u' + str(x))
users_follows_df['destination_id_user'] = users_follows_df['destination_id_user'].map(lambda x: 'u' + str(x))

general_sources_df['id_general_posted'] = general_sources_df['id_general_posted'].map(lambda x: 's' + str(x))
general_sources_df['id_user'] = general_sources_df['id_user'].map(lambda x: 'u' + str(x))

ratings_df['id_general_posted'] = ratings_df['id_general_posted'].map(lambda x: 's' + str(x))
ratings_df['id_user'] = ratings_df['id_user'].map(lambda x: 'u' + str(x))

general_posts_df['id_general_posted'] = general_posts_df['id_general_posted'].map(lambda x: 'p' + str(x))
general_posts_df['id_user'] = general_posts_df['id_user'].map(lambda x: 'u' + str(x))

post_likes_df['id_user'] = post_likes_df['id_user'].map(lambda x: 'u' + str(x))
post_likes_df['id_general_posted'] = post_likes_df['id_general_posted'].map(lambda x: 'p' + str(x))

post_comments_df['id_user'] = post_comments_df['id_user'].map(lambda x: 'u' + str(x))
post_comments_df['id_general_posted'] = post_comments_df['id_general_posted'].map(lambda x: 'p' + str(x))
post_comments_df['id_comment'] = post_comments_df['id_comment'].map(lambda x: 'm' + str(x))

post_comment_likes_df['id_user'] = post_comment_likes_df['id_user'].map(lambda x: 'u' + str(x))
post_comment_likes_df['id_comment'] = post_comment_likes_df['id_comment'].map(lambda x: 'm' + str(x))

users_df['id_user'] = users_df['id_user'].map(lambda x: 'u' + str(x))
users_df['city_id'] = users_df['city_id'].map(lambda x: 't' + str(x))
users_df['curr_ori_id'] = users_df['curr_ori_id'].map(lambda x: 'o' + str(x))


excluded_users = ['u2224989793433945089', 'u2224990815359337474', 'u2248860000527057963', 'u2224991390641685507']
def exclude_users(df, user_id_collumn, excluded_users = excluded_users):
    return df[~df[user_id_collumn].isin(excluded_users)]


courses_follows_df = exclude_users(courses_follows_df, 'id_user')
users_follows_df = exclude_users(users_follows_df, 'origin_id_user')
users_follows_df = exclude_users(users_follows_df, 'destination_id_user')
general_sources_df = exclude_users(general_sources_df, 'id_user')
ratings_df = exclude_users(ratings_df, 'id_user')
general_posts_df = exclude_users(general_posts_df, 'id_user')
post_likes_df = exclude_users(post_likes_df, 'id_user')
post_comments_df = exclude_users(post_comments_df, 'id_user')
post_comment_likes_df = exclude_users(post_comment_likes_df, 'id_user')
users_df = exclude_users(users_df, 'id_user')


# Sparsity
interactions = pd.concat([general_posts_df.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          general_sources_df.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          courses_follows_df.rename(columns = {'id_course':'id_content'})[['id_content', 'id_user']],
                          users_follows_df.rename(columns = {'destination_id_user': 'id_content', 'origin_id_user':'id_user'})[['id_user', 'id_content']],
                          ratings_df.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          post_likes_df.rename(columns = {'id_general_posted':'id_content'})[['id_user', 'id_content']],
                          post_comments_df.rename(columns = {'id_general_posted':'id_content'})[['id_content', 'id_user']],
                          post_comments_df.rename(columns = {'id_comment':'id_content'})[['id_content', 'id_user']],
                          post_comment_likes_df.rename(columns = {'id_comment':'id_content'})[['id_user', 'id_content']]
                          ])

n_interactions = len(interactions)
n_users = len(interactions['id_user'].unique())
n_contents = len(interactions['id_content'].unique())

sparsity = 1-(n_interactions/(n_users*n_contents))


users_followed_any_course = set(courses_follows_df['id_user'].tolist())
last_interactions = []
for id_user in users_followed_any_course:
    last_interactions.append(courses_follows_df[courses_follows_df['created_at'] == max(courses_follows_df[courses_follows_df['id_user'] == id_user]['created_at'])])
last_interactions = list(map(lambda x: x.values[0], last_interactions))


def split_by_last_interaction(df, user_column, last_interactions):
    df_train = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns = df.columns)
    for last_interaction in last_interactions:
        before = df[(df[user_column] == last_interaction[1]) & (df['created_at'] < last_interaction[2])]
        if (before.shape[0] > 0):
            df_train = df_train.append(before, ignore_index = True)
            after = df[(df[user_column] == last_interaction[1]) & (df['created_at'] >= last_interaction[2])]
            df_test = df_test.append(after, ignore_index = True)
        else:
            after = df[(df[user_column] == last_interaction[1]) & (df['created_at'] >= last_interaction[2])]
            df_train = df_train.append(after, ignore_index = True)
    return (df_train, df_test)


(courses_follows_b_t, courses_follows_t) = split_by_last_interaction(courses_follows_df, 'id_user', last_interactions)


(general_posts_b_t, general_posts_t) = split_by_last_interaction(general_posts_df, 'id_user', last_interactions)
(general_sources_b_t, general_sources_t) = split_by_last_interaction(general_sources_df, 'id_user', last_interactions)
(users_follows_b_t, users_follows_t) = split_by_last_interaction(users_follows_df, 'origin_id_user', last_interactions)
(ratings_b_t, ratings_t) = split_by_last_interaction(ratings_df, 'id_user', last_interactions)
(post_likes_b_t, post_likes_t) = split_by_last_interaction(post_likes_df, 'id_user', last_interactions)
(post_comments_b_t, post_comments_t) = split_by_last_interaction(post_comments_df, 'id_user', last_interactions)
(post_comment_likes_b_t, post_comment_likes_t) = split_by_last_interaction(post_comment_likes_df, 'id_user', last_interactions)
users_b = users_df


users_followed_any_course = set(courses_follows_b_t['id_user'].tolist())
last_interactions = []
for id_user in users_followed_any_course:
    last_interactions.append(courses_follows_b_t[courses_follows_b_t['created_at'] == max(courses_follows_b_t[courses_follows_b_t['id_user'] == id_user]['created_at'])])
last_interactions = list(map(lambda x: x.values[0], last_interactions))


(courses_follows_b_tr, courses_follows_tr) = split_by_last_interaction(courses_follows_b_t, 'id_user', last_interactions)


(general_posts_b_tr, general_posts_tr) = split_by_last_interaction(general_posts_b_t, 'id_user', last_interactions)
(general_sources_b_tr, general_sources_tr) = split_by_last_interaction(general_sources_b_t, 'id_user', last_interactions)
(users_follows_b_tr, users_follows_tr) = split_by_last_interaction(users_follows_b_t, 'origin_id_user', last_interactions)
(ratings_b_tr, ratings_tr) = split_by_last_interaction(ratings_b_t, 'id_user', last_interactions)
(post_likes_b_tr, post_likes_tr) = split_by_last_interaction(post_likes_b_t, 'id_user', last_interactions)
(post_comments_b_tr, post_comments_tr) = split_by_last_interaction(post_comments_b_t, 'id_user', last_interactions)
(post_comment_likes_b_tr, post_comment_likes_tr) = split_by_last_interaction(post_comment_likes_b_t, 'id_user', last_interactions)


with open('general_posts_b_tr.pickle', 'wb') as handle:
    pickle.dump(general_posts_b_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('general_posts_b_t.pickle', 'wb') as handle:
    pickle.dump(general_posts_b_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('general_sources_b_tr.pickle', 'wb') as handle:
    pickle.dump(general_sources_b_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('general_sources_b_t.pickle', 'wb') as handle:
    pickle.dump(general_sources_b_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('courses_follows_b_tr.pickle', 'wb') as handle:
    pickle.dump(courses_follows_b_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('courses_follows_b_t.pickle', 'wb') as handle:
        pickle.dump(courses_follows_b_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('users_follows_b_tr.pickle', 'wb') as handle:
    pickle.dump(users_follows_b_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('users_follows_b_t.pickle', 'wb') as handle:
    pickle.dump(users_follows_b_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('ratings_b_tr.pickle', 'wb') as handle:
    pickle.dump(ratings_b_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('ratings_b_t.pickle', 'wb') as handle:
    pickle.dump(ratings_b_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('post_likes_b_tr.pickle', 'wb') as handle:
    pickle.dump(post_likes_b_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('post_likes_b_t.pickle', 'wb') as handle:
    pickle.dump(post_likes_b_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('post_comments_b_tr.pickle', 'wb') as handle:
    pickle.dump(post_comments_b_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('post_comments_b_t.pickle', 'wb') as handle:
    pickle.dump(post_comments_b_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('post_comment_likes_b_tr.pickle', 'wb') as handle:
    pickle.dump(post_comment_likes_b_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('post_comment_likes_b_t.pickle', 'wb') as handle:
    pickle.dump(post_comment_likes_b_t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('users_b.pickle', 'wb') as handle:
    pickle.dump(users_b, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

with open('courses_follows_tr.pickle', 'wb') as handle:
    pickle.dump(courses_follows_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('courses_follows_t.pickle', 'wb') as handle:
    pickle.dump(courses_follows_t, handle, protocol=pickle.HIGHEST_PROTOCOL)