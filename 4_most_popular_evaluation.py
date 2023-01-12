import pickle
import numpy as np


''' Import. '''
with open('courses_follows_b_t.pickle', 'rb') as handle:
    courses_follows_b_t = pickle.load(handle)


def most_popular_evaluate(most_popular, test_set, filter_dict):
    mrrs_5 = []
    mrrs_10 = []
    hrs_5 = []
    hrs_10 = []
    users = []

    for user in np.unique(test_set[:,1]):
        content = test_set[test_set[:,1]==user][:,0]
    # generate N recommendations for this user
        score = most_popular.copy()
        score[filter_dict[user]] = 0.0
        recommendation_list_pos = score[score>0.0]
        recommendation_list_10 = recommendation_list_pos.nlargest(10).index
        recommendation_list_5 = recommendation_list_pos.nlargest(5).index
    # calculate MRR and hit rate
        mrr_5 = 0    
        mrr_10 = 0
        hr_5 = 0
        hr_10 = 0
        if any(i in recommendation_list_10 for i in content):
            mrr_10 = 1.0/(1+min(np.where(np.isin(recommendation_list_10,content))[0]))
            hr_10 = 1
            if any(i in recommendation_list_5 for i in content):
                mrr_5 = 1.0/(1+min(np.where(np.isin(recommendation_list_5,content))[0]))
                hr_5 = 1
        mrrs_5.append(mrr_5)
        mrrs_10.append(mrr_10)
        hrs_5.append(hr_5)        
        hrs_10.append(hr_10)
        users.append(user)
        
    print("MRR5 %f" % np.mean(mrrs_5))
    print("MRR10 %f" % np.mean(mrrs_10))
    print("HR5 %f" % np.mean(hrs_5))    
    print("HR10 %f" % np.mean(hrs_10))  
    
    return (mrrs_5, mrrs_10, hrs_5, hrs_10, users)
    
    
# Evaluating 
test_file_path = 'courses_follows_t.pickle'
content_collumn = 'id_course'
content_type = courses_follows_b_t[[content_collumn, 'id_user']]


with open(test_file_path, 'rb') as handle:
    test_file = pickle.load(handle)
test_file = test_file[[content_collumn, 'id_user']].to_numpy()

test_filter = content_type[content_type['id_user'].isin(test_file[:,1])]
test_filter = test_filter.groupby('id_user', sort=False)[content_collumn].agg(set).to_dict()


mp = courses_follows_b_t[content_collumn].value_counts()

evaluation = most_popular_evaluate(mp, test_file, test_filter)


with open('statistical_significans_most_popular.pickle', 'wb') as handle:
    pickle.dump(evaluation, handle, protocol=pickle.HIGHEST_PROTOCOL)
