import pickle
import numpy as np


''' Import. '''

with open('courses_follows_b_t.pickle', 'rb') as handle:
    courses_follows_b_t = pickle.load(handle)


def most_popular_evaluate(most_popular, test_set, filter_dict):
    
    mrrs = []
    hrs = []
    for user in np.unique(test_set[:,1]):
        content = test_set[test_set[:,1]==user][:,0]
    # generate N recommendations for this user
        score = most_popular.copy()
        score[filter_dict[user]] = 0.0
        recommendation_list_pos = score[score>0.0]
        
        mrr_t = []
        hr_t = []
        for t in range(20):
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
    mrrs = np.stack(mrrs, axis=0)
    hrs = np.stack(hrs, axis=0)
    
    return (np.mean(mrrs, axis=0), np.mean(hrs, axis=0))
    
    
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



with open('varying_thresholds_most_popular.pickle', 'wb') as handle:
    pickle.dump(evaluation, handle, protocol=pickle.HIGHEST_PROTOCOL)