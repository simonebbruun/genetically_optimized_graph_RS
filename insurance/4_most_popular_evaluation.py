import pandas as pd
import numpy as np
import pickle


''' Import. '''
purchase_items_b = pd.concat([pd.read_csv('purchase_items_b.csv'), pd.read_csv('purchase_items_tr.csv')])

purchase_items_t = pd.read_csv('purchase_items_t.csv')

purchase_items_f = pd.read_csv('purchase_items_f_t.csv')


def most_popular_evaluate(most_popular, test_set, filter_dict):
   
    mrrs = []
    hrs = []
    users = []
    for user in np.unique(test_set[:,1]):
        content = test_set[test_set[:,1]==user][:,0]
        
        score = most_popular.copy()
        score.loc[score.index.difference(filter_dict[user])] = 0.0
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
    
    
# Evaluating 
id_user_collumn = 'user_id'
content_collumn = 'item_id'
test_filter = purchase_items_f.groupby('user_id', sort=False)['item_id'].agg(set).to_dict()
test_file = purchase_items_t
test_file = test_file[[content_collumn, id_user_collumn]].to_numpy()

mp = purchase_items_b['item_id'].value_counts()


varying_thresholds, statistical_significans = most_popular_evaluate(mp, test_file, test_filter)


with open('varying_thresholds_most_popular.pickle', 'wb') as handle:
    pickle.dump(varying_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('statistical_significans_most_popular.pickle', 'wb') as handle:
    pickle.dump(statistical_significans, handle, protocol=pickle.HIGHEST_PROTOCOL)
