import pandas as pd
from pandasql import sqldf
mysql = lambda q: sqldf(q, globals())


''' Train. '''
sessions_train = pd.read_csv('sessions_train.csv')


ecommerce_items_b = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'item_', '') as int) as item_id

from sessions_train
where action_section = 'e_commerce'
 and action_object like 'item_%'
group by
event_id,
action_object;
''')


ecommerce_services_b = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'service_', '') as int) as service_id

from sessions_train
where action_section = 'e_commerce'
 and action_object like 'service_%'
group by
event_id,
action_object;
''')


claim_items_b = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'item_', '') as int) as item_id

from sessions_train
where action_section = 'claims_reporting'
 and action_object like 'item_%'
group by
event_id,
action_object;
''')


claim_services_b = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'service_', '') as int) as service_id

from sessions_train
where action_section = 'claims_reporting'
 and action_object like 'service_%'
group by
event_id,
action_object;
''')


info_items_b = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'item_', '') as int) as item_id

from sessions_train
where action_section like 'information_%'
 and action_object like 'item_%'
group by
event_id,
action_object;
''')


info_services_b = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'service_', '') as int) as service_id

from sessions_train
where action_section like 'information_%'
 and action_object like 'service_%'
group by
event_id,
action_object;
''')


account_items_b = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'item_', '') as int) as item_id

from sessions_train
where action_section = 'personal_account'
 and action_object like 'item_%'
group by
event_id,
action_object;
''')


account_services_b = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'service_', '') as int) as service_id

from sessions_train
where action_section = 'personal_account'
 and action_object like 'service_%'
group by
event_id,
action_object;
''')



purchase_events_train = pd.read_csv('purchase_events_train.csv')


purchase_items_b = mysql('''
select
event_id as user_id,
cast(replace(item_id, 'item_', '') as int) as item_id

from purchase_events_train
where valid = 0
 and (event_id in (select user_id from ecommerce_items_b group by user_id)
  or event_id in (select user_id from ecommerce_services_b group by user_id)
  or event_id in (select user_id from claim_items_b group by user_id)
  or event_id in (select user_id from claim_services_b group by user_id)
  or event_id in (select user_id from info_items_b group by user_id)
  or event_id in (select user_id from info_services_b group by user_id)
  or event_id in (select user_id from account_items_b group by user_id)
  or event_id in (select user_id from account_services_b group by user_id));
''')



''' Valid. '''
purchase_items_tr = mysql('''
select
event_id as user_id,
cast(replace(item_id, 'item_', '') as int) as item_id

from purchase_events_train
where valid = 1
 and (event_id in (select user_id from ecommerce_items_b group by user_id)
  or event_id in (select user_id from ecommerce_services_b group by user_id)
  or event_id in (select user_id from claim_items_b group by user_id)
  or event_id in (select user_id from claim_services_b group by user_id)
  or event_id in (select user_id from info_items_b group by user_id)
  or event_id in (select user_id from info_services_b group by user_id)
  or event_id in (select user_id from account_items_b group by user_id)
  or event_id in (select user_id from account_services_b group by user_id));
''')


filter_train = pd.read_csv('filter_train.csv')

purchase_items_f = mysql('''
select
event_id as user_id,
cast(replace(item_id, 'item_', '') as int) as item_id

from filter_train
where valid = 1
 and event_id in (select user_id from purchase_items_tr group by user_id);
''')



''' Test. '''
sessions_test = pd.read_csv('sessions_test.csv')


ecommerce_items_t = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'item_', '') as int) as item_id

from sessions_test
where action_section = 'e_commerce'
 and action_object like 'item_%'
group by
event_id,
action_object;
''')


ecommerce_services_t = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'service_', '') as int) as service_id

from sessions_test
where action_section = 'e_commerce'
 and action_object like 'service_%'
group by
event_id,
action_object;
''')


claim_items_t = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'item_', '') as int) as item_id

from sessions_test
where action_section = 'claims_reporting'
 and action_object like 'item_%'
group by
event_id,
action_object;
''')


claim_services_t = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'service_', '') as int) as service_id

from sessions_test
where action_section = 'claims_reporting'
 and action_object like 'service_%'
group by
event_id,
action_object;
''')


info_items_t = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'item_', '') as int) as item_id

from sessions_test
where action_section like 'information_%'
 and action_object like 'item_%'
group by
event_id,
action_object;
''')


info_services_t = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'service_', '') as int) as service_id

from sessions_test
where action_section like 'information_%'
 and action_object like 'service_%'
group by
event_id,
action_object;
''')


account_items_t = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'item_', '') as int) as item_id

from sessions_test
where action_section = 'personal_account'
 and action_object like 'item_%'
group by
event_id,
action_object;
''')


account_services_t = mysql('''
select
event_id as user_id,
cast(replace(action_object, 'service_', '') as int) as service_id

from sessions_test
where action_section = 'personal_account'
 and action_object like 'service_%'
group by
event_id,
action_object;
''')



purchase_events_test = pd.read_csv('purchase_events_test.csv')


purchase_items_t = mysql('''
select
event_id as user_id,
cast(replace(item_id, 'item_', '') as int) as item_id

from purchase_events_test
where (event_id in (select user_id from ecommerce_items_t group by user_id)
  or event_id in (select user_id from ecommerce_services_t group by user_id)
  or event_id in (select user_id from claim_items_t group by user_id)
  or event_id in (select user_id from claim_services_t group by user_id)
  or event_id in (select user_id from info_items_t group by user_id)
  or event_id in (select user_id from info_services_t group by user_id)
  or event_id in (select user_id from account_items_t group by user_id)
  or event_id in (select user_id from account_services_t group by user_id));
''')


filter_test = pd.read_csv('filter_test.csv')


purchase_items_f_t = mysql('''
select
event_id as user_id,
cast(replace(item_id, 'item_', '') as int) as item_id

from filter_test
where event_id in (select user_id from purchase_items_t group by user_id);
''')



''' Export. '''
ecommerce_items_b.to_csv('ecommerce_items_b.csv', index=False)
ecommerce_services_b.to_csv('ecommerce_services_b.csv', index=False)
claim_items_b.to_csv('claim_items_b.csv', index=False)
claim_services_b.to_csv('claim_services_b.csv', index=False)
info_items_b.to_csv('info_items_b.csv', index=False)
info_services_b.to_csv('info_services_b.csv', index=False)
account_items_b.to_csv('account_items_b.csv', index=False)
account_services_b.to_csv('account_services_b.csv', index=False)

purchase_items_b.to_csv('purchase_items_b.csv', index=False)
purchase_items_tr.to_csv('purchase_items_tr.csv', index=False)
purchase_items_f.to_csv('purchase_items_f.csv', index=False)


ecommerce_items_t.to_csv('ecommerce_items_t.csv', index=False)
ecommerce_services_t.to_csv('ecommerce_services_t.csv', index=False)
claim_items_t.to_csv('claim_items_t.csv', index=False)
claim_services_t.to_csv('claim_services_t.csv', index=False)
info_items_t.to_csv('info_items_t.csv', index=False)
info_services_t.to_csv('info_services_t.csv', index=False)
account_items_t.to_csv('account_items_t.csv', index=False)
account_services_t.to_csv('account_services_t.csv', index=False)

purchase_items_t.to_csv('purchase_items_t.csv', index=False)
purchase_items_f_t.to_csv('purchase_items_f_t.csv', index=False)