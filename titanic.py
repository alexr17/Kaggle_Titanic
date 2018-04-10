import numpy as np
import pandas as pd
import time as time
current_time = time.time()
df_train_data = pd.read_csv('train.csv')
df_test_data = pd.read_csv('test.csv')
df_submission = pd.read_csv('gender_submission.csv')
print_timestamps = False

if print_timestamps:
    print('read_csv took: ' + str(round(time.time()-current_time,4)) + 's')

# going to go through each type of binary classification as it is listed and described on wikipedia
# https://en.wikipedia.org/wiki/Binary_classification#Statistical_binary_classification

# these are six different ways of implementing binary classification methods
# i will attempt to write code for each one

# decision trees
# random forests
# bayesian networks
# support network machines
# neural networks
# logistic regression

#  ---- decision trees ----

grouped = df_train_data.groupby(['Survived'])

df_alive = grouped.get_group(1) # survivors
df_dead = grouped.get_group(0) # not the survivors

# creates a dataframe that counts the probability of surviving for each occurrence in the column
def survival_prob(df_alive, df_dead, col_name):
    alive_vc = df_alive[col_name].value_counts()
    dead_vc = df_dead[col_name].value_counts()
    alive_vc.name = 'alive'
    dead_vc.name = 'dead'
    
    #join the series' together
    df_all_vc = pd.concat([alive_vc,dead_vc],axis=1)

    #removing NaN
    df_all_vc = df_all_vc.fillna(0)

    #total in each group
    df_all_vc['total'] = df_all_vc['alive'] + df_all_vc['dead']

    #percent survived
    df_all_vc[col_name] = df_all_vc['alive'] / df_all_vc['total']
    
    #if the column contains many values that can't be grouped together
    if (len(df_all_vc) > 10):
        if (col_name == 'Name'): #gonna do this later
            return
        elif (col_name == 'Cabin'):
            #grouping and slicing by cabin letter: A->T
            df_all_vc.index = df_all_vc.index.map(lambda x: x[:1])
            df_cabin_groups = df_all_vc.groupby(level=0).sum()
            df_cabin_groups['Cabin'] = df_cabin_groups['alive']/df_cabin_groups['total']
            return df_cabin_groups
        elif (col_name == 'Ticket'):
            return
        else: #age or fare
            groups = []
            if (col_name == 'Age'):
                groups = [0,8,13,18,22,25,30,35,45,50,60,80]
            else:
                groups = [-1,7.5,8,10,15,25,35,70,600]
            # the following is a super powerful function that will group the values using half bins
            df_groups = df_all_vc.groupby(pd.cut(df_all_vc.index.values,groups)).sum()
            df_groups[col_name] = df_groups['alive']/df_groups['total']
            return df_groups

    return df_all_vc

# #outputting to csv
# for col_name in df_train_data.columns.values:
#     if (str(col_name) != 'Name') & (str(col_name) != 'Ticket'): #not using these categories
#         try:
#             df_out = survival_prob(df_alive, df_dead, col_name)
#             df_out.to_csv(col_name + '.csv', index = True)
#         except:
#             print(df_out)
#             print(col_name)

dict_df = {} #making a dictionary of dataframes
for col_name in df_train_data.columns.values:
    if (str(col_name) != 'Name') & (str(col_name) != 'Ticket'): #not using these categories
        dict_df[col_name] = pd.read_csv(col_name + '.csv', index_col=0)

age_groups = [0,8,13,18,22,25,30,35,45,50,60,80]
fare_groups = [-1,7.5,8,10,15,25,35,70,600]

def grouping_str(cat, val):
    if (cat == 'age'):
        cat_group = age_groups
    else:
        cat_group = fare_groups
    string = '('
    for i, x in enumerate(cat_group):
        if (x > val):
            string = string + x + ', ' + cat_group[i-1] + ']'
            return string

print(grouping_str('age', 20))

