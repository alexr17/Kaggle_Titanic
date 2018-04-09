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

# we'll be using separate dataframes for each method
df_dt_test = df_test_data.copy()
df_dt_result = df_submission.copy()
grouped = df_dt_test.merge(df_dt_result,on='PassengerId').groupby(['Survived'])

df_alive = grouped.get_group(1) # survivors
df_dead = grouped.get_group(0) # not the survivors

# creates a series that counts the probability of surviving for each occurrence in the column
def survival_prob(df_alive, df_dead, col_name):
    alive_vc = df_alive[col_name].value_counts()
    dead_vc = df_dead[col_name].value_counts()
    print(alive_vc.name)
    
    

survival_prob(df_alive, df_dead, 'Pclass')