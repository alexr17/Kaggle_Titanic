import numpy as np
import pandas as pd
import time as time
current_time = time.time()
df_train_data = pd.read_csv('train.csv')
df_test_data = pd.read_csv('test.csv')
df_submission = pd.read_csv('gender_submission.csv')
print_timestamps = True

if print_timestamps:
    print('read_csv took: ' + str(round(time.time()-current_time,4)) + 's')

# going to go through each type of binary classification as it is listed and described on wikipedia
# https://en.wikipedia.org/wiki/Binary_classification#Statistical_binary_classification

# decision trees