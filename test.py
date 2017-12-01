# Test Submission 

from sklearn.metrics import log_loss
import numpy as np
import pandas as pd

train_labels = pd.read_csv("data/train_labels.csv", delimiter=',')

p = train_labels['is_duplicate'].mean() # Our predicted probability
print('Predicted score:', log_loss(train_labels['is_duplicate'], np.zeros_like(train_labels['is_duplicate']) + p))

df_test = pd.read_csv('./data/test_data.csv', delimiter=',')
sub = pd.DataFrame({'is_duplicate': 1,'test_id': df_test['test_id']}, columns=['test_id', 'is_duplicate'])
sub.to_csv('output/naive_submission.csv', index=False)
sub.head()