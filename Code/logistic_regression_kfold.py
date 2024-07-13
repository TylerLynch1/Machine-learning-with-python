# Imports
import time
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

# Record the start time for measuring training duration
start_time = time.time()

# Define the number of folds for cross-validation
num_folds = 5  # you can adjust the number of folds as needed

# Define the cross-validation strategy
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define model
logreg = LogisticRegression()

# train model
logreg.fit(x_train, y_train)

# Check if the model is trained before performing cross-validation
if logreg:
  try:
    # Perform cross-validation
    lr_accuracy = cross_val_score(logreg, x_train, y_train, cv=kf, scoring='accuracy')
    lr_precision = cross_val_score(logreg, x_train, y_train, cv=kf, scoring='precision')
    lr_recall = cross_val_score(logreg, x_train, y_train, cv=kf, scoring='recall')
    lr_f1score = cross_val_score(logreg, x_train, y_train, cv=kf, scoring='f1')
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Logistic Regression model was not successfully trained.')
  
# Print the cross-validation results
print('Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)'.format(lr_accuracy.mean() * 100, lr_accuracy.std() * 2))
print('Cross-Validation Precision: {:.2f}% (+/- {:.2f}%)'.format(lr_precision.mean() * 100, lr_precision.std() * 2))
print('Cross-Validation Recall: {:.2f}% (+/- {:.2f}%)'.format(lr_recall.mean() * 100, lr_recall.std() * 2))
print('Cross-Validation F1-score: {:.2f}% (+/- {:.2f}%)'.format(lr_f1score.mean() * 100, lr_f1score.std() * 2))

# Print time
lr_seconds = time.time() - start_time
minutes = lr_seconds / 60
print('Time to run: {:.2f} seconds'.format(lr_seconds), '({:.2f} minutes)'.format(minutes))
