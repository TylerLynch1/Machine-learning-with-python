# Imports
import time
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Record the start time for measuring training duration
start_time = time.time()

# Define the number of folds for cross-validation
num_folds = 5  # you can adjust the number of folds as needed

# Define the cross-validation strategy
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define model
dt = DecisionTreeClassifier(random_state=42)

# Train model
dt.fit(x_train, y_train)

# Check if the model is trained before performing cross-validation
if dt:
  try:
    # Perform cross-validation
    dt_accuracy = cross_val_score(dt, x_train, y_train, cv=kf, scoring='accuracy')
    dt_precision = cross_val_score(dt, x_train, y_train, cv=kf, scoring='precision')
    dt_recall = cross_val_score(dt, x_train, y_train, cv=kf, scoring='recall')
    dt_f1score = cross_val_score(dt, x_train, y_train, cv=kf, scoring='f1')
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Decision Tree model was not successfully trained.')
  
# Print the cross-validation results
print('Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)'.format(dt_accuracy.mean() * 100, dt_accuracy.std() * 2))
print('Cross-Validation Precision: {:.2f}% (+/- {:.2f}%)'.format(dt_precision.mean() * 100, dt_precision.std() * 2))
print('Cross-Validation Recall: {:.2f}% (+/- {:.2f}%)'.format(dt_recall.mean() * 100, dt_recall.std() * 2))
print('Cross-Validation F1-score: {:.2f}% (+/- {:.2f}%)'.format(dt_f1score.mean() * 100, dt_f1score.std() * 2))

# Print time
dt_seconds = time.time() - start_time
minutes = dt_seconds / 60
print('Time to run: {:.2f} seconds'.format(dt_seconds), '({:.2f} minutes)'.format(minutes))
