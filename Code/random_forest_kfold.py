# Imports
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import KFold, cross_val_score

# Record the start time for measuring training duration
start_time = time.time()

# Define the number of folds for cross-validation
num_folds = 5  # You can adjust the number of folds as needed

# Define the cross-validation strategy
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Fit model
rf.fit(x_train, y_train)

# Check if the model is trained before performing cross-validation
if rf:
  try:
    # Perform cross-validation
    rf_accuracy = cross_val_score(rf, x_train, y_train, cv=kf, scoring='accuracy')
    rf_precision = cross_val_score(rf, x_train, y_train, cv=kf, scoring='precision')
    rf_recall = cross_val_score(rf, x_train, y_train, cv=kf, scoring='recall')
    rf_f1score = cross_val_score(rf, x_train, y_train, cv=kf, scoring='f1')
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Random Forest model was not successfully trained.')

# Print the cross-validation results
print('Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)'.format(rf_accuracy.mean() * 100, rf_accuracy.std() * 2))
print('Cross-Validation Precision: {:.2f}% (+/- {:.2f}%)'.format(rf_precision.mean() * 100, rf_precision.std() * 2))
print('Cross-Validation Recall: {:.2f}% (+/- {:.2f}%)'.format(rf_recall.mean() * 100, rf_recall.std() * 2))
print('Cross-Validation F1-score: {:.2f}% (+/- {:.2f}%)'.format(rf_f1score.mean() * 100, rf_f1score.std() * 2))

# Print time
rf_seconds = time.time() - start_time
minutes = rf_seconds / 60
print('Time to run: {:.2f} seconds'.format(rf_seconds), '({:.2f} minutes)'.format(minutes))
