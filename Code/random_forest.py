# Imports
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

# Record the start time for measuring training duration
start_time = time.time()

# Train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Fit model
rf.fit(x_train, y_train)

# Check if the model is trained before making predictions
if rf:
  try:
    print('Random Forest model trained!')
    # Make predictions
    y_pred = rf.predict(x_test)
    # Print metrics
    rf_accuracy, rf_precision, rf_recall, rf_f1score = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=2)
    print('Accuracy: {:.2f}%'.format(rf_accuracy * 100), '\nPrecision: {:.2f}%'.format(rf_precision * 100),
          '\nRecall: {:.2f}%'.format(rf_recall * 100), '\nF1-score: {:.2f}%'.format(rf_f1score * 100))
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Random Forest model was not successfully trained.')

# Print time
rf_seconds = time.time() - start_time
minutes = rf_seconds / 60
print('Time to run: {:.2f} seconds'.format(rf_seconds), '({:.2f} minutes)'.format(minutes))
