import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

# Record the start time for measuring training duration
start_time = time.time()

# Create model
dt = DecisionTreeClassifier(random_state=42)

# Train model
dt.fit(x_train, y_train)

# Check if the model is trained before making predictions
if dt:
  try:
    print('Decision Tree model trained!')
    # Make predictions
    y_pred = dt.predict(x_test)
    # Print metrics
    dt_accuracy, dt_precision, dt_recall, dt_f1score = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=2)
    print('DT Accuracy: {:.2f}%'.format(dt_accuracy * 100), '\nDT Precision: {:.2f}%'.format(dt_precision * 100),
          '\nDT Recall: {:.2f}%'.format(dt_recall * 100), '\nDT F1-score: {:.2f}%'.format(dt_f1score * 100))
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Decision Tree model was not successfully trained.')

dt_seconds = time.time() - start_time
minutes = dt_seconds / 60
print('Time to run: {:.2f} seconds'.format(dt_seconds), '({:.2f} minutes)'.format(minutes))
