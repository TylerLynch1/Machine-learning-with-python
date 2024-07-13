# Imports
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

# Record the start time for measuring training duration
start_time = time.time()

# Define the number of folds for cross-validation
num_folds = 5  # You can adjust the number of folds as needed

# Define the cross-validation strategy
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)

# Fit model
knn.fit(x_train, y_train)

# Check if the model is trained before performing cross-validation
if knn:
  try:
    # perform cross-validation for the Decision Tree model
    knn_accuracy = cross_val_score(knn, x_train, y_train, cv=kf, scoring='accuracy')
    knn_precision = cross_val_score(knn, x_train, y_train, cv=kf, scoring='precision')
    knn_recall = cross_val_score(knn, x_train, y_train, cv=kf, scoring='recall')
    knn_f1score = cross_val_score(knn, x_train, y_train, cv=kf, scoring='f1')
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('KNN model was not successfully trained.')

# Print the cross-validation results
print('Cross-Validation With Accuracy: {:.2f}% (+/- {:.2f}%)'.format(knn_accuracy.mean() * 100, knn_accuracy.std() * 2))
print('Cross-Validation With Precision: {:.2f}% (+/- {:.2f}%)'.format(knn_precision.mean() * 100, knn_precision.std() * 2))
print('Cross-Validation With Recall: {:.2f}% (+/- {:.2f}%)'.format(knn_recall.mean() * 100, knn_recall.std() * 2))
print('Cross-Validation With F1-score: {:.2f}% (+/- {:.2f}%)'.format(knn_f1score.mean() * 100, knn_f1score.std() * 2))

# Print time
knn_seconds = time.time() - start_time
minutes = knn_seconds / 60
print('Time to run: {:.2f} seconds'.format(knn2_seconds), '({:.2f} minutes)'.format(minutes))
