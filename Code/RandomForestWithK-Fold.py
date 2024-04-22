start_time = time.time()
# |------------------------------------------------------ Random Forest Classification (With Cross-validation) ------------------------------------------------------|
# define the number of folds for cross-validation
num_folds = 5  # You can adjust the number of folds as needed

# Define the cross-validation strategy
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
# fit model
rf.fit(x_train, y_train)

if rf:
  try:
    # Perform cross-validation for the Decision Tree model
    rf2_accuracy = cross_val_score(rf, x_train, y_train, cv=kf, scoring='accuracy')
    rf2_precision = cross_val_score(rf, x_train, y_train, cv=kf, scoring='precision')
    rf2_recall = cross_val_score(rf, x_train, y_train, cv=kf, scoring='recall')
    rf2_f1score = cross_val_score(rf, x_train, y_train, cv=kf, scoring='f1')
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Random Forest model was not successfully trained.')
# |------------------------------------------------------ CALCULATE STATISTICS ------------------------------------------------------|
# Print the cross-validation results
print('Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)'.format(rf2_accuracy.mean() * 100, rf2_accuracy.std() * 2))
print('Cross-Validation Precision: {:.2f}% (+/- {:.2f}%)'.format(rf2_precision.mean() * 100, rf2_precision.std() * 2))
print('Cross-Validation Recall: {:.2f}% (+/- {:.2f}%)'.format(rf2_recall.mean() * 100, rf2_recall.std() * 2))
print('Cross-Validation F1-score: {:.2f}% (+/- {:.2f}%)'.format(rf2_f1score.mean() * 100, rf2_f1score.std() * 2))

# Print time
rf2_seconds = time.time() - start_time
minutes = rf2_seconds / 60
print('Time to run: {:.2f} seconds'.format(rf2_seconds), '({:.2f} minutes)'.format(minutes))
