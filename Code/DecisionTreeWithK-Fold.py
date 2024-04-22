start_time = time.time()
# |------------------------------------------------------ Decision Tree Classification (With Cross-validation) ------------------------------------------------------|
# define the number of folds for cross-validation
num_folds = 5  # you can adjust the number of folds as needed

# define the cross-validation strategy
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# define model
dt = DecisionTreeClassifier(random_state=42)
# train model
dt.fit(x_train, y_train)

if dt:
  try:
    # perform cross-validation for the Decision Tree model
    dt2_accuracy = cross_val_score(dt, x_train, y_train, cv=kf, scoring='accuracy')
    dt2_precision = cross_val_score(dt, x_train, y_train, cv=kf, scoring='precision')
    dt2_recall = cross_val_score(dt, x_train, y_train, cv=kf, scoring='recall')
    dt2_f1score = cross_val_score(dt, x_train, y_train, cv=kf, scoring='f1')
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Decision Tree model was not successfully trained.')
  
# |------------------------------------------------------ CALCULATE STATISTICS ------------------------------------------------------|
# print the cross-validation results
print('Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)'.format(dt2_accuracy.mean() * 100, dt2_accuracy.std() * 2))
print('Cross-Validation Precision: {:.2f}% (+/- {:.2f}%)'.format(dt2_precision.mean() * 100, dt2_precision.std() * 2))
print('Cross-Validation Recall: {:.2f}% (+/- {:.2f}%)'.format(dt2_recall.mean() * 100, dt2_recall.std() * 2))
print('Cross-Validation F1-score: {:.2f}% (+/- {:.2f}%)'.format(dt2_f1score.mean() * 100, dt2_f1score.std() * 2))

# print time
dt2_seconds = time.time() - start_time
minutes = dt2_seconds / 60
print('Time to run: {:.2f} seconds'.format(dt2_seconds), '({:.2f} minutes)'.format(minutes))
