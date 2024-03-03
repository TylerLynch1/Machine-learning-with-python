start_time = time.time()
# |------------------------------------------------------ Logistic Regression (With Cross-validation) ------------------------------------------------------|
# define the number of folds for cross-validation
num_folds = 5  # you can adjust the number of folds as needed

# define the cross-validation strategy
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# define model
logreg = LogisticRegression()
# train model
logreg.fit(x_train, y_train)

if logreg:
  try:
    # perform cross-validation for the Logistic Regression model
    lr2_accuracy = cross_val_score(logreg, x_train, y_train, cv=kf, scoring='accuracy')
    lr2_precision = cross_val_score(logreg, x_train, y_train, cv=kf, scoring='precision')
    lr2_recall = cross_val_score(logreg, x_train, y_train, cv=kf, scoring='recall')
    lr2_f1score = cross_val_score(logreg, x_train, y_train, cv=kf, scoring='f1')
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Logistic Regression model was not successfully trained.')
# |------------------------------------------------------ CALCULATE STATISTICS ------------------------------------------------------|
# print the cross-validation results
print('Cross-Validation Accuracy: {:.2f}% (+/- {:.2f}%)'.format(lr2_accuracy.mean() * 100, lr2_accuracy.std() * 2))
print('Cross-Validation Precision: {:.2f}% (+/- {:.2f}%)'.format(lr2_precision.mean() * 100, lr2_precision.std() * 2))
print('Cross-Validation Recall: {:.2f}% (+/- {:.2f}%)'.format(lr2_recall.mean() * 100, lr2_recall.std() * 2))
print('Cross-Validation F1-score: {:.2f}% (+/- {:.2f}%)'.format(lr2_f1score.mean() * 100, lr2_f1score.std() * 2))

# print time
lr2_seconds = time.time() - start_time
minutes = lr2_seconds / 60
print('Time to run: {:.2f} seconds'.format(lr2_seconds), '({:.2f} minutes)'.format(minutes))
