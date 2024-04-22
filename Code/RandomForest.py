start_time = time.time()
# |------------------------------------------------------ Random Forest Classification ------------------------------------------------------|
# train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
# fit model
rf.fit(x_train, y_train)
# |------------------------------------------------------ CALCULATE STATISTICS ------------------------------------------------------|
if rf:
  try:
    print('Random Forest model trained!')
    # make predictions
    y_pred = rf.predict(x_test)
    # print metrics
    rf_accuracy, rf_precision, rf_recall, rf_f1score = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=2)
    print('Accuracy: {:.2f}%'.format(rf_accuracy * 100), '\nPrecision: {:.2f}%'.format(rf_precision * 100),
          '\nRecall: {:.2f}%'.format(rf_recall * 100), '\nF1-score: {:.2f}%'.format(rf_f1score * 100))
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Random Forest model was not successfully trained.')

# print time
rf_seconds = time.time() - start_time
minutes = rf_seconds / 60
print('Time to run: {:.2f} seconds'.format(rf_seconds), '({:.2f} minutes)'.format(minutes))
