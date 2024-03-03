start_time = time.time()
# |------------------------------------------------------ Logistic Regression ------------------------------------------------------|
# define model
logreg = LogisticRegression()
# train model
logreg.fit(x_train, y_train)
# |------------------------------------------------------ CALCULATE STATISTICS ------------------------------------------------------|
if logreg:
  try:
    print('Logistic Regression model trained!')
    # make predictions
    y_pred = logreg.predict(x_test)
    # print metrics
    lr_accuracy, lr_precision, lr_recall, lr_f1score = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=2)
    print('LR Accuracy: {:.2f}%'.format(lr_accuracy * 100), '\nLR Precision: {:.2f}%'.format(lr_precision * 100),
          '\nLR Recall: {:.2f}%'.format(lr_recall * 100), '\nLR F1-score: {:.2f}%'.format(lr_f1score * 100))
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('Logistic Regression model was not successfully trained.')

lr_seconds = time.time() - start_time
minutes = lr_seconds / 60
print('Time to run: {:.2f} seconds'.format(lr_seconds), '({:.2f} minutes)'.format(minutes))
