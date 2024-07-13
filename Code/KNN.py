# Record the start time for measuring training duration
start_time = time.time()

# train model
knn = KNeighborsClassifier(n_neighbors=5)
# fit model
knn.fit(x_train, y_train)

# Check if the K-Nearest Neighbors model is trained before making predictions
if knn:
  try:
    print('K-Nearest Neighbors model trained!')
    # make predictions
    y_pred = knn.predict(x_test)
    # print metrics
    knn_accuracy, knn_precision, knn_recall, knn_f1score = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), fbeta_score(y_test, y_pred, beta=2)
    print('Accuracy: {:.2f}%'.format(knn_accuracy * 100), '\nPrecision: {:.2f}%'.format(knn_precision * 100),
          '\nRecall: {:.2f}%'.format(knn_recall * 100), '\nF1-score: {:.2f}%'.format(knn_f1score * 100))
  except Exception as e:
    print('Error occurred during prediction:', e)
else:
  print('KNN model was not successfully trained.')

# print time
knn_seconds = time.time() - start_time
minutes = knn_seconds / 60
print('Time to run: {:.2f} seconds'.format(knn_seconds), '({:.2f} minutes)'.format(minutes))
