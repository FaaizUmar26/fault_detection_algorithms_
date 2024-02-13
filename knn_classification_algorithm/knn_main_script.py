import knn_data_clean
import knn_data_loader
import knn_model_trainer
import knn_model_accuracy
import knn_confusion_matrix

file_path = 'C:\pycharm\SZCAV.csv'
data = knn_data_clean.load_data(file_path)
X_train, X_test, y_train, y_test = knn_data_loader.split_data(data)
classifier = knn_model_trainer.train_knn_classifier(X_train, y_train)
accuracy = knn_model_accuracy.test_model(classifier, X_test, y_test)
print(f'Accuracy: {accuracy}')

# Plot confusion matrix
classes = ['Fault', 'No Fault']  # Replace with your class names
y_pred = classifier.predict(X_test)
knn_confusion_matrix.plot_confusion_matrix(y_test, y_pred, classes)
