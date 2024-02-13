import svm_data_loader
import svm_data_clean
import svm_model_trainer
import svm_model_accuracy
import svm_confusion_matrix
from sklearn.model_selection import train_test_split

file_path = 'C:\pycharm\SZCAV.csv'
data = svm_data_loader.load_data(file_path)
data = svm_data_clean.clean_data(data)
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)
classifier = svm_model_trainer.train_svm_classifier(X_train, y_train)
accuracy = svm_model_accuracy.test_model(classifier, X_test, y_test)
print(f'Accuracy: {accuracy}')

# Plot confusion matrix
classes = ['Fault',"No Fault" ]  # Replace with your class names
svm_confusion_matrix.plot_confusion_matrix(y_test, classifier.predict(X_test), classes)
