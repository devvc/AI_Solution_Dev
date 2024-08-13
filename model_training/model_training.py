import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# data_path = '/data'
data_path = r"C:/NYP_JiayiCourses/Y3S1/EGT309 - AI SOLUTION DEVELOPMENT PROJECT/App/volume"
#data_path = r"C:/Users/tian yu/Documents/nyp/EGT309/App/App/volume"
saved_models_path = f'{data_path}/saved_models'

class ModelTrain:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(random_state=42),
            'lr': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=42))
            ])
        }

    def get_train_test_datasets(self, data_path):
        X_train = pd.read_csv(f'{data_path}/X_train.csv')
        X_test = pd.read_csv(f'{data_path}/X_test.csv')
        y_train = pd.read_csv(f'{data_path}/y_train.csv', header=None).values.ravel()
        y_test = pd.read_csv(f'{data_path}/y_test.csv', header=None).values.ravel()
        return X_train, X_test, y_train, y_test

    def train_model(self, model_key, X_train, y_train, saved_models_path):
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found in self.models.")

        model = self.models[model_key]
        model.fit(X_train, y_train)

        # Save the trained model
        model_filename = f'{saved_models_path}/{model_key}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

        return model

    def evaluate_model(self, clf, X_test, y_test):
        y_pred = clf.predict(X_test)

        # Print model accuracy
        self.get_model_accuracy(y_test, y_pred)

        # Print model classification report
        self.get_model_classification_report(y_test, y_pred)

        # Print confusion matrix
        self.get_confusion_matrix(y_test, y_pred)

        # If the model is a RandomForest, print feature importance
        if isinstance(clf, RandomForestClassifier):
            self.get_feature_importance(clf, X_test.columns)

    def get_model_accuracy(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy of model: {accuracy:.4f}')

    def get_model_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred)
        print('Classification Report of model:\n', report)

    def get_confusion_matrix(self, y_true, y_pred, plot_heatmap=False):
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion matrix:\n', cm)

        if plot_heatmap:
            import seaborn as sns
            import matplotlib.pyplot as plt

            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()

    def get_feature_importance(self, model, feature_names):
        importance = model.feature_importances_
        feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)
        print('Feature Importance:\n', feature_importance)


if __name__ == "__main__":
    Trainer = ModelTrain()
    X_train, X_test, y_train, y_test = Trainer.get_train_test_datasets(data_path)

    for model_key in Trainer.models:
        print(f'Training and evaluating model: {model_key}...')
        clf = Trainer.train_model(model_key, X_train, y_train, saved_models_path)
        Trainer.evaluate_model(clf, X_test, y_test)
        print(f'Model evaluation for model {model_key} done!\n')
