import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,classification_report
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv(r"C:\Users\ANANYA ROHATGI\OneDrive\Desktop\phishing url\PhiUSIIL_Phishing_URL_Dataset.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
df.dropna(inplace=True)
print(df['label'].value_counts())
print(df.describe())
numerical_df=df.select_dtypes(include=['float64','int64'])
corr_matrix=numerical_df.corr()
status_corr=corr_matrix['label']
print(status_corr.shape)

def feature_selector_correlation(cmatrix, threshold):

    selected_features = []
    feature_score = []
    i=0
    for score in cmatrix:
        if abs(score)>threshold:
            selected_features.append(cmatrix.index[i])
            feature_score.append( ['{:3f}'.format(score)])
        i+=1
    result = list(zip(selected_features,feature_score))
    return result

features_selected = feature_selector_correlation(status_corr, 0.2)
print(features_selected)

selected_features = []
for feature, score in features_selected:
    if feature != 'status':
        selected_features.append(feature)
        
print(selected_features)
        
X = df[selected_features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler= StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1]
    },
    'Random Forest': {
        'n_estimators': [100],
        'max_depth': [None, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100],
        'learning_rate': [0.1]
    },
    'SVM': {
        'C': [1],
        'kernel': ['linear']
    },
    'KNN': {
        'n_neighbors': [5],
        'p': [2]
    }
}
print(X_train.columns)
results = {}
for name, clf in classifiers.items():
    # Use RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=clf, 
        param_distributions=param_grids[name],  # Use param_distributions
        n_iter=5,  # Reduced the number of random combinations
        cv=2,       # Reduced cross-validation folds
        n_jobs=-1,  # Use all available cores
        scoring='accuracy',
        random_state=42
    )
    random_search.fit(X_train_scaled, y_train)
    results[name] = random_search
    print(f"{name} best parameters: {random_search.best_params_} with score: {random_search.best_score_}")
    
    
X_test_scaled = scaler.transform(X_test)
for name, random_search in results.items():
    print(f"{name}:")
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:",  random_search.best_score_)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print()
    
print("Summary of Best Models:")
for name, random_search in results.items():
  print(f"{name}")
  print("best Parameters:", random_search.best_params_)
  print("Best Score:", random_search.best_score_)
  print()

model=GradientBoostingClassifier(learning_rate=0.01, n_estimators=100)
model.fit(X_train, y_train)
with open('phishing_model.pkl','wb') as model_file:
  pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
  pickle.dump(scaler, scaler_file)
  
y_predict= model.predict(X_test)
accuracy= accuracy_score(y_test, y_predict)
print(accuracy)

gb_Accuracy_Score = accuracy_score(y_test,y_predict)
gb_JaccardIndex = jaccard_score(y_test,y_predict)
gb_F1_Score = f1_score(y_test,y_predict)
gb_Log_Loss = log_loss(y_test,y_predict)

print(f"Accuracy: {gb_Accuracy_Score}")
print(f"Jaccard Index: {gb_JaccardIndex}")
print(f"F1 Score: {gb_F1_Score}")
print(f"Log Loss: {gb_Log_Loss}")

gb_conf_matrix= confusion_matrix(y_test, y_predict)
print(gb_conf_matrix)

sns.heatmap(gb_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

gb_report= classification_report(y_test,y_predict)
print(gb_report)