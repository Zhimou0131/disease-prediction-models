import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle


df = pd.read_csv('heart.csv')


selected_features = ['age', 'sex', 'cp', 'trestbps', 'thalach', 'exang']
X = df[selected_features]
y = df['target']

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])


knn_scores = cross_val_score(knn_pipeline, X, y, cv=5, scoring='accuracy')


print("KNN - 每个折的准确度:", knn_scores)
print("KNN - 平均准确度:", knn_scores.mean())

knn_pipeline.fit(X, y)

pickle.dump(knn_pipeline, open('heart_disease.pkl', 'wb'))