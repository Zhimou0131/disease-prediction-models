import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle
df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
X = df[['HighBP', 'BMI', 'HeartDiseaseorAttack', 'GenHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Income']]
y = df['Diabetes_binary']  
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logReg', LogisticRegression(solver='lbfgs', max_iter=1000)) 
])
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')


print("每个折的准确度:", scores)
print("平均准确度:", scores.mean())

pipeline.fit(X, y)

pickle.dump(pipeline, open('diabetes.pkl', 'wb'))