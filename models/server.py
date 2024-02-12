import pandas as pd
from flask import Flask, request, jsonify
from waitress import serve
import pickle
import logging
logging.basicConfig(level=logging.DEBUG)
# install flask, waitress into your ananconda environment
# use the commands 
#     pip install Flask
#     pip install Waitress

app = Flask(__name__)
app.debug = True
heart_disease = pickle.load(open('heart_disease.pkl', 'rb')) 
diabetes = pickle.load(open('diabetes.pkl', 'rb')) 

@app.route('/heart_disease', methods=['GET', 'POST'])

def callModelOne():
    app.logger.debug('Received request for heart_disease prediction')
    age = request.args.get('age', type=float)
    app.logger.debug(f'Received parameters: age={age}')
    try:
        age = request.form.get('age', type=float)
        sex = request.form.get('sex', type=float)
        cp = request.form.get('cp', type=float)
        trestbps = request.form.get('trestbps', type=float)
        thalach = request.form.get('thalach', type=float)
        exang = request.form.get('exang', type=float)
    

        input_features = pd.DataFrame([[age, sex, cp, trestbps, thalach, exang]],
                                  columns=['age', 'sex', 'cp', 'trestbps', 'thalach', 'exang'])
        prediction_proba = heart_disease.predict_proba(input_features)
       
        prob_of_disease = prediction_proba[0][1]
        prediction = int(prob_of_disease >= 0.5)  
        
        return jsonify({'prediction': prob_of_disease, 'class': prediction})
    except Exception as e:
        app.logger.error('Failed to make prediction', exc_info=e)
        return jsonify({'error': str(e)})
@app.route('/diabetes', methods=['GET','POST'])
def callModelTwo():
    try:
        HighBP = request.form.get('HighBP', type=float)
        BMI = request.form.get('BMI', type=float)
        HeartDiseaseorAttack = request.form.get('HeartDiseaseorAttack', type=float)
        GenHlth = request.form.get('GenHlth', type=float)
        PhysHlth = request.form.get('PhysHlth', type=float)
        DiffWalk = request.form.get('DiffWalk', type=float)
        Age = request.form.get('Age', type=float)
        Income = request.form.get('Income', type=float)
    
 
        input_features = pd.DataFrame([[HighBP, BMI, HeartDiseaseorAttack, GenHlth, PhysHlth, DiffWalk, Age, Income]],
                                  columns=['HighBP', 'BMI', 'HeartDiseaseorAttack', 'GenHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Income'])
    

        prediction_proba = diabetes.predict_proba(input_features)
       
        prob_of_disease = prediction_proba[0][1]
        prediction = int(prob_of_disease >= 0.5)
        return jsonify({'prediction': prob_of_disease, 'class': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

# run the server
if __name__ == '__main__':
    print("Starting the server.....")
    # serve(app, host="0.0.0.0", port=8080)
    app.run(host='192.168.1.98', port=5000)
