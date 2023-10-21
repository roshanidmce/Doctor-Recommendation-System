import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])

def predict():
    missing_value = ["N/a", "na", np.nan]
    df = pd.read_csv("data.csv", na_values=missing_value)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df.isnull().sum()
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Experience']])
    df[['Experience']] = scaled
    df['Awards'] = 60*df['Padma_Vibhushan']+50*df['Padma_Bhushan']+40*df['Padma_Shri'] + \
        30*df['Dhanvantari_Award']+20 * \
        df['BC_Roy_National_Award']+10*df['Other_Awards']
    scaled2 = scaler.fit_transform(df[['Awards']])
    df[['Awards']] = scaled2
    df.drop('Padma_Vibhushan',
            axis='columns', inplace=True)
    df.drop('Padma_Bhushan',
            axis='columns', inplace=True)
    df.drop('Padma_Shri',
            axis='columns', inplace=True)
    df.drop('Dhanvantari_Award',
            axis='columns', inplace=True)
    df.drop('BC_Roy_National_Award',
            axis='columns', inplace=True)
    df.drop('Other_Awards',
            axis='columns', inplace=True)
    km = KMeans(n_clusters=4)
    y_predict = km.fit_predict(df[['Awards', 'Experience']])
    df['cluster'] = y_predict
    Doctor_Name = request.form.get("username")

    specialisaton = request.form.get("speciality")

    city = request.form.get("city")

    doctor_Experience = int(request.form.get("experience"))

    doctor_Awards_Points = int(request.values.get("rating"))

    predicted_user = km.predict(
        [[doctor_Experience, doctor_Awards_Points]])
    final = []
    if(predicted_user < 4):  # for outliers
        for i in range((df.shape[0])):
            if(str(df.iloc[i, 2]).count(city) > 0 and str(df.iloc[i, 1]).count(specialisaton) > 0 and df.iloc[i, 6] <= predicted_user and Experience_Normalised < float(df.iloc[i, 4]) and Awards_Point_Normalised < float(df.iloc[i, 5])):
                final.append(df.iloc[i])

    hi=[]
    if(len(final)):
            hi.append(x)
    return render_template('index.html', prediction_text=hi)


if __name__ == '__main__':
    app.run(debug = True)
