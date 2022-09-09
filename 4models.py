import pandas as pd
from sklearn.model_selection import train_test_split
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle


from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle

mp_drawing = mp.solutions.drawing_utils # For Drawing the cordinates
mp_hands = mp.solutions.hands # Solution specific for hand coordinate

df = pd.read_csv('up.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)



pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),

}


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

fit_models['gb'].predict(X_test)




from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle


for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))
fit_models['gb'].predict(X_test)
with open('up gb.pkl2', 'wb') as f:
    pickle.dump(fit_models['lr'], f)

