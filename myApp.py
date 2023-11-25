import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import datetimPIPe
import plotly.express as px


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")


PM1_pred = pd.read_csv("Forecasting/Data/PM1_predictions.csv")
PM2_pred = pd.read_csv("Forecasting/Data/PM2.5_predictions.csv")
PM10_pred = pd.read_csv("Forecasting/Data/PM10_predictions.csv")
AQI_pred = pd.read_csv("Forecasting/Data/AQI_predictions.csv")
bishkek_data = pd.read_csv("Datasets/Bishkek_data.csv")
data_Q = pd.read_csv("Datasets/pm2_data.csv")
data_Q.dropna(inplace=True)
pollutants_D = pd.read_csv("Datasets/grid-export.csv")


# -------------------------------Title of the Page -----------------------------------
st.title('Predicting and Forecasting Air Pollution in Bishkek')


# --------------------------------- Side Bar --------------------------------------
st.sidebar.header("Choose Page")
pages = st.sidebar.selectbox('Which page you want visit?',
                             ('Analyze','Predictions', 'Forecasting'))

if pages=='Analyze':
        st.subheader("Air Quality in Bishkek")
        opt1 = st.selectbox("Visualize graph Based on", ('min', 'max', 'median', 'variance'))
        graph_1 = px.line(bishkek_data, x="Date", y=opt1, color='Specie')
        st.plotly_chart(graph_1)

        st.subheader("Quality of Air in Bishkek from 2019 to 2022")
        graph_2 = px.pie(data_Q, names='AQI Category')
        st.plotly_chart(graph_2)
        
        st.subheader('Concentration of pollutants in Bishkek Air')
        opt2 = st.selectbox("Which pollutant you want to visualize?", ('PM1(mcg/m³)', 'PM10(mcg/m³)', 'PM2.5(mcg/m³)', 'NO(mcg/m³)','NO2(mcg/m³)','SO2(mcg/m³)','Temperature(°C)'))
        graph_3 = px.line(pollutants_D, x="Day", y=opt2)
        st.plotly_chart(graph_3)


if pages == 'Predictions':
    models = st.sidebar.selectbox('Which models you want to use?',
                        ('Classification', 'Regression'))
#  -------------------------------------- Classification models --------------------------------------
    if models =='Classification':
        option = st.selectbox(
        'You can use any of the given modules to predict the quality of air in Bishkek',
        ('LightGBM', 'KNN', 'XGboost'))
        def user_input_features():
            year = st.sidebar.slider('Year', 2022, 2026, 2023)
            month = st.sidebar.slider('Month', 1, 12, 4)
            day = st.sidebar.slider('Day', 0, 6, 4)
            hour = st.sidebar.slider('Hour', 0, 23, 7)
            cons = st.sidebar.slider('NowCast', 0, 150, 46)
            aqi = st.sidebar.slider('AQI', 0, 600, 50)
            raw = st.sidebar.slider('Raq Conc.', 0, 150, 50)
            data = {'Year': year, 'Month':month, 'Day' : day, 'Hour': hour, 'NowCast Conc.':cons, 'AQI':aqi, 'Raw Conc.':raw}
            features = pd.DataFrame(data, index=[0])
            return features

        df = user_input_features() 
        st.subheader('User Input Values')
        st.write(df)

        # --------------------------classication models --------------------------
        if option =='KNN':
            pickled_model = pickle.load(open('Classification/KNN', 'rb'))
            predictions = pickled_model.predict(df)
            if str(predictions[0])=='0':
                air = 'The quality of predicted air is : Good'
            elif str(predictions[0]) =='1':
                air = 'The quality of predicted air is : Hazardous'
            elif  str(predictions[0]) == '2':
                air = 'The quality of predicted air is : Moderate'
            elif  str(predictions[0]) == '3':
                air = 'The quality of predicted air is : Unhealthy'
            elif  str(predictions[0]) == '4':
                air = 'The quality of predicted air is : Unhealthy for Sensitive Groups'
            elif  str(predictions[0]) == '5':
                air = 'The quality of predicted air is : Ver unhealthy'
    
            evaluation_classification = {'Accuracy': 0.97, 'Precision':0.97, 'Recall' : 0.97, 'F1-score': 0.97}

        # elif option =='Catboost':
        #     pickled_model = pickle.load(open('Classification/CatBoost', 'rb'))
        #     predictions = pickled_model.predict(df)
        #     if str(predictions[0][0])=='0':
        #         air = 'The quality of predicted air is : Good'
        #     elif str(predictions[0][0]) =='1':
        #         air = 'The quality of predicted air is : Hazardous'
        #     elif  str(predictions[0][0]) == '2':
        #         air = 'The quality of predicted air is : Moderate'
        #     elif  str(predictions[0][0]) == '3':
        #         air = 'The quality of predicted air is : Unhealthy'
        #     elif  str(predictions[0][0]) == '4':
        #         air = 'The quality of predicted air is : Unhealthy for Sensitive Groups'
        #     elif  str(predictions[0][0]) == '5':
        #         air = 'The quality of predicted air is : Ver unhealthy'
        #     evaluation_classification = {'Accuracy': 0.995, 'Precision':0.99, 'Recall' : 0.99, 'F1-score': 0.99}

        elif option =='LightGBM':
            pickled_model = pickle.load(open('Classification/LightGBM', 'rb'))
            predictions = pickled_model.predict(df)
            if str(predictions[0])=='0':
                air = 'The quality of predicted air is : Good'
            elif str(predictions[0]) =='1':
                air = 'The quality of predicted air is : Hazardous'
            elif  str(predictions[0]) == '2':
                air = 'The quality of predicted air is : Moderate'
            elif  str(predictions[0]) == '3':
                air = 'The quality of predicted air is : Unhealthy'
            elif  str(predictions[0]) == '4':
                air = 'The quality of predicted air is : Unhealthy for Sensitive Groups'
            elif  str(predictions[0]) == '5':
                air = 'The quality of predicted air is : Ver unhealthy'

            evaluation_classification = {'Accuracy': 0.997, 'Precision':0.99, 'Recall' : 0.99, 'F1-score': 0.99}


        elif option =='XGboost':
            pickled_model = pickle.load(open('Classification/XGBoost', 'rb'))
            predictions = pickled_model.predict(df)
            if str(predictions[0])=='0':
                air = 'The quality of predicted air is : Good'
            elif str(predictions[0]) =='1':
                air = 'The quality of predicted air is : Hazardous'
            elif  str(predictions[0]) == '2':
                air = 'The quality of predicted air is : Moderate'
            elif  str(predictions[0]) == '3':
                air = 'The quality of predicted air is : Unhealthy'
            elif  str(predictions[0]) == '4':
                air = 'The quality of predicted air is : Unhealthy for Sensitive Groups'
            elif  str(predictions[0]) == '5':
                air = 'The quality of predicted air is : Ver unhealthy'

            evaluation_classification = {'Accuracy': 0.98, 'Precision':0.98, 'Recall' : 0.98, 'F1-score': 0.98}

        st.subheader("Predictions")
        st.write(air)
        st.subheader("Performance of model on testing dataset")
        st.write(pd.DataFrame(evaluation_classification, index=[0]))

# ------------------------------------------------------------- Regression models --------------------------------------
    if models =='Regression':
        reg_opt = st.selectbox(
        'You can use any of the given modules to predict the quantity of pollutants of air in Bishkek',
        ('LightGBM', 'Extra Trees', 'Random Forest'))

        def user_input_features_reg():
                aqi = st.sidebar.slider('AQI', 0, 400, 80)
                NO = st.sidebar.slider('NO', 0, 150, 50)
                NO2 = st.sidebar.slider('NO2', 0, 50, 25)
                CH20 = st.sidebar.slider('CH20', 0, 30, 3)
                SO2 = st.sidebar.slider('SO2', 0, 100, 5)
                temp = st.sidebar.slider('Temperature', -20, 50, 14)
                hum = st.sidebar.slider('Humidity', 0, 100, 50)
                data_re = {'AQI US': aqi, 'NO(mcg/m³)':NO, 'NO2(mcg/m³)' : NO2, 'CH2O(mcg/m³)': CH20, 'SO2(mcg/m³)':SO2, 'Temperature(°C)':temp, 'Humidity(%)':hum}
                features_re = pd.DataFrame(data_re, index=[0])
                return features_re
    
        df = user_input_features_reg() 
        st.subheader('User Input Values')
        st.write(df)

        # if reg_opt=='Catboost':
        #     pm1_model = pickle.load(open('Regression/PM1/CatBoost_reg', 'rb'))
        #     pm2_model = pickle.load(open('Regression/PM2/CatBoost', 'rb'))
        #     pm10_model = pickle.load(open('Regression/PM10/CatBoost_reg', 'rb'))

        #     pm1_pred = pm1_model.predict(df)
        #     pm2_pred = pm2_model.predict(df)
        #     pm10_pred = pm10_model.predict(df)
        #     predictions = {'PM1': pm1_pred[0], "PM2.5": pm2_pred[0], "PM10":pm10_pred[0]}
        #     reg_pre = pd.DataFrame(predictions, index=[0])
        #     st.subheader("Predictions")
        #     st.write(reg_pre)

        #     matrix = {'#':['R1-Score', 'MAE', 'MSE'], 'PM1': [0.85, 0.97, 2.67], 'PM2': [0.83, 23.26, 2020.34], 'PM10': [0.95, 3.03, 22.61]}
        #     matrices = pd.DataFrame(matrix).set_index('#')
        #     st.subheader('Evaluation Matrices')
        #     st.write(matrices)


        if reg_opt=='LightGBM':
            pm1_model = pickle.load(open('Regression/PM1/Lightgbm', 'rb'))
            pm2_model = pickle.load(open('Regression/PM2/LightGBM', 'rb'))
            pm10_model = pickle.load(open('Regression/PM10/Lightgbm', 'rb'))

            pm1_pred = pm1_model.predict(df)
            pm2_pred = pm2_model.predict(df)
            pm10_pred = pm10_model.predict(df)
            predictions = {'PM1': pm1_pred[0], "PM2.5": pm2_pred[0], "PM10":pm10_pred[0]}
            reg_pre = pd.DataFrame(predictions, index=[0])
            st.subheader("Predictions")
            st.write(reg_pre)
            
            matrix = {'#':['R1-Score', 'MAE', 'MSE'], 'PM1': [0.81, 1.02, 3.32], 'PM2': [0.75, 31.71, 3041.1], 'PM10': [0.93, 3.67, 32.61]}
            matrices = pd.DataFrame(matrix).set_index('#')
            st.subheader('Evaluation Matrices')
            st.write(matrices)

        elif reg_opt=='Extra Trees':
            pm1_model = pickle.load(open('Regression/PM1/ExtraTrees', 'rb'))
            pm2_model = pickle.load(open('Regression/PM2/ExtraTree', 'rb'))
            pm10_model = pickle.load(open('Regression/PM10/ExtraTrees', 'rb'))

            pm1_pred = pm1_model.predict(df)
            pm2_pred = pm2_model.predict(df)
            pm10_pred = pm10_model.predict(df)
            predictions = {'PM1': pm1_pred[0], "PM2.5": pm2_pred[0], "PM10":pm10_pred[0]}
            reg_pre = pd.DataFrame(predictions, index=[0])
            st.subheader("Predictions")
            st.write(reg_pre)
            
            matrix = {'#':['R1-Score', 'MAE', 'MSE'], 'PM1': [0.85, 0.96, 2.69], 'PM2': [0.94, 20.79, 1761.79], 'PM10': [0.94, 3.18, 26.32]}
            matrices = pd.DataFrame(matrix).set_index('#')
            st.subheader('Evaluation Matrices')
            st.write(matrices)

        elif reg_opt=='Random Forest':
            pm1_model = pickle.load(open('Regression/PM1/RandomForest', 'rb'))
            pm2_model = pickle.load(open('Regression/PM2/RandomForest', 'rb'))
            pm10_model = pickle.load(open('Regression/PM10/RandomForest', 'rb'))

            pm1_pred = pm1_model.predict(df)
            pm2_pred = pm2_model.predict(df)
            pm10_pred = pm10_model.predict(df)
            predictions = {'PM1': pm1_pred[0], "PM2.5": pm2_pred[0], "PM10":pm10_pred[0]}
            reg_pre = pd.DataFrame(predictions, index=[0])
            st.subheader("Predictions")
            st.write(reg_pre)
            
            matrix = {'#':['R1-Score', 'MAE', 'MSE'], 'PM1': [0.83, 0.98, 3.02], 'PM2': [0.83, 22.35, 2008.79], 'PM10': [0.94, 3.29, 26.32]}
            matrices = pd.DataFrame(matrix).set_index('#')
            st.subheader('Evaluation Matrices')
            st.write(matrices)


# ------------------------------------------Forecasting models -----------------------------
if pages == 'Forecasting':
    models_forecast = st.sidebar.selectbox('Select any option for forecasting',
                        ('PM1', 'PM2.5', 'PM10', 'AQI'))
    num = st.sidebar.slider('How many days you want to forecast?', 0, 30, 7)
    new_num = 437 + num

    if models_forecast =="PM1":
        pm1_model_for = pickle.load(open('Forecasting/PM1', 'rb'))
        y_range = np.linspace(4, 6, num)
        re = pm1_model_for.predict(start=438,end=new_num,dynamic=True)
        Df = pd.DataFrame({'Predictions':re, 'X':y_range})

        fig = px.line(Df, x='X', y='Predictions')
        st.subheader("Forecasting PM1")
        st.plotly_chart(fig)
        st.subheader("Comparision between actual and predicted values on testing data")
        fig2 = px.line(PM1_pred)
        st.plotly_chart(fig2)

    if models_forecast =="PM2.5":
        pm1_model_for = pickle.load(open('Forecasting/PM2.5', 'rb'))
        y_range = np.linspace(4, 6, num)
        re = pm1_model_for.predict(start=438,end=new_num,dynamic=True)
        Df = pd.DataFrame({'Predictions':re, 'X':y_range})

        fig = px.line(Df, x='X', y='Predictions')
        st.subheader("Forecasting PM2.5")
        st.plotly_chart(fig)
        st.subheader("Comparision between actual and predicted values on testing data")
        fig2 = px.line(PM2_pred)
        st.plotly_chart(fig2)

    if models_forecast =="PM10":
        pm1_model_for = pickle.load(open('Forecasting/PM10', 'rb'))
        y_range = np.linspace(4, 6, num)
        re = pm1_model_for.predict(start=438,end=new_num,dynamic=True)
        Df = pd.DataFrame({'Predictions':re, 'X':y_range})

        fig = px.line(Df, x='X', y='Predictions')
        st.subheader("Forecasting PM10")
        st.plotly_chart(fig)
        st.subheader("Comparision between actual and predicted values on testing data")
        fig2 = px.line(PM10_pred)
        st.plotly_chart(fig2)

    if models_forecast =="AQI":
        pm1_model_for = pickle.load(open('Forecasting/AQI', 'rb'))
        y_range = np.linspace(4, 6, num)
        re = pm1_model_for.predict(start=438,end=new_num,dynamic=True)
        Df = pd.DataFrame({'Predictions':re, 'X':y_range})

        fig = px.line(Df, x='X', y='Predictions')
        st.subheader("Forecasting AQI")
        st.plotly_chart(fig)
        st.subheader("Comparision between actual and predicted values on testing data")
        fig2 = px.line(AQI_pred)
        st.plotly_chart(fig2)





            



