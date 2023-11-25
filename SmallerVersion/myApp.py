import streamlit
import pandas
import pickle
import plotly.express as px


bishkek_data = pandas.read_csv("Datasets/Bishkek_data.csv")
data_Q = pandas.read_csv("Datasets/pm2_data.csv")
data_Q.dropna(inplace=True)
pollutants_D = pandas.read_csv("Datasets/grid-export.csv")

streamlit.title('Open EMIT')

streamlit.sidebar.header("Choose Page")
pages = streamlit.sidebar.selectbox(
    'Which page you want visit?',
    (
            'Analyze',
            'Predictions'
    )
)

if pages == 'Analyze':
    streamlit.subheader("Air quality in Central Asian region")
    option = streamlit.selectbox(
        "Visualize graph Based on",
        (
                'median',
                'variance'
        )
    )
    graph_1 = px.line(bishkek_data, x="Date", y=option, color='Specie')
    streamlit.plotly_chart(graph_1)

    streamlit.subheader("Quality of air in Central Asian region from 2019/02 to 2022/11")
    graph_2 = px.pie(data_Q, names='AQI Category')
    streamlit.plotly_chart(graph_2)

    streamlit.subheader('Concentration of pollutants in Central Asian region air')
    options = streamlit.selectbox("What specific environmental pollutant are you interested in visualizing?", (
            'PM1(mcg/m³)',
            'PM10(mcg/m³)',
            'PM2.5(mcg/m³)',
            'NO(mcg/m³)',
            'NO2(mcg/m³)',
            'SO2(mcg/m³)',
            'Temperature(°C)',
            'CH4'
    ))
    if options == 'CH4':
        options = 'CH4 '
    graph_3 = px.line(pollutants_D, x="Day", y=options)
    streamlit.plotly_chart(graph_3)

if pages == 'Predictions':
    models = streamlit.sidebar.selectbox(
        'Which models you want to use?',
        (
                'Classification',
        )
    )

    if models == 'Classification':
        option = streamlit.selectbox(
            'You can use any of the given modules to predict the quality of air in Central Asian',
            (
                    'KNN',
            )
        )

        def user_input_features():
            year = streamlit.sidebar.slider('Year', 2022, 2030, 2023)
            month = streamlit.sidebar.slider('Month', 1, 12, 4)
            day = streamlit.sidebar.slider('Day', 0, 6, 4)
            hour = streamlit.sidebar.slider('Hour', 0, 23, 7)
            cons = streamlit.sidebar.slider('NowCast', 0, 150, 46)
            aqi = streamlit.sidebar.slider('AQI', 0, 600, 50)
            raw = streamlit.sidebar.slider('Raq Conc.', 0, 150, 50)
            data = {
                    'Year': year,
                    'Month': month,
                    'Day': day,
                    'Hour': hour,
                    'NowCast Conc.': cons,
                    'AQI': aqi,
                    'Raw Conc.': raw}
            features = pandas.DataFrame(data, index=[0])
            return features

        df = user_input_features()
        streamlit.subheader('User Input Values')
        streamlit.write(df)

        if option == 'KNN':
            pickled_model = pickle.load(open('Classification/KNN', 'rb'))
            predictions = pickled_model.predict(df)
            statuses = (
                    'Good',
                    'Hazardous',
                    'Moderate',
                    'Unhealthy',
                    'Unhealthy for Sensitive Groups',
                    'Ver unhealthy',
            )
            quality = 'The quality of predicted air is : '
            air = ''
            for index, status in enumerate(statuses):
                if predictions[0] == index:
                    air = f'{quality}{status}'
                    break

            evaluation_classification = {'Accuracy': 0.81, 'Precision': 0.82, 'Recall': 0.91, 'F1-score': 0.89}

        streamlit.subheader("Predictions")
        streamlit.write(air)
        streamlit.subheader("Performance of model on testing dataset")
        streamlit.write(pandas.DataFrame(evaluation_classification, index=[0]))
