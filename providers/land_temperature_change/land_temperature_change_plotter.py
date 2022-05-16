import calendar
from datetime import datetime

import numexpr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet
from sklearn.linear_model import LinearRegression

import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go

from providers.hypothesis_container import HypothesisContainer
from providers.land_temperature_change.land_temperature_change_provider import LandTemperatureChangeProvider


class LandTemperatureChangePlotter:

    @staticmethod
    def plot_global_temperature_change(country='World'):
        df = LandTemperatureChangeProvider.get_land_temperature_change_df()

        if country == 'World':
            df0 = df[df['Country Name'] == 'World']
        else:
            df0 = df[df['Country Name'] == country]

        df0.set_index("Months", inplace=True)
        df0 = df0.loc[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                       'November', 'December']]
        df0.reset_index(inplace=True)

        fig = px.line_polar(df0, r=df0.tem_change, theta=df0.Months, animation_frame='year', line_close=True)

        fig.update_layout(
            title=f'{country} Temperature Fluctuation Over The Years By Month',
            polar=dict(
                angularaxis=dict(
                    visible=True,
                    showline=True,
                    gridcolor="grey",
                ),
                radialaxis=dict(
                    visible=True,
                    range=[-0.5, 3],
                    showline=True,
                    gridcolor="grey",
                ),
            ),
            template='seaborn',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))

        pyo.plot(fig, validate=False, filename='temp change month.html')

    @staticmethod
    def plot_global_temperature_change_scatter():
        df = LandTemperatureChangeProvider.get_land_temperature_change_df()

        df0 = df[df['Months'] == 'Meteorological year']
        df1 = df0[df0['Country Name'] == 'World'].copy()

        year_numeric = [float(i) for i in df1.year]

        reg = LinearRegression().fit(np.vstack(year_numeric), df1.tem_change)
        df1['fit'] = reg.predict(np.vstack(year_numeric))

        fig = go.Figure()

        fig.add_trace(go.Scatter(name='Average World Temperature', x=df1.year, y=df1.tem_change))

        fig.add_trace(go.Scatter(name='Correlation', x=df1.year, y=df1['fit'], mode='lines'))

        fig.update_layout(
            title='Global Temperature Fluctuation Over The Years',
            template='seaborn',
            paper_bgcolor="rgb(234, 234, 242)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))

        fig.update_xaxes(type='category', title='Years')
        fig.update_yaxes(title='Global Temperature Change')

        pyo.plot(fig, validate=False, filename='global temp change.html')

    @staticmethod
    def plot_global_temperature_change_prediction():
        df = LandTemperatureChangeProvider.get_land_temperature_change_df()

        df = df[~df['Months'].isin(['Summer', 'Fall', 'Winter', 'Spring', 'Meteorological year'])]

        print(df.columns)

        df['date'] = df.apply(lambda x: datetime(year=int(x['year']),
                                                 month=list(calendar.month_name).index(x['Months']),
                                                 day=1),
                              axis=1)

        df = df.groupby('date')['tem_change'].mean().reset_index()

        df.dropna(how='any', inplace=True)

        ts = df.rename(columns={'date': 'ds', 'tem_change': 'y'})

        numexpr.set_num_threads(numexpr.detect_number_of_cores())
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(ts)

        future = model.make_future_dataframe(periods=12, freq='MS')
        forecast = model.predict(future)

        model.plot(forecast)

        plt.title('Global Temperature Change With 4 Years Prediction')
        plt.xlabel('Year')
        plt.ylabel('Temperature')

        plt.show()

    @staticmethod
    def significant_difference_in_temperature_between_countries(country1='Germany', country2='Japan', start_year=2010):
        df = LandTemperatureChangeProvider.get_land_temperature_change_df()

        df.set_index("year", inplace=True)
        df = df.loc[map(str, np.arange(start_year, 2020))]

        df_country1 = df[(df['Country Name'] == country1) & (df['tem_change'].notna())]['tem_change'].tolist()
        df_country2 = df[(df['Country Name'] == country2) & (df['tem_change'].notna())]['tem_change'].tolist()

        temperature_country1 = df_country1[:min(len(df_country1), len(df_country2))]
        temperature_end_date = df_country2[:min(len(df_country1), len(df_country2))]

        df1 = pd.DataFrame({
            f'Temperature Change in {country1}': temperature_country1,
            f'Temperature Change in {country2}': temperature_end_date
        })

        results = HypothesisContainer.visualize_test(df1,
                                                     f'Temperature Change in {country1}',
                                                     f'Temperature Change in {country2}',
                                                     title=f'Temperature Change Difference between '
                                                           f'{country1} and {country2} (Effect Rate)'
                                                     )

        print(results)

        plt.show()
