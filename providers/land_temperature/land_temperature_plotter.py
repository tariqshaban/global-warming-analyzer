import numexpr
import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from prophet import Prophet

from providers.hypothesis_container import HypothesisContainer
from providers.land_temperature.land_temperature_provider import LandTemperatureProvider


class LandTemperaturePlotter:

    @staticmethod
    def plot_global_temperature(year=2010):
        df = LandTemperatureProvider.get_land_temperature_by_country_df()

        countries = np.unique(df['Country'])
        year_temp = []
        for country in countries:
            year_temp.append(
                df[
                    (df['Country'] == country) &
                    (df['dt'].dt.year == year)]
                ['AverageTemperature'].mean())

        data = [dict(
            type='choropleth',
            locations=countries,
            z=year_temp,
            locationmode='country names',
            text=countries,
            marker=dict(
                line=dict(color='rgb(0,0,0)', width=1)),
            colorbar=dict(autotick=True, tickprefix='',
                          title='# Average\nTemperature,\n°C')
        )
        ]

        layout = dict(
            title=f'Average Land Temperature in Countries During {year}',
            geo=dict(
                showframe=False,
                showocean=True,
                oceancolor='rgb(0,255,255)',
                projection=dict(
                    type='orthographic',
                    rotation=dict(
                        lon=60,
                        lat=10),
                ),
                lonaxis=dict(
                    showgrid=True,
                    gridcolor='rgb(102, 102, 102)'
                ),
                lataxis=dict(
                    showgrid=True,
                    gridcolor='rgb(102, 102, 102)'
                )
            ),
        )

        fig = dict(data=data, layout=layout)
        pyo.plot(fig, validate=False, filename=f'worldmap temp {year}.html')

    @staticmethod
    def plot_average_global_temperature():
        df = LandTemperatureProvider.get_land_temperature_global_df()

        years = np.unique(df['dt'].dt.year)
        mean_temp_world = []
        mean_temp_world_uncertainty = []

        for year in years:
            mean_temp_world.append(df[df['dt'].dt.year == year]['LandAverageTemperature'].mean())
            mean_temp_world_uncertainty.append(df[df['dt'].dt.year == year]['LandAverageTemperatureUncertainty'].mean())

        trace0 = go.Scatter(
            x=years,
            y=np.array(mean_temp_world) + np.array(mean_temp_world_uncertainty),
            fill=None,
            mode='lines',
            name='Uncertainty top',
            line=dict(
                color='rgb(0, 255, 255)',
            )
        )
        trace1 = go.Scatter(
            x=years,
            y=np.array(mean_temp_world) - np.array(mean_temp_world_uncertainty),
            fill='tonexty',
            mode='lines',
            name='Uncertainty bot',
            line=dict(
                color='rgb(0, 255, 255)',
            )
        )

        trace2 = go.Scatter(
            x=years,
            y=mean_temp_world,
            name='Average Temperature',
            line=dict(
                color='rgb(199, 121, 093)',
            )
        )
        data = [trace0, trace1, trace2]

        layout = go.Layout(
            xaxis=dict(title='Year'),
            yaxis=dict(title='Average Temperature, in °C'),
            title='Average Land Temperature in The World',
            showlegend=False)

        fig = go.Figure(data=data, layout=layout)
        pyo.plot(fig, validate=False, filename=f'world average temp.html')

    @staticmethod
    def plot_global_temperature_change_prediction():
        df = LandTemperatureProvider.get_land_temperature_global_df()

        df = df.filter(['dt', 'LandAverageTemperature'])

        df.dropna(how='any', inplace=True)

        df = df.groupby('dt')['LandAverageTemperature'].mean().reset_index()

        ts = df.rename(columns={'dt': 'ds', 'LandAverageTemperature': 'y'})

        numexpr.set_num_threads(numexpr.detect_number_of_cores())
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(ts)

        future = model.make_future_dataframe(periods=264, freq='MS')
        forecast = model.predict(future)

        model.plot(forecast)

        plt.title('Global Temperature With 22 Years Prediction')
        plt.xlabel('Year')
        plt.ylabel('Temperature')

        plt.show()

    @staticmethod
    def significant_difference_in_temperature_between_years(date1=1900, date2=2000):
        df = LandTemperatureProvider.get_land_temperature_by_country_df()

        df_date1 = df[(df['dt'].dt.year == date1) & (df['AverageTemperature'].notna())][
            'AverageTemperature'].tolist()
        df_date2 = df[(df['dt'].dt.year == date2) & (df['AverageTemperature'].notna())][
            'AverageTemperature'].tolist()

        temperature_date1 = df_date1[:min(len(df_date1), len(df_date2))]
        temperature_date2 = df_date2[:min(len(df_date1), len(df_date2))]

        df1 = pd.DataFrame({
            f'Land Temperature in {date1}': temperature_date1,
            f'Land Temperature in {date2}': temperature_date2
        })

        results = HypothesisContainer.visualize_test(df1,
                                                     f'Land Temperature in {date1}',
                                                     f'Land Temperature in {date2}',
                                                     title=f'Temperature Difference between {date1} and {date2}'
                                                     )

        print(results)

        plt.show()
