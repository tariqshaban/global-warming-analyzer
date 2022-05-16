import numexpr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet

from providers.hypothesis_container import HypothesisContainer
from providers.sea_level_change.sea_level_change_provider import SeaLevelChangeProvider


class SeaLevelChangePlotter:

    @staticmethod
    def plot_sea_level_change():
        df = SeaLevelChangeProvider.get_sea_level_change_df()

        ts = df.groupby(["Time"])["GMSL"].sum()
        plt.figure(figsize=(14, 8))
        plt.title('Global Average Absolute Sea Level Change')
        plt.xlabel('Time')
        plt.ylabel('Sea Level Change')
        plt.plot(ts)

        plt.show()

    @staticmethod
    def plot_sea_level_prediction():
        df = SeaLevelChangeProvider.get_sea_level_change_df()

        ts = df.rename(columns={'Time': 'ds', 'GMSL': 'y', 'GMSL uncertainty': 'yhat'})

        numexpr.set_num_threads(numexpr.detect_number_of_cores())
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(ts)

        future = model.make_future_dataframe(periods=48, freq='MS')
        forecast = model.predict(future)

        model.plot(forecast)

        plt.title('Global Average Absolute Sea Level Change With 4 Years Prediction')
        plt.xlabel('Year')
        plt.ylabel('Global Mean Sea Level (GMSL)')

        plt.show()

    @staticmethod
    def significant_difference_in_sea_level_between_decades(decade1=1900, decade2=2000):
        df = SeaLevelChangeProvider.get_sea_level_change_df()

        decade1 = decade1 - (decade1 % 10)
        decade1 = np.arange(decade1, decade1 + 10)

        decade2 = decade2 - (decade2 % 10)
        decade2 = np.arange(decade2, decade2 + 10)

        df_start_date = df[(df['Time'].dt.year.isin(decade1)) & (df['GMSL'].notna())][
            'GMSL'].tolist()
        df_end_date = df[(df['Time'].dt.year.isin(decade2)) & (df['GMSL'].notna())][
            'GMSL'].tolist()

        sea_level_start_date = df_start_date[:min(len(df_start_date), len(df_end_date))]
        sea_level_end_date = df_end_date[:min(len(df_start_date), len(df_end_date))]

        df1 = pd.DataFrame({
            f'Sea Level in {decade1[0]}': sea_level_start_date,
            f'Sea Level in {decade2[0]}': sea_level_end_date
        })

        results = HypothesisContainer.visualize_test(df1,
                                                     f'Sea Level in {decade1[0]}',
                                                     f'Sea Level in {decade2[0]}',
                                                     title=f'Sea Level Difference between {decade1[0]} and {decade2[0]}'
                                                     )

        print(results)

        plt.show()
