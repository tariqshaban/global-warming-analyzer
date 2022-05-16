import pandas as pd


class LandTemperatureProvider:
    __land_temperature_by_country_df = None
    __land_temperature_global_df = None

    @staticmethod
    def get_land_temperature_by_country_df():
        if LandTemperatureProvider.__land_temperature_by_country_df is not None:
            return LandTemperatureProvider.__land_temperature_by_country_df.copy()

        df = pd.read_csv('assets/GlobalLandTemperaturesByCountry.csv')

        df = df[
            ~df['Country'].isin(
                ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',
                 'United Kingdom', 'Africa', 'South America'])]

        df = df.replace(
            ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],
            ['Denmark', 'France', 'Netherlands', 'United Kingdom'])

        df['dt'] = \
            pd.to_datetime(df['dt'], errors='coerce')

        LandTemperatureProvider.__land_temperature_by_country_df = df

        return LandTemperatureProvider.__land_temperature_by_country_df.copy()

    @staticmethod
    def get_land_temperature_global_df():
        if LandTemperatureProvider.__land_temperature_global_df is not None:
            return LandTemperatureProvider.__land_temperature_global_df.copy()

        df = pd.read_csv('assets/GlobalTemperatures.csv')

        df['dt'] = \
            pd.to_datetime(df['dt'], errors='coerce')

        LandTemperatureProvider.__land_temperature_global_df = df

        return LandTemperatureProvider.__land_temperature_global_df.copy()
