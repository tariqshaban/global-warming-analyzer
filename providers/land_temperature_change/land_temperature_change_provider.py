import pandas as pd


class LandTemperatureChangeProvider:
    __land_temperature_change_df = None

    @staticmethod
    def get_land_temperature_change_df():
        if LandTemperatureChangeProvider.__land_temperature_change_df is not None:
            return LandTemperatureChangeProvider.__land_temperature_change_df.copy()

        df = pd.read_csv('assets/Environment_Temperature_change_E_All_Data_NOFLAG.csv', encoding='latin-1')

        df.rename(columns={'Area': 'Country Name'}, inplace=True)
        df.set_index('Months', inplace=True)
        df.rename({'Dec\x96Jan\x96Feb': 'Winter', 'Mar\x96Apr\x96May': 'Spring', 'Jun\x96Jul\x96Aug': 'Summer',
                   'Sep\x96Oct\x96Nov': 'Fall'}, axis='index', inplace=True)
        df.reset_index(inplace=True)

        df = df[df['Element'] == 'Temperature change']

        df.drop(['Area Code', 'Months Code', 'Element Code', 'Element', 'Unit'], axis=1, inplace=True)

        df = df.melt(id_vars=["Country Name", "Months", ], var_name="year", value_name="tem_change")
        df["year"] = [i.split("Y")[-1] for i in df.year]

        LandTemperatureChangeProvider.__land_temperature_change_df = df

        return LandTemperatureChangeProvider.__land_temperature_change_df.copy()
