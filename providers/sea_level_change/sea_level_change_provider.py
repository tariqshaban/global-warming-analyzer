from datetime import datetime

import pandas as pd


class SeaLevelChangeProvider:
    __sea_level_change_df = None

    @staticmethod
    def get_sea_level_change_df():
        if SeaLevelChangeProvider.__sea_level_change_df is not None:
            return SeaLevelChangeProvider.__sea_level_change_df.copy()

        df = pd.read_csv('assets/sea_levels_2015.csv')

        df.Time = df.Time.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

        SeaLevelChangeProvider.__sea_level_change_df = df

        return SeaLevelChangeProvider.__sea_level_change_df.copy()
