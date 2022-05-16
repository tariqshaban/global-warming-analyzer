from providers.land_temperature.land_temperature_plotter import LandTemperaturePlotter
from providers.land_temperature_change.land_temperature_change_plotter import LandTemperatureChangePlotter
from providers.sea_level_change.sea_level_change_plotter import SeaLevelChangePlotter

# Ranging from 1743 to 2013
# ----------------------------------------------------------------------------------------------------------------------
LandTemperaturePlotter.plot_global_temperature(year=1950)
LandTemperaturePlotter.plot_global_temperature(year=2010)
LandTemperaturePlotter.plot_average_global_temperature()
LandTemperaturePlotter.significant_difference_in_temperature_between_years(date1=1743, date2=2013)
LandTemperaturePlotter.plot_global_temperature_change_prediction()

# Ranging from 1961 to 2019 (1971 reference "ground-zero")
# ----------------------------------------------------------------------------------------------------------------------
LandTemperatureChangePlotter.plot_global_temperature_change()
LandTemperatureChangePlotter.plot_global_temperature_change(country='Jordan')
LandTemperatureChangePlotter.plot_global_temperature_change_scatter()
LandTemperatureChangePlotter.significant_difference_in_temperature_between_countries(country1='Jordan',
                                                                                     country2='Saudi Arabia',
                                                                                     start_year=2010)
LandTemperatureChangePlotter.plot_global_temperature_change_prediction()

# Ranging from 1880 to 2013 (1988 reference "ground-zero")
# ----------------------------------------------------------------------------------------------------------------------
SeaLevelChangePlotter.plot_sea_level_change()
SeaLevelChangePlotter.significant_difference_in_sea_level_between_decades(decade1=1980, decade2=2000)
SeaLevelChangePlotter.significant_difference_in_sea_level_between_decades(decade1=1880, decade2=2000)
SeaLevelChangePlotter.plot_sea_level_prediction()
