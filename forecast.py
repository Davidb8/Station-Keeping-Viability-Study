'''gets forecast data'''

from datetime import datetime, timedelta
import numpy as np

import SystemConfiguration

import os
import pytz

import netCDF4 as nc


from math import radians, cos, sin, asin, sqrt


def round_time(time: datetime):
    # rounds time

    seconds = (
        time -
        time.replace(minute=0, second=0, microsecond=0)).total_seconds()

    # If the number of seconds is less than 30 minutes, round down to the start of the hour
    if seconds < 1800:
        time = time.replace(minute=0, second=0, microsecond=0)
    # Otherwise, round up to the start of the next hour
    else:
        time = time.replace(minute=0, second=0,
                            microsecond=0) + timedelta(hours=1)

    return time


def update_stored_data(system_state: SystemConfiguration.SimpleSystemState):
    ''' Function to ensure system_state.current_forecast has the correct data'''

    pass


data = [(64, 14968.24, 15003.50), (63, 15292.44, 15329.24),
        (62, 15619.30, 15657.70), (61, 15948.85, 15988.88),
        (60, 16281.10, 16322.83), (59, 16616.09, 16659.55),
        (58, 16953.83, 16999.08), (57, 17294.53, 17341.62),
        (56, 17638.78, 17687.77), (55, 17987.41, 18038.35),
        (54, 18341.27, 18394.25), (53, 18701.27, 18756.34),
        (52, 19068.35, 19125.61), (51, 19443.55, 19503.09),
        (50, 19827.95, 19889.88), (49, 20222.24, 20286.66),
        (48, 20627.87, 20694.90), (47, 21045.00, 21114.77),
        (46, 21473.98, 21546.62), (45, 21915.16, 21990.82),
        (44, 22368.91, 22447.75), (43, 22835.68, 22917.85),
        (42, 23316.04, 23401.71), (41, 23810.67, 23900.02),
        (40, 24320.28, 24413.50), (39, 24845.63, 24942.93),
        (38, 25387.55, 25489.15), (37, 25946.90, 26053.04),
        (36, 26524.63, 26635.56), (35, 27121.74, 27237.73),
        (34, 27739.29, 27860.64), (33, 28378.46, 28505.47),
        (32, 29040.48, 29173.50), (31, 29726.69, 29866.09),
        (30, 30438.54, 30584.71), (29, 31177.59, 31330.96),
        (28, 31945.53, 32106.57)]

geometric_altitude_dict = {
    level: geometric_altitude
    for level, _, geometric_altitude in data
}
geopotential_altitude_dict = {
    level: geopotential_altitude
    for level, geopotential_altitude, _ in data
}

ALTITUDE_OPTIONS = []
for level in range(28, 65):
    ALTITUDE_OPTIONS.append(geometric_altitude_dict[level])


def find_closest_level(altitude) -> int:
    closest_level = None
    min_difference = float('inf')

    for level, level_altitude in geometric_altitude_dict.items():
        difference = abs(level_altitude - altitude)
        if difference < min_difference:
            min_difference = difference
            closest_level = level

    return closest_level


def find_lowest_wind_speed_level(system_state: SystemConfiguration.SimpleSystemState):
    '''Returns the altitude (in meters) with the lowest windspeed

    Assumes NETCDF4 Datafile can be found
    '''

    # unpack values
    lat = system_state.latitude
    lon = system_state.longitude
    if lon < 0:
        lon += 360

    time = system_state.time.astimezone(pytz.UTC)

    time = round_time(time)

    # Initialize the lowest wind speed and level
    lowest_wind_speed = np.inf
    lowest_level = None

    data_directory = rf'/data/'

    # Iterate over the levels
    for level in range(28, 65):  # Modify the range as per your level values
        file_name = f'{time.year}_{time.month:02d}_{time.day:02d}_{level}_wind_data.nc'
        file_path = os.path.join(data_directory, file_name)
        # Check if the file exists
        # print(file_path)
        if os.path.isfile(file_path):
            # Open the netCDF file
            wind_data = nc.Dataset(file_path)

            time_var = wind_data.variables['time']

            time_index = nc.date2index(time, time_var)

            # Find the index of the latitude and longitude closest to the requested point
            lat_var = wind_data.variables['latitude']
            lon_var = wind_data.variables['longitude']
            lat_index = (abs(lat_var[:] - lat)).argmin()
            lon_index = (abs(lon_var[:] - lon)).argmin()

            # Read the u and vwind component
            u_wind = wind_data.variables['u'][time_index, lat_index, lon_index]
            v_wind = wind_data.variables['v'][time_index, lat_index, lon_index]

            # Calculate the wind speed
            wind_speed = np.sqrt(u_wind**2 + v_wind**2)
            # Check if the current level has the lowest wind speed
            if np.min(wind_speed) < lowest_wind_speed:
                lowest_wind_speed = np.min(wind_speed)
                lowest_level = level

            # Close the netCDF file
            wind_data.close()
    return geometric_altitude_dict[lowest_level]


def get_forecast_data(system_state: SystemConfiguration.SimpleSystemState):
    ''' Gets forecast data from stored NETCDF4 File'''

    # unpack values
    lat = system_state.latitude
    lon = system_state.longitude
    time = system_state.time.astimezone(pytz.UTC)

    if lon < 0:
        lon += 360

    time = round_time(time)
    wind_file = None
    forecast = {}

    system_state = update_stored_data(system_state)

    wind_file = system_state.current_forecast.data

    time_var = wind_file.variables['time']
    time_index = nc.date2index(time, time_var)

    if not wind_file == None:
        # Find the index of the latitude and longitude closest to the requested point
        lat_var = wind_file.variables['latitude']
        lon_var = wind_file.variables['longitude']
        lat_index = (abs(lat_var[:] - lat)).argmin()
        lon_index = (abs(lon_var[:] - lon)).argmin()

        # Get the data at the requested time, pressure, and location for each variable
        u_wind = wind_file.variables['u'][time_index, lat_index, lon_index]
        v_wind = wind_file.variables['v'][time_index, lat_index, lon_index]

        u_wind = float(u_wind)
        v_wind = float(v_wind)

        forecast = {}
        forecast['UGRD'] = u_wind
        forecast['VGRD'] = v_wind

    return forecast
