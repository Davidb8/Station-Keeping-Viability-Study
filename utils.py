# downsamples data set to every 1000 values
def downsample(values, sample=1000):
    return [value for i, value in enumerate(values) if i % sample == 0]


from math import sin, cos, sqrt, atan2, radians

def coordinate_difference(coord1:list, coord2:list):
    '''takes in two coordinates and returns the meter difference between them
    similar to geopy.distance.geodesic but runs infinitely faster, but less accurate
    In the range of 1,200,000 meter differences, there is an error of 1,700 meters, so 
    the error is insignificant, being that it is only used to do comparative distances'''

    # approximate radius of earth in km
    R = 6373.0

    # get latitude and longitude values from each coordinate
    lat1 = radians(coord1[0])
    lon1 = radians(coord1[1])
    lat2 = radians(coord2[0])
    lon2 = radians(coord2[1])

    # get delta of coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # determine distance
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    # convert km to meters
    distance *= 1000

    # return distance
    return distance


import numpy as np

def tangent_distance(current_location:np.ndarray,target_location:np.ndarray = np.array([0,0])) -> float:
    '''returns the straight line distance between two points'''

    # Calculate x difference
    x_diff = abs(current_location[0] - target_location[0])

    # Calculate y difference
    y_diff = abs(current_location[1] - target_location[1])

    # Determine distance (scale downward to deal with integer overflow)
    distance = 1000 * np.sqrt( (x_diff / 1000)**2 + (y_diff / 1000)**2 )

    # Return difference
    return distance