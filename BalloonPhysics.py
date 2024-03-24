import math
from datetime import timedelta
import constants
from SystemConfiguration import SimpleSystemState


def distanceTraveledToCoordinates(fromLat: float, fromLong: float,
                                  angle: float, distance: float) -> list:
    '''Generates new lat lon from a distance traveled in a direction'''

    delta = distance / constants.EARTH_RADIUS
    theta = math.radians(fromLat)
    phi = math.radians(fromLong)
    gamma = math.radians(angle)

    # calculate sines and cosines
    c_theta = math.cos(theta)
    s_theta = math.sin(theta)

    c_phi = math.cos(phi)
    s_phi = math.sin(phi)

    c_delta = math.cos(delta)
    s_delta = math.sin(delta)

    c_gamma = math.cos(gamma)
    s_gamma = math.sin(gamma)

    # calculate end vector
    x = c_delta * c_theta * c_phi - s_delta * (s_theta * c_phi * c_gamma +
                                               s_phi * s_gamma)
    y = c_delta * c_theta * s_phi - s_delta * (s_theta * s_phi * c_gamma -
                                               c_phi * s_gamma)
    z = s_delta * c_theta * c_gamma + c_delta * s_theta

    # calculate end lat long
    theta2 = math.asin(z)
    phi2 = math.atan2(y, x)

    latitude = math.degrees(theta2)
    longitude = math.degrees(phi2)

    return [latitude, longitude]


def applyVelocityViability(system_state: SimpleSystemState):
    '''

    Function that takes in the current state of the Balloon --> its position in space/time, as well as its current u,v,z 
    velocities, returns its new state 1 time step later

    '''
    # increment time
    system_state.time += timedelta(seconds=system_state.time_step)
    system_state.flight_duration += system_state.time_step

    # determines position delta
    delta_u_pos = system_state.velocity.u * system_state.time_step
    delta_v_pos = system_state.velocity.v * system_state.time_step
    delta_z_pos = system_state.velocity.z * system_state.time_step

    # calculate trajectory
    angle = math.degrees(
        math.atan2(
            system_state.velocity.u,
            system_state.velocity.v))  # gets direction with respect to East
    try:
        distance_traveled = math.sqrt(
            math.pow(delta_v_pos / 1000, 2) +
            math.pow(delta_u_pos / 1000, 2)) * 1000
    except:
        print(delta_u_pos, delta_v_pos)
        print(system_state.velocity.u, system_state.velocity.v)
        exit(1)

    # determine new coordinate position
    lat, long = distanceTraveledToCoordinates(system_state.latitude,
                                              system_state.longitude, angle,
                                              distance_traveled)

    # increments altitude
    system_state.altitude += delta_z_pos

    # update coordinate position
    system_state.latitude = lat
    system_state.longitude = long

    # return dictionary of current_step conditions
    return system_state
