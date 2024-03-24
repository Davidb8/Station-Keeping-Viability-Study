'''Script that makes the predictions of the Balloon's path'''

# imports
from BalloonPhysics import *
from forecast import *
import SystemConfiguration


def generate_next_step_viability(
        system_state: SystemConfiguration.SimpleSystemState):
    '''
    Determines next position of the balloon based on current balloon and atmospheric conditions

    Function for viability study (so very over simplified)

    '''

    # get forecast data at specific time - need U and V velocities
    forecast_values = get_forecast_data(system_state)

    new_z_velocity = 0
    target_velocity = 0.5

    # if at float
    if system_state.status == 'Float':
        new_z_velocity = 0

    # if done with transition
    elif abs(system_state.altitude - system_state.target_altitude
             ) < target_velocity * system_state.time_step * 1.5:
        system_state.status = 'Float'
        system_state.altitude = system_state.target_altitude
        new_z_velocity = 0

    # if needing to ascend
    elif system_state.status == 'Ascent':
        new_z_velocity = target_velocity

    # if needing to descend
    elif system_state.status == 'Descent':
        new_z_velocity = -1 * target_velocity

    new_velocty = SystemConfiguration.VelocityVector(u=forecast_values['UGRD'],
                                                     v=forecast_values['VGRD'],
                                                     z=new_z_velocity)
    system_state.velocity = new_velocty

    # determine next step
    system_state = applyVelocityViability(system_state)

    return system_state
