import SystemConfiguration
import Prediction
import constants
import numpy as np
from datetime import datetime, timedelta
import forecast
# from matplotlib import pyplot as plt
import utils
import multiprocessing
import os
import shutil
from timezonefinder import TimezoneFinder
import pytz
import sys
import pandas as pd
import gc
'''
System for running tree-based algorithm for station keeping

Operates for a Generic Balloon System

Assumptions:
1) The Balloons horizontal speeds are equal to that of the wind at its pressure level
2) The Balloon's altitude is perfectly and intentionally controlled
    a) The Balloon doesn't decay
    b) The Balloon ascends or descents to new pressure levels at 0.5 m/s
'''

MAX_RADIUS = 250_000
COVERAGE_RADIUS = 50_000
MIN_DISTANCE_FROM_TARGET = 40_000
MAX_TRANSITIONS = 50
LEVEL_OPTIONS = [28, 64]
chosen_paths: list[list[SystemConfiguration.TransitionOption]] = []
START_DATE: datetime
END_DATE: datetime

DISTANCE_DELTA = 15000
DEPTH = 2
MAX_DURATION = 60  # days


def StationKeep(config: SystemConfiguration.ViabilityConfig, output_dir: str,
                error_log_file_path: str):
    global END_DATE
    global START_DATE
    global chosen_paths
    chosen_paths = []

    # get system Data classes
    system_state = SystemConfiguration.SimpleSystemState()

    # make output folder
    if not os.path.exists(f'{output_dir}'):
        os.mkdir(f'{output_dir}')

    print(f"Beginning Sim: {config.location}")

    # Get Launch conditions for system

    tz = pytz.timezone('UTC')
    launch_time = tz.localize(config.start_time)

    START_DATE = launch_time
    target_month = START_DATE.month + 2
    target_year = START_DATE.year

    if target_month > 12:
        target_month -= 12
        target_year += 1

    # Find the last day of the target month
    next_month = datetime(target_year, target_month, 1)
    END_DATE = next_month - timedelta(hours=1)
    END_DATE = tz.localize(END_DATE)

    # set initial values
    system_state.time = launch_time  # datetime object
    system_state.latitude = config.start_latitude
    system_state.longitude = config.start_longitude
    system_state.current_forecast.day = 0
    system_state.current_forecast.level = 0
    system_state.PID = 1
    system_state.sim_name = config.location
    system_state.time_step = constants.TIME_STEP
    system_state.start_date = launch_time
    start_altitude = forecast.find_lowest_wind_speed_level(system_state)
    system_state.altitude = start_altitude

    forecastData = forecast.get_forecast_data(system_state)

    system_state.velocity = SystemConfiguration.VelocityVector(
        u=forecastData["UGRD"], v=forecastData['VGRD'], z=0.0001)
    system_state.viability_study = True
    system_state.jumps_remaining = MAX_TRANSITIONS
    system_state.target_location = [
        config.start_latitude, config.start_longitude
    ]

    # update system
    system_state.status = 'Float'
    system_state.target_altitude = system_state.altitude

    system_state.time_step = 2

    ### Loop until outside Radius ###

    prepare_for_transition = False
    # boolean to determine whether to investigate alternative altitudes
    investigateAltitudes = False
    FailedInvestigationTime = system_state.time
    counter = 0

    sim_start_time = datetime.now()
    sim_end_time = sim_start_time + timedelta(hours=7 * 24 + 12)
    current_distance = 0
    previous_distance = float('inf')
    target_transition_altitude = start_altitude
    target_transition_time = END_DATE

    exit_condition = ''

    while True:
        # exit conditions
        if (current_distance > MAX_RADIUS):
            exit_condition = 'MAX RADIUS EXCEEDED'
            break

        if (system_state.time > END_DATE):
            exit_condition = 'MAX DURATION EXCEEDED'
            break

        if (datetime.now() > sim_end_time):
            exit_condition = 'CLUSTER OUT OF TIME'
            break

        # Check if the error file exists
        if os.path.isfile(error_log_file_path):
            # Get the size of the error file
            file_size = os.path.getsize(error_log_file_path)

            # If the file is not empty, exit the program
            if file_size > 0:
                print("Error file is not empty. Exiting the program.",
                      file=sys.stderr)

                # clear temp folder
                dir_path = rf'/data/temp/{system_state.sim_name}/'
                dir_path = rf'/data/temp/{system_state.sim_name}/'
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)

                # rename folder to failed
                current_directory = output_dir
                new_directory = output_dir[:-1] + ' FAILED/'

                # Rename the directory
                os.rename(current_directory, new_directory)

                raise Exception(
                    "LVS 180: Error file is not empty. Exiting the program.")

        # record current distance from target
        current_distance = utils.coordinate_difference(
            system_state.target_location,
            [system_state.latitude, system_state.longitude])

        ## Check if balloon should look for alternative routes ##
        Investigation_Conditions = [
            (current_distance > MIN_DISTANCE_FROM_TARGET),
            (current_distance < MAX_RADIUS), (system_state.jumps_remaining
                                              > 0),
            (current_distance
             > previous_distance), (not prepare_for_transition),
            (system_state.status == 'Float'),
            ((system_state.time - FailedInvestigationTime).total_seconds()
             >= (3600 * counter))
        ]

        if all(Investigation_Conditions):
            investigateAltitudes = True
            print("Investigating Altitudes")

        ## Generate Optimal Options ##
        if investigateAltitudes:

            ### LOG ###
            df = system_state.export_flight_df(downsample=10)
            filename = f"{output_dir}{config.location} Fight Log ({system_state.time.strftime('%m-%d-%H')}).csv"
            df.to_csv(filename)
            del (df)
            system_state.export_flight_kml_large(filename)
            try:
                best_system = ideal_transition_decision(system_state.clone(),
                                                        depth=DEPTH)
            except Exception as e:
                print(
                    f"System Failed to Generate Ideal Transit Decision due to {str(e)}",
                    file=sys.stderr,
                    flush=True)
                raise Exception(
                    f"[LVS 229: System Failed to Generate Ideal Transit Decision due to {str(e)}]"
                )
            best_path = best_system.predictive_path
            if len(best_path) == 0:
                counter += 1
                FailedInvestigationTime = system_state.time
                print("Failed to find more optimal Altitude", flush=True)

            else:
                best_decision = best_path[0]
                target_transition_time: datetime = best_decision.transition_time
                target_transition_altitude: float = best_decision.transition_altitude

                if target_transition_altitude == system_state.altitude:
                    counter += 1
                    FailedInvestigationTime = system_state.time
                    print("Failed to find more optimal Altitude", flush=True)

                else:
                    system_state.altitude_decisions.append(str(best_decision))
                    chosen_paths.append(best_path)
                    prepare_for_transition = True
                    print("Prepping for Transition", flush=True)

            del (best_system)

            investigateAltitudes = False

        ## Implement best plan ##
        if prepare_for_transition:

            # see if time to transition
            if np.abs((system_state.time -
                       target_transition_time).total_seconds()) < 60:

                system_state.target_altitude = target_transition_altitude

                if system_state.altitude < target_transition_altitude:
                    system_state.status = 'Ascent'
                elif system_state.altitude > target_transition_altitude:
                    system_state.status = 'Descent'

                prepare_for_transition = False
                system_state.jumps_remaining -= 1
                counter = 0

        # Step Balloon Forward
        try:
            Prediction.generate_next_step_viability(system_state)
        except Exception as e:
            if 'No such file or directory' in str(e):
                forecast.preload_data_multiprocess(system_state)
                Prediction.generate_next_step_viability(system_state)

            else:
                raise Exception(
                    f"[LVS 285: Failed to Generate Next step on main Sim due to {str(e)}]"
                )

        system_state.update_flight_path()

        # see if within coverage radius --> add to Time on Target
        if current_distance < COVERAGE_RADIUS:
            system_state.time_on_target += system_state.time_step

        print(system_state.log())
        previous_distance = current_distance

    # Output Logging
    filename = f"{output_dir}{config.location} Finished.csv"
    df = system_state.export_flight_df(downsample=10)
    df.to_csv(filename)
    system_state.export_flight_kml_large(filename)
    for path in chosen_paths:
        print([str(decision) for decision in path])
    print(
        f"Total Time on Target: {timedelta(seconds = system_state.time_on_target)} | {system_state.time_on_target}"
    )
    print(
        f"Total Runtime: {timedelta(seconds = (datetime.now() - sim_start_time).total_seconds())}"
    )

    transition_time_list = []
    transition_altitude_list = []
    for path in chosen_paths:
        decision1 = path[0]
        transition_time_list.append(decision1.transition_time)
        transition_altitude_list.append(decision1.transition_altitude)

    data = pd.DataFrame({
        'Altitude': transition_altitude_list,
        'Time': transition_time_list
    })
    # Name of the output CSV file
    filename = f"{output_dir}Altitude Decisions.csv"
    data.to_csv(filename, index=False)

    config_data = pd.DataFrame(
        {
            'Location': [config.location],
            'Latitude': [config.start_latitude],
            'Longitude': [config.start_longitude],
            'Time': [config.start_time]
        },
        index=[0])
    filename = f"{output_dir}config.csv"  # Name of the output CSV file
    config_data.to_csv(filename)

    tot = system_state.time_on_target
    flight_duration = (system_state.time - launch_time).total_seconds()
    tot_percent = round(tot / flight_duration * 100, 2)

    output_data = {
        'Name': system_state.sim_name,
        'Runtime': (datetime.now() - sim_start_time).total_seconds(),
        'Location': config.location,
        'Start Altitude': start_altitude,
        'Start Latitude': config.start_latitude,
        'Start Longitude': config.start_longitude,
        'Start Time': config.start_time,
        'End Time': system_state.time,
        'Flight Duration': flight_duration,
        'Time on Target': system_state.time_on_target,
        'Time on Target Percent': tot_percent,
        'Transitions Made': MAX_TRANSITIONS - system_state.jumps_remaining,
        'End State': exit_condition,
    }
    data = pd.DataFrame([output_data])
    # Name of the output CSV file
    filename = f"{output_dir}Sim Output Data.csv"
    data.to_csv(filename, index=False)

    # Clear Temp Folder
    dir_path = rf'/data/temp/{system_state.sim_name}/'
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    return 0


def ideal_transition_decision(
        system_state: SystemConfiguration.SimpleSystemState,
        depth=3) -> SystemConfiguration.SimpleSystemState:
    '''
    Function that will take in the current location and then allow 3 degrees of transition to occur
    simulating every possible transition time and altitude, take each one and repeat for 3 transitions
    from that determine which option set will cause the Balloon to have the greatest time on target,
    and then choose that option

    This process will require recursive calling of functions, this function will not be recursively called.
    It will be the main manager, and just begin calling the system and receiving the output of all 
    paths and their respective Time on Target. Then eventually returning the best next altitude/time 
    to transition to maximize time on targets

    '''

    # record true conditions
    true_system = system_state.clone()

    # Descale Precision:
    system_state.time_step = 15

    # prep balloon
    true_system.prep_recursion(depth)

    # generate output dictionary
    true_system.current_forecast = SystemConfiguration.ERA5Forecast(0, 0, None)
    true_system.flight_path = []

    PredictiveRoutes = SystemConfiguration.RouteStorage(true_system.clone())

    # make temp folder
    dir_path = rf'data/temp/'
    dir_path = rf'/data/temp/'

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    dir_path = rf'/data/temp/{system_state.sim_name}/'
    dir_path = rf'/data/temp/{system_state.sim_name}/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    recursive_route_explorer(PredictiveRoutes, true_system.clone(), 0)

    target_altitude = true_system.altitude
    system_list = PredictiveRoutes.get_system_list()
    system_list.sort(
        key=lambda system:
        (system.time_on_target, -abs(system.altitude - target_altitude),
         (system_state.time - system_state.start_date).total_seconds()),
        reverse=True)
    if len(system_list) == 0:
        return true_system

    best_system = system_list[0].clone()

    # yay it finished --> what did we get?
    print()
    print(f"Options Ran Length: {len(PredictiveRoutes.system_list)}",
          flush=True)
    top_ToT = best_system.time_on_target
    top_path = best_system.predictive_path

    print(
        f"Best Time on Target: {timedelta(seconds = round(top_ToT))} ({round(top_ToT)} seconds)",
        flush=True)

    for idx, path in enumerate(top_path):
        add_tot = getattr(best_system, f"additional_tot_{idx+1}")

        print(f"{path} | {timedelta(seconds=add_tot)}", flush=True)

    # Reset Precision
    system_state.time_step = 2
    system_state.Predictive_System = False

    # reset PID
    system_state.PID = 1

    # clear temp folder
    if os.path.exists(dir_path):
        pass

    del (PredictiveRoutes)
    gc.collect()

    # return ideal path
    return best_system


def recursive_route_explorer(
        PredictiveRoutes: SystemConfiguration.RouteStorage,
        system_state: SystemConfiguration.SimpleSystemState,
        recursion_depth: int):
    '''
    Recursively called function to populate dictionary of all combinations of transitionary
    options with their respective Times on Target

    STEPS:

    Function 1: Matrix of Transition Options
    Takes in the current location, time, and outer radius. It will then simulate the balloons trajectory at its 
    current altitude. every predetermined step (25km) it will record the current time and location. It will do 
    this every 25km until the Balloon leave the max radius. These will thus list
    options to investigate --> time (and the corresponding location), as well as their additional time on target.
    This can then be iterated over with every altitude option to give every transition option


    For every combo of time/altitude:
    -- add the combo to the path list (time,alt)

    Function 2: Simulate Transition
    Takes in the current time, location, and the altitude to investigate, and then will simulate the balloon
    transitioning to that new location, then return the time and location of the balloon after transitioning,
    as well as the additional time on target from that transition

    Function3: Simulate Float
    Takes in the current time and location of the balloon immediately after transitioning to the new altitude
    and then simulates the balloon floating. It will float until the conditions are met to begin investigating
    new altitudes again, where it will return current time and location, and the additional time spent on target.

    Recursion:
    Run route_identifier with the current conditions and 1 less depth

    Base case:
    When depth == 0 the path list will have 3 transitions. Adds that path to the Output dictionary, with the
    key being the the path (as to ensure the dictionary length is correct) and the value is the ToT.

    Thus each route will be recursively explored, and upon completion of exploration, the route will be added
    to the output dictionary
    '''

    # base case
    if system_state.depth == 0:
        # add path to output dictionary

        PredictiveRoutes.add_route(system_state.clone())
        current_best_system = PredictiveRoutes.get_best_system().clone()

        outputText = f"#{len(PredictiveRoutes.system_list)}: "
        for path in system_state.predictive_path:
            formatted_dt = path.transition_time.strftime("%m-%d-%Y %H:%M:%S")
            outputText += f"({formatted_dt}, {round(path.transition_altitude)}) "

        outputText += f'| {round(system_state.time_on_target)}'
        for i in range(1, recursion_depth + 1):
            add_tot = getattr(system_state, f"additional_tot_{i}")
            outputText += f'| {round(add_tot)}'

        outputText += f'| Best ToT: {round(current_best_system.time_on_target)}'
        print(outputText, flush=True)
        del (system_state)
        return

    ### Route Reducing Protocol ###
    # if already finished first path --> if first path gave 0 additional time on target --> dont run anymore
    if recursion_depth > 0:
        st = datetime.now()
        current_best_system = PredictiveRoutes.get_best_system().clone()
        if current_best_system.time_on_target > system_state.time_on_target:

            for i in range(1, recursion_depth + 1):
                add_tot = getattr(system_state, f"additional_tot_{i}")
                add_tot_best = getattr(current_best_system,
                                       f"additional_tot_{i}")

                if add_tot == 0 and add_tot_best != 0:
                    print(
                        f"No Longer Pursuing System with ToT: {system_state.time_on_target}",
                        flush=True)
                    return
        system_state.queued_time += (datetime.now() - st).total_seconds()

    # generate transition matrix
    tt = datetime.now()
    option_set: list[
        SystemConfiguration.SimpleSystemState] = transition_matrix_generator(
            system_state.clone())
    system_state.transit_matrix_time += (datetime.now() - tt).total_seconds()

    # check if at top level of recursion --> if so--> multiprocess all sets
    if recursion_depth == 0:

        # create list of (system,altitude) to be investigated
        combos: list[tuple[SystemConfiguration.SimpleSystemState, float]] = []
        for system in option_set:
            # reset forecast
            system.current_forecast = SystemConfiguration.ERA5Forecast(
                0, 0, None)
            for altitude in forecast.ALTITUDE_OPTIONS:

                combos.append((system.clone(), altitude))

        print(
            f'Matrix Generated | {len(combos)}  options to run over {multiprocessing.cpu_count()}',
            flush=True)

        pool = multiprocessing.Pool()

        # clear forecast data
        processes = [
            pool.apply_async(parallel_helper_func,
                             args=(PredictiveRoutes, pair[0].clone(), pair[1],
                                   recursion_depth, idx))
            for idx, pair in enumerate(combos)
        ]
        [p.get() for p in processes]

    else:
        # get initial time on target
        init_tot = system_state.time_on_target

        # iterate through transition matrix options

        for system in option_set:

            # iterate through altitudes
            for altitude in [
                    alt for alt in forecast.ALTITUDE_OPTIONS
                    if alt != system.altitude
            ]:

                # sets balloon to iterated system
                system_state = system.clone()

                # add time,alt combo to path list
                system_state.predictive_path.append(
                    SystemConfiguration.TransitionOption(
                        system_state.time, altitude))
                setattr(
                    system_state, f"decision_{recursion_depth+1}",
                    SystemConfiguration.TransitionOption(
                        system_state.time, altitude))

                # Perform Altitude Transition Simulation
                system_state = altitude_transition_simulator(
                    system_state.clone(), altitude)

                # Perform Float Simulation
                system_state = float_transition_simulator(system_state.clone())

                setattr(system_state, f"additional_tot_{recursion_depth+1}",
                        system_state.time_on_target - init_tot)

                # check to see if better system
                st = datetime.now()
                PredictiveRoutes.set_best_system(system_state)
                system_state.queued_time += (datetime.now() -
                                             st).total_seconds()

                # Explore Further Routes
                system_state.depth -= 1

                # Explore Routes
                recursive_route_explorer(PredictiveRoutes,
                                         system_state.clone(),
                                         recursion_depth + 1)

    # after exploring all options --> return full OutputDict
    return PredictiveRoutes


def parallel_helper_func(PredictiveRoutes: SystemConfiguration.RouteStorage,
                         system_state: SystemConfiguration.SimpleSystemState,
                         altitude: float, recursion_depth: int, idx: int):

    # records node start time
    node_start_time = datetime.now()
    system_state = system_state.clone()
    system_state.node_start_time = node_start_time

    # set PID
    pid = multiprocessing.current_process().pid
    system_state.PID = pid
    system_state.time_spent_accessing_data = 0
    system_state.transit_matrix_time = 0
    system_state.altitude_change_time = 0
    system_state.float_time = 0
    system_state.queued_time = 0

    print(
        f'Beginning Traversal of Route #{idx}: {system_state.time} to {round(altitude)} on {pid}',
        flush=True)

    # add time,alt combo to path list
    system_state.predictive_path.append(
        SystemConfiguration.TransitionOption(system_state.time, altitude))
    setattr(system_state, f"decision_{recursion_depth+1}",
            SystemConfiguration.TransitionOption(system_state.time, altitude))

    # get initial time on target
    init_tot = system_state.time_on_target

    try:
        # Perform Altitude Transition Simulation
        system_state = altitude_transition_simulator(system_state.clone(),
                                                     altitude)

        # Perform Float Simulation
        system_state = float_transition_simulator(system_state.clone())

        setattr(system_state, f"additional_tot_{recursion_depth+1}",
                system_state.time_on_target - init_tot)

        # check to see if better system
        st = datetime.now()
        PredictiveRoutes.set_best_system(system_state)
        system_state.queued_time += (datetime.now() - st).total_seconds()

        # Explore Further Routes
        system_state.depth -= 1
        recursive_route_explorer(PredictiveRoutes, system_state.clone(),
                                 recursion_depth + 1)

    except Exception as e:
        print(f"Node {pid} Failed due to {str(e)}",
              file=sys.stderr,
              flush=True)
        raise Exception(f"[LVS 720: Node {pid} Failed due to {str(e)}]")

    node_run_time = (datetime.now() - node_start_time).total_seconds()
    print(f"{pid} runtime: {node_run_time}", flush=True)

    # delete node
    data_path = rf'/data/temp/{system_state.sim_name}/{system_state.PID}/'
    data_path = rf'/data/temp/{system_state.sim_name}/{system_state.PID}/'
    shutil.rmtree(data_path)


def transition_matrix_generator(
        system_state: SystemConfiguration.SimpleSystemState):

    # set distance between investigations
    distance_delta = DISTANCE_DELTA

    # get current distance to target
    distance_from_target = utils.coordinate_difference(
        system_state.target_location,
        [system_state.latitude, system_state.longitude])

    option_set = []

    # set precision
    system_state.time_step = 15

    # iterate though different transition beginning times (and locations)
    # only do locations that will begin transitioning within max radius
    while (distance_from_target < MAX_RADIUS):

        # reset distance traveled
        distance_traveled = 0

        # record system state
        option_set.append(system_state.clone())

        # loop while moved less than desired distance
        while distance_traveled < distance_delta:
            if (system_state.time > END_DATE) or (system_state.time
                                                  < START_DATE):
                return option_set

            # Update system
            try:
                Prediction.generate_next_step_viability(system_state)
            except Exception as e:
                print(
                    f"Failed to generate next step for {system_state.PID} due to {str(e)}",
                    file=sys.stderr,
                    flush=True)
                raise ValueError(
                    f"[LVS 784: Failed to generate next step for {system_state.PID} due to {str(e)}]"
                )

            # see if within coverage radius --> add to Time on Target
            if distance_from_target < COVERAGE_RADIUS:
                system_state.time_on_target += system_state.time_step

            # update distance traveled
            distance_traveled += utils.tangent_distance(
                np.array([system_state.velocity.u, system_state.velocity.v]) *
                system_state.time_step)

            # get new distance to target --> make sure not outside radius
            distance_from_target = utils.coordinate_difference(
                system_state.target_location,
                [system_state.latitude, system_state.longitude])

    # return list of transitionary times/locations
    return option_set


def altitude_transition_simulator(
        system_state: SystemConfiguration.SimpleSystemState,
        target_altitude: float):
    '''does a quick simulation to move the balloon from one altitude to the next,
    and return the position of the balloon at that new altitude, and the new time, and how much additional time it spent on target
    '''

    transit_start = datetime.now()
    # if the heights are the same --> return same location
    if np.abs(system_state.altitude - target_altitude) < 50:
        return system_state

    # set transition up
    if system_state.altitude < target_altitude:
        system_state.status = 'Ascent'
    else:
        system_state.status = 'Descent'

    system_state.target_altitude = target_altitude

    # set precision
    system_state.time_step = 60

    # loop until the system is at float
    while system_state.status != 'Float':

        if (system_state.time > END_DATE) or (system_state.time < START_DATE):
            return system_state

        try:
            Prediction.generate_next_step_viability(system_state)
        except Exception as e:
            print(
                f"Failed to generate next step for {system_state.PID} due to {str(e)}",
                file=sys.stderr,
                flush=True)
            raise ValueError(
                f"[LVS 842: Failed to generate next step for {system_state.PID} due to {str(e)}]"
            )

        # get new distance to target --> to see if in coverage radius
        distance_from_target = utils.coordinate_difference(
            system_state.target_location,
            [system_state.latitude, system_state.longitude])

        # see if within coverage radius --> add to Time on Target
        if distance_from_target < COVERAGE_RADIUS:
            system_state.time_on_target += system_state.time_step

    system_state.altitude_change_time += (datetime.now() -
                                          transit_start).total_seconds()
    return system_state


def float_transition_simulator(
        system_state: SystemConfiguration.SimpleSystemState):
    '''function that will take in the current location time and altitude, and simulate the balloon
    floating until it meets the criteria to investigate alternative altitudes. At which point
    it will return the current time, location, and how much additional time it spent on target'''

    float_start_time = datetime.now()
    # initialize values
    current_distance_from_target = utils.coordinate_difference(
        system_state.target_location,
        [system_state.latitude, system_state.longitude])
    previous_distance_from_target = float('inf')
    conditions = [False]

    # set precision
    system_state.time_step = 60

    # iterate until the previous distance is < current distance (thus moving away from target)
    while not all(conditions):

        # increment distances
        previous_distance_from_target = current_distance_from_target

        ## simulate generic balloon floating behavior ##

        if (system_state.time > END_DATE) or (system_state.time < START_DATE):
            return system_state

        try:
            Prediction.generate_next_step_viability(system_state)
        except Exception as e:
            print(
                f"Failed to generate next step for {system_state.PID} due to {str(e)}",
                file=sys.stderr,
                flush=True)
            raise ValueError(
                f"[LVS 896: Failed to generate next step for {system_state.PID} due to {str(e)}]"
            )

        # see if within coverage radius --> add to Time on Target
        if current_distance_from_target < COVERAGE_RADIUS:
            system_state.time_on_target += system_state.time_step

        # Determine distance from target
        current_distance_from_target = utils.coordinate_difference(
            system_state.target_location,
            [system_state.latitude, system_state.longitude])

        # determine if needs to look for alternative altitudes
        conditions = [
            (current_distance_from_target > MIN_DISTANCE_FROM_TARGET),
            (previous_distance_from_target < current_distance_from_target),
        ]

    system_state.float_time += (datetime.now() -
                                float_start_time).total_seconds()
    # return final system
    return system_state


if __name__ == '__main__':

    sim_start_time = datetime.now()
    sim_end_time = sim_start_time + timedelta(hours=7 * 24 + 12)
    original_stdout = sys.stdout

    error_count = 0

    while True:

        if error_count >= 3:
            break

        if (datetime.now() > sim_end_time):
            break

        output_dir = 'Location Viability Outputs/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        location_list = pd.read_csv('Location Study List.csv',
                                    parse_dates=['date'],
                                    dtype={'str_index': str})
        # Iterate through each row in the DataFrame
        for idx, row in location_list.iterrows():
            # Construct the beginning part of the folder name
            folder_name_start = "#" + str(row['str_index'])

            # Check if there's a directory in target_dir that starts with folder_name_start
            if not any(
                    dir.startswith(folder_name_start)
                    for dir in os.listdir(output_dir)):
                break

        date = row['date'].strftime('%Y-%m-%d')
        filename = f"#{row['str_index']} - {date} ({round(row['latitude'])},{round(row['longitude'])})"
        output_dir += filename + '/'

        running_output_dir = output_dir[:-1] + ' Running/'
        config = SystemConfiguration.ViabilityConfig(
            start_time=row['date'],
            start_latitude=row['latitude'],  # TARGET_LOCATION[0],
            start_longitude=row['longitude'],  # TARGET_LOCATION[1],
            location=filename)

        if not os.path.exists(f'{running_output_dir}'):
            os.mkdir(f'{running_output_dir}')

        log_file = os.path.join(running_output_dir, 'output.txt')
        error_log_file = os.path.join(running_output_dir, 'error_log.txt')
        sys.stdout = open(log_file, 'w')
        sys.stderr = open(error_log_file, 'a')

        try:
            print(f"Beginning Run {filename}",
                  file=original_stdout,
                  flush=True)
            StationKeep(config, running_output_dir, error_log_file)
            print(f"Finished Run {filename}", file=original_stdout, flush=True)
            os.rename(running_output_dir, output_dir)

        except Exception as e:
            print(f"Failed Run {filename}: {str(e)}",
                  file=original_stdout,
                  flush=True)
            print(f"\nFailed to run due to {str(e)}\n",
                  file=sys.stderr,
                  flush=True)

            # rename folder to failed
            failed_output_dir = output_dir[:-1] + ' FAILED/'

            # Rename the directory
            os.rename(running_output_dir, failed_output_dir)

            error_count += 1

        # Clear Temp Folder
        dir_path = rf'/data/temp/{filename}/'
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
