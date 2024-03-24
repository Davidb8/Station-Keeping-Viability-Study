from dataclasses import dataclass, field, fields
from datetime import datetime
import pandas as pd
import math
import simplekml
import copy
import utils
from typing import NamedTuple
import multiprocessing


class VelocityVector(NamedTuple):
    u: float
    v: float
    z: float


@dataclass()
class ViabilityConfig():
    '''These are data values for initializing a system to run a viability study'''
    start_time: datetime
    start_latitude: float
    start_longitude: float
    location: str


@dataclass()
class ERA5Forecast():
    day: int
    level: int
    data: list


class TransitionOption:

    def __init__(self, transition_time: datetime, transition_altitude: float):
        self.transition_time = transition_time
        self.transition_altitude = transition_altitude

    def __str__(self):
        time = self.transition_time.strftime("%m-%d-%Y %H:%M:%S")
        return f"({time}, {round(self.transition_altitude)})"


@dataclass()
class SimpleSystemState:
    '''Class of Dynamic Conditions of Balloon system for a simple Balloon system'''
    # Dynamic Condition
    status: str
    time: datetime
    latitude: float = field(metadata={'units': 'degrees'})
    longitude: float = field(metadata={'units': 'degrees'})
    altitude: float = field(metadata={'units': 'meters'})
    velocity: VelocityVector = field(metadata={'units': 'm/s'})
    target_altitude: float = field(metadata={'units': 'm'})
    flight_path: list[dict]
    altitude_decisions: list[str]
    time_on_target: float = field(metadata={'units': 's'})
    current_forecast: ERA5Forecast
    PID: int
    Predictive_System: bool
    viability_study: bool
    jumps_remaining: int
    target_location: list
    depth: int
    predictive_path: list[TransitionOption]
    flight_duration: int
    sim_name: str
    time_step: float
    start_date: datetime
    time_spent_accessing_data: float
    transit_matrix_time: float
    altitude_change_time: float
    float_time: float
    queued_time: float
    node_start_time: datetime

    def __init__(self):
        self.time = None
        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.status = None
        self.target_altitude = None
        self.time_on_target = 0
        self.flight_path = []
        self.altitude_decisions = []
        self.current_forecast = ERA5Forecast(None, None, None)
        self.PID = None
        self.Predictive_System = False
        self.predictive_path = []
        self.flight_duration = 0
        self.sim_name = None
        self.time_step = None
        self.time_spent_accessing_data = 0
        self.transit_matrix_time = 0
        self.altitude_change_time = 0
        self.float_time = 0
        self.queued_time = 0
        self.node_start_time = None

    def prep_recursion(self, depth):
        '''prepares the system for existing under recursion'''
        for i in range(1, depth + 1):
            setattr(self, f"decision_{i}", None)
            setattr(self, f"additional_tot_{i}", 0)
        self.Predictive_System = True
        self.depth = depth
        setattr(self, f"additional_tot_total", 0)

    def update_flight_path(self):
        '''Adds current step to flight_path'''

        # create dictionary of current values
        step = {}
        for attr in self.__dict__:
            # Ignore private attributes and methods
            if not attr.startswith('__') and not (attr in [
                    'current_forecast', 'Predictive_System', 'PID',
                    'flight_path', 'viability_study'
            ]):

                value = getattr(self, attr)
                # Only include non-method attributes
                if not callable(value):
                    if attr == 'velocity':
                        step['u'] = value.u
                        step['v'] = value.v
                        step['z'] = value.z
                    if attr == 'altitude_decisions':
                        step[attr] = copy.deepcopy(value)
                    else:
                        step[attr] = value

            # adds step to flight_path

        self.flight_path.append(step)

    def get_position(self):
        return (self.time.strftime("%H:%M:%S"), self.status, self.altitude,
                self.latitude, self.longitude)

    def get_velocities(self):
        return (self.time.strftime("%H:%M:%S"), self.velocity.u,
                self.velocity.v, self.velocity.z)

    def log(self):
        return (self.time.strftime("%H:%M:%S"), self.status,
                round(self.altitude, 1), round(self.velocity.u, 1),
                round(self.velocity.v, 1), round(self.velocity.z, 2),
                round(self.latitude, 4), round(self.longitude,
                                               4), round(self.time_on_target))

    def export_flight_df(self, downsample=0):
        # Create a list of dictionaries with velocity split into u, v, and z

        dict = self.flight_path
        if downsample != 0:
            dict = utils.downsample(dict, sample=downsample)

        # Use the first dictionary to extract the column names
        columns = list(dict[0].keys())

        # Create the DataFrame directly from the list of dictionaries
        df = pd.DataFrame(dict, columns=columns)

        return df

    def export_flight_kml(self, flightname):

        coordinates = []

        # get list of coordinates

        for dict in self.flight_path:
            # extract the values for latitude, longitude, and altitude
            lat = dict['latitude']
            lon = dict['longitude']
            alt = dict['altitude']
            # create a new tuple with these values and append it to the list
            coordinates.append((lon, lat, alt))

        # convert to kml
        kml = simplekml.Kml()
        linestring = kml.newlinestring(name='Flight Path')

        linestring.coords = coordinates[::10]

        # set the style of the linestring
        linestring.style.linestyle.color = simplekml.Color.red
        linestring.style.linestyle.width = 4
        linestring.altitudemode = simplekml.AltitudeMode.absolute

        # save the KML file
        kml.save(f'{flightname} - flight_path.kml')

    def export_flight_kml_large(self, flightname):
        coordinates = []

        # get list of coordinates
        for dict in self.flight_path:
            # extract the values for latitude, longitude, and altitude
            lat = dict['latitude']
            lon = dict['longitude']
            alt = dict['altitude']
            # create a new tuple with these values and append it to the list
            coordinates.append((lon, lat, alt))

        # convert to kml
        kml = simplekml.Kml()

        # split the coordinates into chunks
        max_coordinates_per_linestring = 64000
        num_chunks = math.ceil(
            len(coordinates) / max_coordinates_per_linestring)

        for i in range(num_chunks):
            chunk_coordinates = coordinates[i * max_coordinates_per_linestring:
                                            (i + 1) *
                                            max_coordinates_per_linestring]

            # create a new LineString for each chunk
            linestring = kml.newlinestring(name=f'Flight Path - Part {i + 1}')
            linestring.coords = chunk_coordinates[::]

            # set the style of the linestring
            linestring.style.linestyle.color = simplekml.Color.red
            linestring.style.linestyle.width = 4
            linestring.altitudemode = simplekml.AltitudeMode.absolute

        # save the KML file
        kml.save(f'{flightname} - flight_path.kml')

    def clone(self):
        self.current_forecast = ERA5Forecast(0, 0, None)
        clone = copy.deepcopy(self)
        return clone


class RouteStorage:
    system_list: list[SimpleSystemState]
    best_system: dict[str, SimpleSystemState]

    # best_system:SimpleSystemState

    def __init__(self, init_balloon: SimpleSystemState = None):

        manager = multiprocessing.Manager()
        self.system_list = manager.list()

        init_balloon.flight_path = []
        init_balloon.current_forecast = ERA5Forecast(0, 0, None)
        self.best_system = manager.dict()
        self.best_system['Best System'] = init_balloon.clone()
        self.best_system['Init System'] = init_balloon.clone()

    def add_route(self, balloon_system: SimpleSystemState) -> None:
        balloon_system.flight_path = []
        balloon_system.current_forecast = ERA5Forecast(0, 0, None)

        self.system_list.append(balloon_system.clone())

    def set_best_system(self, balloon_system: SimpleSystemState) -> None:
        if balloon_system.time_on_target > self.best_system[
                'Best System'].time_on_target:
            balloon_system.flight_path = []
            balloon_system.current_forecast = ERA5Forecast(0, 0, None)

            self.best_system['Best System'] = balloon_system.clone()

    def get_best_system(self) -> SimpleSystemState:
        return self.best_system['Best System'].clone()

    def get_init_system(self) -> SimpleSystemState:
        return self.best_system['Init System'].clone()

    def get_system_list(self) -> list[SimpleSystemState]:
        return copy.deepcopy(self.system_list)

    def sort_system_states(self, target_altitude):
        self.system_list.sort(
            key=lambda system:
            (system.time_on_target, -abs(system.altitude - target_altitude),
             system.time_on_target / system.flight_duration),
            reverse=True)
