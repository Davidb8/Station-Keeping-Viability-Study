from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import netCDF4


def compile_output_data():
    # Step 1: Set your main directory path
    main_directory = 'Location Viability Outputs'

    # Step 2: Initialize an empty list to store the DataFrames
    dataframes_list = []

    # Step 3: Loop through each subdirectory and read the CSV file
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path):
            csv_file_path = os.path.join(subdir_path, 'Sim Output Data.csv')
            if os.path.isfile(csv_file_path):
                df = pd.read_csv(csv_file_path)
                dataframes_list.append(df)
            else:
                print(f"NO OUTPUT DATA FOUND FOR {subdir}")

    # Step 4: Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes_list, ignore_index=True)

    # Step 5: Save the combined data to a new CSV file
    output_file_path = 'Location Viability Outputs/Full Sim Output Data.csv'
    combined_df.to_csv(output_file_path, index=False)

    print(
        "Data from all 'Sim Output Data.csv' files has been combined and saved"
    )


def gen_histograms(data: pd.DataFrame):

    ### GENERATE OVERALL HISTOGRAM BY LATITUDE ###

    unique_latitudes = np.sort(np.round(data['Start Latitude'].unique()))
    new_bins = list(unique_latitudes)
    new_labels = [f"{new_bins[i]}" for i in range(len(new_bins[:-1]))]

    # Categorize the 'Start Latitude' into the finer latitude regions
    data['Latitude Region'] = pd.cut(data['Start Latitude'],
                                     bins=new_bins,
                                     labels=new_labels,
                                     right=False)

    # Compute the average 'Time on Target' for each fine latitude region
    avg_time_on_target_fine = data.groupby(
        'Latitude Region')['Days on Target'].mean()

    # Compute the number of unique sites for each fine latitude region
    unique_sites_per_latitude = data.groupby(
        'Latitude Region')['Start Longitude'].nunique()

    # Plotting the results
    fig, ax = plt.subplots(figsize=(18, 10))
    fig = avg_time_on_target_fine.plot(kind='bar',
                                       ax=ax,
                                       color='lightcoral',
                                       label='average')

    # Annotate the number of unique sites below each bar

    plt.text(-1.2,
             -5.5,
             f'Sites\nper\nLatitude',
             ha='center',
             va='bottom',
             rotation=0,
             color='black',
             fontsize=25)
    for i, (value, count) in enumerate(
            zip(avg_time_on_target_fine, unique_sites_per_latitude)):
        plt.text(i,
                 -4.5,
                 f'{count}',
                 ha='center',
                 va='bottom',
                 rotation=0,
                 color='black',
                 fontsize=20)

    # Determine the bounding box dimensions
    box_height = 3  # additional padding
    box_width = 20
    box_bottom_left_x = -1.5  # subtracting half bar width
    box_bottom_left_y = -4  # additional padding

    # Draw bounding box
    rectangle = Rectangle((box_bottom_left_x, box_bottom_left_y),
                          box_width,
                          box_height,
                          fill=False,
                          edgecolor='black',
                          linewidth=1.2)
    # ax.add_patch(rectangle)

    plt.title("Average Time on Target all Season", fontsize=35, pad=15)
    plt.xlabel('Latitude Region', fontsize=25)
    plt.ylabel('Average Time on Target (Days)', fontsize=25)
    plt.xticks(rotation=90, fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    plt.savefig("Full Histogram.svg")
    plt.show()

    ### GENERATE SEASONAL HISTOGRAMS BY LATITUDE ###

    avg_time_on_target_fine_season = data.groupby(
        ['Latitude Region', 'Season'])['Days on Target'].mean()

    y_max = round(avg_time_on_target_fine_season.max())

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    # Spring
    avg_time_on_target_fine_season[:, 'Spring'].plot(
        kind='bar', color='green', ax=axs[0, 0], label='Spring')
    axs[0, 0].set_title("Average Time on Target April", fontsize=30)
    axs[0, 0].set_xlabel('Latitude Region', fontsize=25)
    axs[0, 0].set_ylabel('Average Time on Target (Days)', fontsize=25)
    axs[0, 0].tick_params(axis='x', labelrotation=90, labelsize=20)
    axs[0, 0].tick_params(axis='y', labelsize=20)
    axs[0, 0].set_ylim((0, y_max))

    # Autumn
    avg_time_on_target_fine_season[:, 'Autumn'].plot(
        kind='bar', color='orange', ax=axs[0, 1], label='Autumn')
    axs[0, 1].set_title("Average Time on Target October", fontsize=30)
    axs[0, 1].set_xlabel('Latitude Region', fontsize=25)
    axs[0, 1].set_ylabel('Average Time on Target (Days)', fontsize=25)
    axs[0, 1].tick_params(axis='x', labelrotation=90, labelsize=20)
    axs[0, 1].tick_params(axis='y', labelsize=20)
    axs[0, 1].set_ylim((0, y_max))

    # Winter
    avg_time_on_target_fine_season[:, 'Winter'].plot(
        kind='bar', color='blue', ax=axs[1, 0], label='Winter')
    axs[1, 0].set_title("Average Time on Target January", fontsize=30)
    axs[1, 0].set_xlabel('Latitude Region', fontsize=25)
    axs[1, 0].set_ylabel('Average Time on Target (Days)', fontsize=25)
    axs[1, 0].tick_params(axis='x', labelrotation=90, labelsize=20)
    axs[1, 0].tick_params(axis='y', labelsize=20)
    axs[1, 0].set_ylim((0, y_max))

    # Summer
    avg_time_on_target_fine_season[:, 'Summer'].plot(
        kind='bar', color='red', ax=axs[1, 1], label='Summer')
    axs[1, 1].set_title('Average Time on Target July', fontsize=30)
    axs[1, 1].set_xlabel('Latitude Region', fontsize=25)
    axs[1, 1].set_ylabel('Average Time on Target (Days)', fontsize=25)
    axs[1, 1].tick_params(axis='x', labelrotation=90, labelsize=20)
    axs[1, 1].tick_params(axis='y', labelsize=20)
    axs[1, 1].set_ylim((0, y_max))

    plt.tight_layout()
    plt.savefig("Seasonal Histograms.svg")
    plt.show()


def numerical_output(data):
    # 2. Average "Time on Target" by Season
    avg_time_by_season = data.groupby('Season')['Time on Target'].mean() / (
        24 * 3600)

    # 3. Compute the average "Time on Target" for each combination of season and latitude region
    avg_time_by_season_and_latitude = data.groupby(
        ['Latitude Region', 'Season'])['Time on Target'].mean().unstack()

    # 4. Relative Performance by Latitude Region and Season
    grouped_data = data.groupby(['Latitude Region',
                                 'Season'])['Time on Target'].mean().unstack()
    overall_avg_time_on_target = data.groupby(
        'Latitude Region')['Time on Target'].mean()
    relative_performance = grouped_data.subtract(overall_avg_time_on_target,
                                                 axis=0)

    avg_time_by_start_date_and_latitude = data.groupby(
        ['Latitude Region', 'Start Time'])['Time on Target'].mean().unstack()

    print('avg_time_by_season')
    print(avg_time_by_season)
    print('avg_time_by_season_and_latitude')
    print(avg_time_by_season_and_latitude)
    print('relative_performance')
    print(relative_performance)
    print('avg_time_by_start_date_and_latitude')
    print(avg_time_by_start_date_and_latitude)


def trajectoryPloting():
    # Load datasets
    bad_trajectory = pd.read_csv(
        r'Location Viability Outputs\#06652 - 2018-07-01 (10,18)\#06652 - 2018-07-01 (10,18) 2018 - Jul Finished.csv',
        parse_dates=['time'])
    good_trajectory = pd.read_csv(
        r'Location Viability Outputs\#08898 - 2019-10-01 (-10,124)\#08898 - 2019-10-01 (-10,124) 2019 - Oct Finished.csv',
        parse_dates=['time'])

    # Filter good trajectory to only include the first 2 days
    good_trajectory = good_trajectory[good_trajectory['flight_duration'] <= 2 *
                                      24 * 60 * 60]

    # Extract origin points
    good_origin = (good_trajectory['longitude'].iloc[0],
                   good_trajectory['latitude'].iloc[0])
    bad_origin = (bad_trajectory['longitude'].iloc[0],
                  bad_trajectory['latitude'].iloc[0])

    # Plotting function

    # Extracting altitude and time data for the first two days for 08898
    good_trajectory_altitude = good_trajectory[['flight_duration', 'altitude']]

    good_trajectory['total_velocity'] = np.sqrt(good_trajectory['u']**2 +
                                                good_trajectory['v']**2)
    bad_trajectory['total_velocity'] = np.sqrt(bad_trajectory['u']**2 +
                                               bad_trajectory['v']**2)

    min_altitude = 15000
    max_altitude = 32000
    norm = plt.Normalize(min_altitude, max_altitude)
    cmap = plt.cm.viridis

    def plot_trajectory(data, origin, title, ax):
        # Create a color map based on altitude
        colors = cmap(norm(data['altitude']))

        # Add origin marker
        ax.scatter(origin[0], origin[1], color='red', label='Origin', s=100)

        # Add circles for radius
        circle_50 = plt.Circle(origin,
                               50 / 111,
                               color='g',
                               fill=False,
                               linewidth=2,
                               label='50km radius')
        circle_250 = plt.Circle(origin,
                                250 / 111,
                                color='k',
                                fill=False,
                                linewidth=2,
                                label='250km radius')
        ax.add_artist(circle_50)
        ax.add_artist(circle_250)

        # Scatter plot for trajectory with color mapping
        sc = ax.scatter(data['longitude'],
                        data['latitude'],
                        c=data['altitude'],
                        cmap=cmap,  # Ensure the colormap is specified
                        # norm=norm,  # Ensure normalization is used here
                        vmin=min_altitude,
                        vmax=max_altitude,
                        label='Trajectory',
                        s=10)

        # Setting axis limits and labels
        ax.set_xlim(origin[0] - 2.5, origin[0] + 2.5)
        ax.set_ylim(origin[1] - 2.5, origin[1] + 2.5)
        ax.set_xlabel('Longitude', fontsize=25)
        ax.set_ylabel('Latitude', fontsize=25)
        ax.tick_params(labelsize=20)
        ax.set_title(title, fontsize=30, pad=20)
        # Add this line to make the plot square in aspect
        ax.set_aspect('equal')

        # ax.legend()
        ax.grid(True)

        # Add a colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=.12)
        cbar.set_label('Altitude', fontsize=25)
        cbar.ax.tick_params(labelsize=20)
        # cbar.set((min_altitude,max_altitude))

    # Plot good trajectory

    # Create figure and axes for side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns

    # Plot trajectories on their respective axes
    duration_bad = bad_trajectory['flight_duration'].to_numpy()[-1] / (60 * 60)
    plot_trajectory(good_trajectory, good_origin,
                    'Good Trajectory (#08898) 48 hours', axs[0])
    plot_trajectory(bad_trajectory, bad_origin,
                    f'Bad Trajectory (#06652) {round(duration_bad)} hours', axs[1])

    plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
    plt.savefig("Trajectories Side by Side.svg")
    #
    #
    #
    #
    #

    # Plotting good altitude over time with color
    fig, ax1 = plt.subplots(figsize=(16, 7))
    colors = cmap(norm(good_trajectory['altitude']))
    sc = ax1.scatter(good_trajectory['flight_duration'] / (60 * 60),
                     good_trajectory['altitude'],
                     c=colors,
                     s=10)
    # Instantiate a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Velocity (m/s)', color=color, fontsize=25)
    ax2.plot(good_trajectory['flight_duration'] / (60 * 60),
             good_trajectory['total_velocity'],
             color=color,
             label='Velocity')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=20)

    # cbar = plt.colorbar(sc, ax=ax,)
    # cbar.set_label('Altitude')
    ax1.set_title(
        'Balloon Altitude and Velocity Over Time for #08898 (First 48 Hours)', fontsize=30, pad=15)
    ax1.set_xlabel('Time (hours)', fontsize=25)
    ax1.set_ylabel('Altitude (m)', fontsize=25)
    ax1.tick_params(labelsize=20)

    plt.savefig("Good Trajectory Altitude Plot.svg")
    #
    #
    #
    #

    # Plotting bad altitude over time with color
    fig, ax1 = plt.subplots(figsize=(16, 7))
    colors = cmap(norm(bad_trajectory['altitude']))
    sc = ax1.scatter(bad_trajectory['flight_duration'] / (60 * 60),
                     bad_trajectory['altitude'],
                     c=colors,
                     s=10)
    # cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('Altitude')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Velocity (m/s)', color=color, fontsize=25)
    ax2.plot(bad_trajectory['flight_duration'] / (60 * 60),
             bad_trajectory['total_velocity'],
             color=color,
             label='Velocity')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=20)

    ax1.set_title(
        'Balloon Altitude and Velocity Over Time for #06652 (First 12 Hours)', fontsize=30, pad=15)
    ax1.set_xlabel('Time (hours)', fontsize=25)
    ax1.set_ylabel('Altitude (m)', fontsize=25)
    ax1.tick_params(labelsize=20)

    plt.savefig("Bad Trajectory Altitude Plot.svg")
    plt.show()


def wind_plots():

    # Open the dataset
    dataset = netCDF4.Dataset(r'')
    level_data = [(64, 14968.24, 15003.50), (63, 15292.44, 15329.24),
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

    # Extract the relevant data
    latitudes = dataset.variables['latitude'][:]
    levels = dataset.variables['level'][:]
    time = dataset.variables['time'][:]
    longitudes = dataset.variables['longitude'][:]
    u_wind = dataset.variables[
        'u'][:, :, :, :]  # Assuming the variable name is 'u'
    v_wind = dataset.variables[
        'v'][:, :, :, :]  # Assuming the variable name is 'v'

    timestep = 15  # for example, to choose the first timestep
    longitude_index = 15  # for example, to choose the first longitude

    fig, ax = plt.subplots(figsize=(8, 10))
    unique_latitudes = []  # List to collect unique latitude indices for x-ticks
    for longitude_index in range(660, 680, 60):
        cmap = plt.cm.twilight  # Colormap for mapping directions to colors

        for latidx in range(0, 721, 90):
            latitude_index = latidx

            if dataset.variables['latitude'][latitude_index] not in unique_latitudes:
                unique_latitudes.append(
                    dataset.variables['latitude'][latitude_index])

            # longitude_index = 90  # Example longitude
            timestep = 20  # Example timestep

            u_data = u_wind[timestep, 3:, latitude_index, longitude_index]
            v_data = v_wind[timestep, 3:, latitude_index, longitude_index]

            # Normalize wind vectors
            magnitude = np.sqrt(u_data**2 + v_data**2)
            magnitude[magnitude == 0] = 1
            u_normalized = u_data / magnitude
            v_normalized = v_data / magnitude

            # Calculate angle and map it to a color
            angle = np.arctan2(v_normalized, u_normalized)
            # Normalize angle to [0, 1] and map to color
            color = cmap((angle + np.pi) / (2 * np.pi))

            x_vals = np.full_like(
                u_normalized, dataset.variables['latitude'][latitude_index])
            y_vals = [d[2] for d in level_data]

            ax.quiver(x_vals[1::2], y_vals[1::2], u_normalized[1::2], v_normalized[1::2], color=color[1::2],
                      units='dots', scale=0.05, width=5, scale_units='dots',  headaxislength=3, headlength=3.5)

        ax.set_xticks(unique_latitudes)
        ax.set_xticklabels(
            [f"{lat:.2f}" for lat in unique_latitudes], rotation=45, ha="right")

        # ax = plt.gca()  # Get current axis
        # Adjust x, y, size as needed
        plt.title(
            f'Wind Column Direction at Various Latitudes |\n Longitude {dataset.variables["longitude"][longitude_index]} | July 2021', fontsize=25, pad=15)
        plt.xlabel('Latitude', fontsize=25)
        plt.ylabel('Altitude (m)', fontsize=25)
        plt.ylim(level_data[0][2]-1000, level_data[-1][2]+1000)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.grid(True)
        # cbar = plt.colorbar(label='Wind Direction (°)')
        # cbar.set_ticks(np.arange(0, 360, 30))
        plt.savefig("Wind Plot.svg")
        plt.show()


df = pd.read_csv(r'Output Data.csv',
                 parse_dates=['Start Time', 'End Time'])

compile_output_data()
wind_plots()

df = pd.read_csv(r'Full Sim Output Data.csv',
                 parse_dates=['Start Time', 'End Time'])

# trajectoryPloting()
# gen_histograms(df)
# numerical_output(df)
