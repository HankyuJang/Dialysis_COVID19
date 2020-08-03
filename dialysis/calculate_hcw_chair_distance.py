"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: Aug, 2020

Description: This script computes HCP to chair distances of a given day.

"""

import argparse
import pandas as pd
import numpy as np
import random as rd

def calculate_hcw_chair_distance(df_hcw, station_x, station_y, n_hcw, n_chairs, max_time):
    hcw_chair_dist = np.zeros((n_hcw, n_chairs, max_time))
    for t in range(1, max_time+1):
        # Calculate hcw chair distance
        hcw_chair_dist[:,:,t-1] = hcw_chair_distance(df_hcw, station_x, station_y, n_hcw, n_chairs, t)
    return hcw_chair_dist

def hcw_chair_distance(df_hcw, station_x, station_y, n_hcw, n_chairs, time):
    hcw_chair_dist = np.full((n_hcw, n_chairs), 47).astype(float)

    df_temp = df_hcw[df_hcw.time==time].reset_index(drop=True)
    for index, row in df_temp.iterrows(): # For each hcw
        for i in range(6):# For chairs in the right side
            if row['x'] > station_x[i][0]:
                d_x = 0
            else:
                d_x = station_x[i][0] - row['x']
            if row['y'] > station_y[i][1]:
                d_y = row['y'] - station_y[i][1]
            elif row['y'] > station_y[i][0]:
                d_y = 0
            else:
                d_y = station_y[i][0] - row['y']
            hcw_chair_dist[int(row['ID'])-1, i] = euclidean_dist(d_x, d_y)
        for i in range(6, n_chairs):
            if row['x'] > station_x[i][1]:
                d_x = row['x'] - station_x[i][1]
            else:
                d_x = 0
            if row['y'] > station_y[i][1]:
                d_y = row['y'] - station_y[i][1]
            elif row['y'] > station_y[i][0]:
                d_y = 0
            else:
                d_y = station_y[i][0] - row['y']
            hcw_chair_dist[int(row['ID'])-1, i] = euclidean_dist(d_x, d_y)
    return hcw_chair_dist

def distance_to_feet(distance):
    return distance / 0.042838596 # 12 * 0.003569883 = 0.042838596

def euclidean_dist(x, y):
    return pow(pow(x,2) + pow(y,2), 0.5) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute arrays for simulation at Dialysis unit')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    args = parser.parse_args()
    day = args.day

    df_station = pd.read_csv("data/station_0ft.csv")

    # Change the distance metric to feet
    df_station['x'] = distance_to_feet(df_station['x'])
    df_station['y'] = distance_to_feet(df_station['y'])

    # Check the station sizes (coordinates) and get min, max coordinates
    n_station = 12
    station_sizes = np.zeros((n_station, 2))
    station_x = np.zeros((n_station, 2))
    station_y = np.zeros((n_station, 2))

    for i in range(1, n_station+1):
        temp = df_station[df_station["station"]==i]

        station_x[i-1] = temp['x'].min(), temp['x'].max()
        station_y[i-1] = temp['y'].min(), temp['y'].max()
        
    ############################################################33
    print()
    print("Calculating hcw-chair distances in day {}".format(day))
    filename_hcw = "data/HCP_locations/latent_positions_day_{}.csv".format(day)
    df_hcw = pd.read_csv(filename_hcw)
    df_hcw['x'] = distance_to_feet(df_hcw['x'])
    df_hcw['y'] = distance_to_feet(df_hcw['y'])

    n_hcw = df_hcw.ID.max()
    n_chairs = 9
    max_time = df_hcw.time.max()

    hcw_chair_dist = calculate_hcw_chair_distance(df_hcw, station_x, station_y, n_hcw, n_chairs, max_time)

    for h in range(n_hcw):
        df_chair_dist = pd.DataFrame(
                data=hcw_chair_dist[h,:,:].T, 
                columns=["Chair"+str(c) for c in range(1,n_chairs+1)]
                )
        df_chair_dist.to_csv("data/HCP_chair_distance/day{}/HCP{}_chair_distance_day{}.csv".format(day, h+1, day), index=False)
