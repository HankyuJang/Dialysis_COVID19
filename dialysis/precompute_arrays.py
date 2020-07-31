"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: July, 2020

Description: This script precomputes contact in dialysis unit

Contact if two individuals (HCWs and Patients) are within 6 ft distances with each other.

# 1. HCW arrays
The script computes following numpy arrays that are used in the simulation for specified day and distance.
- day: day of the collected data (1 - 10)
- dist: distance that is used to define a contact (contact if distance btw entities < dist)

1. hcw_hcw_contact
2. hcw_hcw_contact_both_center
3. hcw_handwash_prox
4. hcw_center_prox
5. hcw_at_center
"""
import argparse
import pandas as pd
import numpy as np
import random as rd
import datetime 

def distance_to_feet(distance):
    return distance / 0.042838596 # 12 * 0.003569883 = 0.042838596

def get_distance(loc1, loc2):
    return np.sqrt(np.sum(np.power(loc1-loc2, 2)))

# Change HCW latent position measures to ft
def preprocess_hcw_positions(df_hcw):
    df_hcw['x'] = distance_to_feet(df_hcw['x'])
    df_hcw['y'] = distance_to_feet(df_hcw['y'])
    return df_hcw

# Generate station x, y coordinates from the given station file
def preprocess_st(df_st):
    df_st['x'] = distance_to_feet(df_st['x'])
    df_st['y'] = distance_to_feet(df_st['y'])

    n_station = df_st['station'].max()
    station_x = np.zeros((n_station, 2))
    station_y = np.zeros((n_station, 2))

    for i in range(1, n_station+1):
        temp = df_st[df_st["station"]==i]
        station_x[i-1] = temp['x'].min(), temp['x'].max()
        station_y[i-1] = temp['y'].min(), temp['y'].max()

    return station_x, station_y

def get_hcw_hcw_contact(time):
    hcw_hcw_pairs = []
    hcw_hcw_pairs_both_center = []

    df_temp = df_hcw[df_hcw.time==time].reset_index(drop=True)
    n = df_temp.shape[0]

    for i in range(n):
        for j in range(i+1, n):
            dist = get_distance(df_temp.loc[i, ['x', 'y']], df_temp.loc[j, ['x', 'y']])
            if dist <= dist:
                i_id = df_temp.loc[i, 'ID']
                i_x = df_temp.loc[i, 'x']
                i_y = df_temp.loc[i, 'y']

                j_id = df_temp.loc[j, 'ID']
                j_x = df_temp.loc[j, 'x']
                j_y = df_temp.loc[j, 'y']
                # Check if both hcws are in the nurses station
                if isin_station(11, i_x, i_y) and isin_station(11, j_x, j_y):
                    hcw_hcw_pairs_both_center.append((i_id, j_id))
                else:
                    hcw_hcw_pairs.append((i_id, j_id))

    return hcw_hcw_pairs, hcw_hcw_pairs_both_center#, hcw_hcw_pairs_center

# hcws who are at the hand-wahing station
def hcw_st(time):
    df_temp = df_hcw[df_hcw.time==time].reset_index(drop=True)
    hcw_hand_washing = []
    hcw_center = []

    for index, row in df_temp.iterrows(): # For each hcw
        # hcw-handwashing
        for i in range(n_chair, n_chair+2):
            if isin_station(i, row['x'], row['y']):
                hcw_hand_washing.append(int(row['ID']))
                # count += 1
        # hcw-nurses station
        for i in range(n_chair+2, n_chair+3):
            if isin_station(i, row['x'], row['y']):
                hcw_center.append(int(row['ID']))

    return hcw_hand_washing, hcw_center

# Return True if the HCW is in the station 
def isin_station(idx, pos_x, pos_y):
    return station_x[idx][0] < pos_x < station_x[idx][1] and station_y[idx][0] < pos_y < station_y[idx][1]

def get_hcw_arrays(n_hcw, max_time):
    temp_hcw_hand_washing = []
    temp_hcw_center = []
    temp_hcw_hcw_pairs = []
    temp_hcw_hcw_pairs_both_center = []
    hcw_at_center = np.zeros((n_hcw, max_time)).astype(bool)

    for t in range(1, max_time+1):
        # Is there hcw-patient contact? hcw hand washing event?
        hcw_hand_washing, hcw_center = hcw_st(t)
        temp_hcw_hand_washing.append(hcw_hand_washing)
        temp_hcw_center.append(hcw_center)

        for hcw in hcw_center:
            hcw_at_center[hcw-1, t-1] = True

        # Is there hcw-hcw contact?
        hcw_hcw_pairs, hcw_hcw_pairs_both_center = get_hcw_hcw_contact(t)

        temp_hcw_hcw_pairs.append(hcw_hcw_pairs)
        # temp_hcw_hcw_pairs_center.append(hcw_hcw_pairs_center)
        temp_hcw_hcw_pairs_both_center.append(hcw_hcw_pairs_both_center)

    #############################################################3
    # Start compute hcw_hcw_contact
    hcw_hcw_contact = np.zeros((n_hcw, n_hcw, max_time)).astype(bool)
    for t in range(max_time):
        for pair in temp_hcw_hcw_pairs[t]:
            if pair == []:
                continue
            else:
                hcw1, hcw2 = pair
                hcw_hcw_contact[hcw1-1, hcw2-1, t] = True
    # End compute hcw_hcw_contact
    #############################################################3
    # Start compute hcw_hcw_contact_both_center
    hcw_hcw_contact_both_center = np.zeros((n_hcw, n_hcw, max_time)).astype(bool)
    for t in range(max_time):
        for pair in temp_hcw_hcw_pairs_both_center[t]:
            if pair == []:
                continue
            else:
                hcw1, hcw2 = pair
                hcw_hcw_contact_both_center[hcw1-1, hcw2-1, t] = True
    # End compute hcw_hcw_contact_both_center
    #############################################################3
    # Start compute hcw_handwash_prox
    hcw_handwash_prox = np.zeros((n_hcw, max_time)).astype(bool)
    for t in range(max_time):
        for hcw in temp_hcw_hand_washing[t]:
            if hcw == []:
                continue
            else:
                hcw_handwash_prox[hcw-1, t] = True
    # End compute hcw_handwash_prox
    #############################################################3
    # Start compute hcw_center_prox
    hcw_center_prox = np.zeros((n_hcw, max_time)).astype(bool)
    for t in range(max_time):
        for hcw in temp_hcw_center[t]:
            if hcw == []:
                continue
            else:
                hcw_center_prox[hcw-1, t] = True
    # End compute hcw_center_prox
    #############################################################3

    return hcw_hcw_contact, hcw_hcw_contact_both_center, hcw_handwash_prox, hcw_center_prox, hcw_at_center 

def choose_time(chair_dwell_time, c, start, end):
    a = chair_dwell_time[c, start:end]
    p = a / a.sum()
    return start + np.random.choice(a=a, p=p)

# Generate numpy arrays that have relevant patient information for the simulation
# 1) patient_in_chair: boolean numpy array that denote whether the chair is occupied by a patient over the course of a day
# 2) t_in: patient arrival time index
# 3) t_out: patient leaving time index
# 4) chairs: chair corresponding t_in and t_out
def get_patient_info(t_in_s, t_in_e, t_out_s, t_out_e, chairs, chair_dwell_time, n_chair, max_time):
    n_patient = len(chairs)
    patient_in_chair = np.zeros((n_chair, max_time)).astype(bool)
    t_in = np.zeros(t_in_s.shape).astype(int)
    t_out = np.zeros(t_out_s.shape).astype(int)

    for idx in range(n_patient):
        t_incoming = choose_time(chair_dwell_time, chairs[idx], t_in_s[idx], t_in_e[idx])
        t_leaving = choose_time(chair_dwell_time, chairs[idx], t_out_s[idx], t_out_e[idx])
        patient_in_chair[chairs[idx], t_incoming: t_leaving] = True
        t_in[idx] = t_incoming
        t_out[idx] = t_leaving

    ###################################################################
    return patient_in_chair, np.array(t_in), np.array(t_out)

def patient_scheduling(session, morning, afternoon, n_patients, t_in, c_in):
    patients = np.arange(n_patients//2)
    morning_patients = (t_in <= morning).nonzero()[0]
    evening_patients = (t_in > afternoon).nonzero()[0]
    afternoon_patients = np.array(list(set(patients) - set(morning_patients) - set(evening_patients)))
    # print(morning_patients)
    # print(afternoon_patients)
    # print(evening_patients)

    morning_chairs = [c_in[patient] for patient in morning_patients]
    afternoon_chairs = [c_in[patient] for patient in afternoon_patients]
    evening_chairs = [c_in[patient] for patient in evening_patients]
    
    np.random.shuffle(morning_patients)
    np.random.shuffle(afternoon_patients)
    np.random.shuffle(evening_patients)
    # print()
    # print(morning_patients)
    # print(afternoon_patients)
    # print(evening_patients)

    patient_placement = {"morning": {}, "afternoon": {}, "evening": {}}
    for idx, c in enumerate(morning_chairs):
        if session == "MWF":
            patient_placement["morning"][c] = morning_patients[idx]
        else:
            patient_placement["morning"][c] = morning_patients[idx] + n_patients//2
    for idx, c in enumerate(afternoon_chairs):
        if session == "MWF":
            patient_placement["afternoon"][c] = afternoon_patients[idx]
        else:
            patient_placement["afternoon"][c] = afternoon_patients[idx] + n_patients//2
    for idx, c in enumerate(evening_chairs):
        if session == "MWF":
            patient_placement["evening"][c] = evening_patients[idx]
        else:
            patient_placement["evening"][c] = evening_patients[idx] + n_patients//2

    return patient_placement

def get_patient_arrays(simulation_length, n_hcw, max_time, morning, afternoon, n_chair, day, contact_distance):
    # hcw_chair_dist = np.load("data/hcw_chair_distance_day{}.npy".format(day))

    hcw_chair_dist = np.zeros((n_hcw, n_chair, max_time))
    for h in range(n_hcw):
        df_chair_dist = pd.read_csv("data/HCP_chair_distance/day{}/HCP{}_chair_distance_day{}.csv".format(day, h+1, day))
        hcw_chair_dist[h,:,:] = df_chair_dist.values.T

    hcw_chair_prox = hcw_chair_dist <= contact_distance

    # These are precomputed numpy arrays that will be used to sample patients
    # npzfile = np.load("data/patient_info_day{}.npz".format(day))
    # t_in_s, t_in_e, t_out_s, t_out_e = npzfile["t_in_s"], npzfile["t_in_e"], npzfile["t_out_s"], npzfile["t_out_e"]
    # chairs, chair_dwell_time  = npzfile["chairs"], npzfile["chair_dwell_time"]

    df_dialysis_session = pd.read_csv("data/dialysis_sessions/csv/patient_info_day_{}.csv".format(day))
    t_in_s = df_dialysis_session.t_in_s.values 
    t_in_e = df_dialysis_session.t_in_e.values 
    t_out_s = df_dialysis_session.t_out_s.values 
    t_out_e = df_dialysis_session.t_out_e.values 
    chairs = df_dialysis_session.chair.values 

    df_patient_at_chair_likelihood = pd.read_csv("data/dialysis_sessions/csv/patient_at_chair_likelihood_day_{}.csv".format(day))
    chair_dwell_time = df_patient_at_chair_likelihood.values.T

    # there are 2 sessions (MWF, TThS)
    n_patient = chairs.shape[0] * 2

    hcw_patient_contact_arrays = np.zeros((simulation_length, n_hcw, n_patient, max_time)).astype(bool)
    patient_patient_contact_arrays = np.zeros((simulation_length, n_patient, n_patient, max_time)).astype(bool)

    for d in range(simulation_length):
    # for d in [1]:
        hcw_patient_contact = np.zeros((n_hcw, n_patient, max_time)).astype(bool)
        patient_patient_contact = np.zeros((n_patient, n_patient, max_time)).astype(bool)
        ###################################################################
        # M,W,F
        if d % 7 in {0, 2, 4}:
            patient_in_chair, t_in, t_out = get_patient_info(t_in_s, t_in_e, t_out_s, t_out_e, chairs, chair_dwell_time, n_chair, max_time)
            patient_placement = patient_scheduling("MWF", morning, afternoon, n_patient, t_in, chairs)

        ###################################################################
        # T,Th,S
        elif d % 7 in {1, 3, 5}:
            patient_in_chair, t_in, t_out = get_patient_info(t_in_s, t_in_e, t_out_s, t_out_e, chairs, chair_dwell_time, n_chair, max_time)
            patient_placement = patient_scheduling("TThS", morning, afternoon, n_patient, t_in, chairs)

        ###################################################################
        # Sunday
        elif d % 7 == 6:
            continue 

        # HCW-Patient contact
        for p_idx, c in enumerate(chairs):
            contact = hcw_chair_prox[:, c, t_in[p_idx]:t_out[p_idx]]
            if t_in[p_idx] <= morning:
                hcw_patient_contact[:, patient_placement["morning"][c], t_in[p_idx]:t_out[p_idx]] = contact
            elif t_in[p_idx] > afternoon:
                hcw_patient_contact[:, patient_placement["evening"][c], t_in[p_idx]:t_out[p_idx]] = contact
            else:
                hcw_patient_contact[:, patient_placement["afternoon"][c], t_in[p_idx]:t_out[p_idx]] = contact
        # print(hcw_patient_contact.sum(axis=2).T)
        hcw_patient_contact_arrays[d] = hcw_patient_contact 

        # Patient-Patient contact
        for c in range(0,8):
            if c==5:
                continue
            # print(c)
            contact = patient_in_chair[c] & patient_in_chair[c+1]

            try:
                p1, p2 = patient_placement["morning"][c], patient_placement["morning"][c+1]
                patient_patient_contact[p1,p2,:morning] = contact[:morning]
                patient_patient_contact[p2,p1,:morning] = contact[:morning]
            except:
                pass
                # print("morning: chair is empty")

            try:
                p1, p2 = patient_placement["afternoon"][c], patient_placement["afternoon"][c+1]
                patient_patient_contact[p1,p2,morning:afternoon] = contact[morning:afternoon]
                patient_patient_contact[p2,p1,morning:afternoon] = contact[morning:afternoon]
            except:
                pass
                # print("afternoon: chair is empty")

            try:
                p1, p2 = patient_placement["evening"][c], patient_placement["evening"][c+1]
                patient_patient_contact[p1,p2,afternoon:] = contact[afternoon:]
                patient_patient_contact[p2,p1,afternoon:] = contact[afternoon:]
            except:
                pass
                # print("evening: chair is empty")

        patient_patient_contact_arrays[d] = patient_patient_contact 
        # np.savetxt("data/patient_patient_contact.csv", patient_patient_contact.sum(axis=2), fmt="%d", delimiter=',')
        # np.save("data/day{}/hcw_patient_contact_{}, hcw_patient_contact".format(day, d))
    return hcw_patient_contact_arrays, patient_patient_contact_arrays

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute arrays for simulation at Dialysis unit')
    parser.add_argument('-p', '--path', type=str, default="/usr/local/data/dialysis/",
                        help= 'path to the directory. If you are running this file in the Vinci server, use the default path.')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-d', '--distance', type=int, default=6,
                        help= 'distance threshold (in feet)')
    args = parser.parse_args()

    path = args.path
    day = args.day
    contact_distance = args.distance

    # Read date_time.txt that contains start, end times
    # df starts from 0. Day1 = 0th row. Day10 = 9th row
    df = pd.read_table("data/date_time.txt", sep='.', header=None, names = ['year','month','day','hour','minute','second'])
    row = df.iloc[day-1]
    session_start = datetime.datetime(row.year, row.month, row.day, row.hour, row.minute, row.second)
    morning_start = datetime.datetime(row.year, row.month, row.day, 9, 0, 0)
    morning = (morning_start - session_start).seconds // 8
    afternoon = morning + 5 * 60 * 60 // 8

    # number of chairs in the dialysis unit
    n_chair = 9
    # length in days
    simulation_length = 30

    ###################################################################
    # 1. Compute HCW arrays
    ###################################################################
    df_st = pd.read_csv("data/station_0ft.csv")
    station_x, station_y = preprocess_st(df_st)

    filename_hcw = "latent_positions_day_{}.csv".format(day)
    # Uncomment following line if you're using data stored in the server
    # filename_hcw = "LatentPositionsData/latent_positions_day_{}.csv".format(day)
    df_hcw = preprocess_hcw_positions(pd.read_csv(path + filename_hcw))

    max_time = df_hcw.time.max()
    n_hcw = df_hcw.ID.unique().shape[0]

    # HCW-HCW contact
    print("Computing HCW arrays: hcw_hcw_contact, hcw_hcw_contact_both_center, hcw_handwash_prox, hcw_center_prox, hcw_at_center...")
    hcw_hcw_contact, hcw_hcw_contact_both_center, hcw_handwash_prox, hcw_center_prox, hcw_at_center = get_hcw_arrays(n_hcw, max_time)

    outfile = "contact_data/hcw_arrays_day{}_{}ft".format(day, contact_distance)
    np.savez(outfile, hcw_hcw_contact=hcw_hcw_contact, hcw_hcw_contact_both_center=hcw_hcw_contact_both_center, hcw_handwash_prox=hcw_handwash_prox, hcw_center_prox=hcw_center_prox, hcw_at_center=hcw_at_center)

    ###################################################################
    # 2. Compute HCW-Patient and Patient-Patient contact over 30 days
    ###################################################################
    # >>> hcw_chair_dist.min()
    # 0.0
    # >>> hcw_chair_dist.max()
    # 47.0

    print("Computing HCW-Patient contact and Patient-Patient contact for {} consecutive days...".format(simulation_length))
    hcw_patient_contact_arrays, patient_patient_contact_arrays = get_patient_arrays(simulation_length, n_hcw, max_time, morning, afternoon, n_chair, day, contact_distance)

    # Zero out lower triangle
    n_patient = patient_patient_contact_arrays.shape[1]
    for j in range(n_patient):
        for i in range(j+1, n_patient):
            patient_patient_contact_arrays[:,i,j,:] = 0

    outfile = "contact_data/patient_arrays_day{}_{}ft".format(day, contact_distance)
    np.savez(outfile, hcw_patient_contact_arrays=hcw_patient_contact_arrays, patient_patient_contact_arrays=patient_patient_contact_arrays)
