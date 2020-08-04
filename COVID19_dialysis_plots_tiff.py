"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: May, 2020

Description: This script generates figures

Figures are saved in `dialysis/tiff/plots`
"""
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def countplot(infection_array):
    infection, count = np.unique(infection_array, return_counts=True)
    infection = infection.astype(int)
    count = count.astype(int)
    x_array = np.arange(1, np.max(infection)+1).astype(int)
    y_array = np.zeros((x_array.shape)).astype(int)
    for inf, c in zip(infection, count):
        y_array[inf-1] = c
    return x_array, y_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw cumulative incidence')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of HCPs')
    parser.add_argument('-s', '--scenario', type=int, default=0,
                        help= 'Infection source: scenario 0 = patient, scenario 1 = HCP')
    args = parser.parse_args()

    day = args.day
    s = args.scenario
    n_HCPs = 11
    n_patients = 40

    if s == 0:
        source = "patient"
    elif s == 1:
        source = "HCP"

    ############################################################################################################3
    # Read npzfile
    npzfile = np.load("dialysis/results/day{}/final_scenario{}.npz".format(day, s))
    B_n_inf_rec = npzfile["B_n_inf_rec"]
    B_transmission_route = npzfile["B_transmission_route"]
    B_population = npzfile["B_population"]
    B_R0 = npzfile["B_R0"]
    B_generation_time = npzfile["B_generation_time"]
    H0_n_inf_rec = npzfile["H0_n_inf_rec"]
    H0_transmission_route = npzfile["H0_transmission_route"]
    H0_population = npzfile["H0_population"]
    H0_R0 = npzfile["H0_R0"]
    H0_generation_time = npzfile["H0_generation_time"]
    H1_n_inf_rec = npzfile["H1_n_inf_rec"]
    H1_transmission_route = npzfile["H1_transmission_route"]
    H1_population = npzfile["H1_population"]
    H1_R0 = npzfile["H1_R0"]
    H1_generation_time = npzfile["H1_generation_time"]
    P2_n_inf_rec = npzfile["P2_n_inf_rec"]
    P2_transmission_route = npzfile["P2_transmission_route"]
    P2_population = npzfile["P2_population"]
    P2_R0 = npzfile["P2_R0"]
    P2_generation_time = npzfile["P2_generation_time"]
    H2P2_n_inf_rec = npzfile["H2P2_n_inf_rec"]
    H2P2_transmission_route = npzfile["H2P2_transmission_route"]
    H2P2_population = npzfile["H2P2_population"]
    H2P2_R0 = npzfile["H2P2_R0"]
    H2P2_generation_time = npzfile["H2P2_generation_time"]
    H2P2v2_n_inf_rec = npzfile["H2P2v2_n_inf_rec"]
    H2P2v2_transmission_route = npzfile["H2P2v2_transmission_route"]
    H2P2v2_population = npzfile["H2P2v2_population"]
    H2P2v2_R0 = npzfile["H2P2v2_R0"]
    H2P2v2_generation_time = npzfile["H2P2v2_generation_time"]
    N0_n_inf_rec = npzfile["N0_n_inf_rec"]
    N0_transmission_route = npzfile["N0_transmission_route"]
    N0_population = npzfile["N0_population"]
    N0_R0 = npzfile["N0_R0"]
    N0_generation_time = npzfile["N0_generation_time"]
    N1_n_inf_rec = npzfile["N1_n_inf_rec"]
    N1_transmission_route = npzfile["N1_transmission_route"]
    N1_population = npzfile["N1_population"]
    N1_R0 = npzfile["N1_R0"]
    N1_generation_time = npzfile["N1_generation_time"]
    H3P1_n_inf_rec = npzfile["H3P1_n_inf_rec"]
    H3P1_transmission_route = npzfile["H3P1_transmission_route"]
    H3P1_population = npzfile["H3P1_population"]
    H3P1_R0 = npzfile["H3P1_R0"]
    H3P1_generation_time = npzfile["H3P1_generation_time"]
    Bp_n_inf_rec = npzfile["Bp_n_inf_rec"]
    Bp_transmission_route = npzfile["Bp_transmission_route"]
    Bp_population = npzfile["Bp_population"]
    Bp_R0 = npzfile["Bp_R0"]
    Bp_generation_time = npzfile["Bp_generation_time"]
    Bpp_n_inf_rec = npzfile["Bpp_n_inf_rec"]
    Bpp_transmission_route = npzfile["Bpp_transmission_route"]
    Bpp_population = npzfile["Bpp_population"]
    Bpp_R0 = npzfile["Bpp_R0"]
    Bpp_generation_time = npzfile["Bpp_generation_time"]
    Bppp_n_inf_rec = npzfile["Bppp_n_inf_rec"]
    Bppp_transmission_route = npzfile["Bppp_transmission_route"]
    Bppp_population = npzfile["Bppp_population"]
    Bppp_R0 = npzfile["Bppp_R0"]
    Bppp_generation_time = npzfile["Bppp_generation_time"]
    npzfile.close()
    ############################################################################################################3
    # Shape of arrays (B*)
    # dim0: n_repeat
    # dim1: n_Dtype
    # dim2: n_cases
    ############################################################################################################3
    # Shape of arrays (H*)
    # dim0: n_repeat
    # dim1: n_Dtype
    # dim2: n_cases
    # dim3: QC_list
    ############################################################################################################3
    # >>> B_n_inf_rec.shape
    # (500, 4, 3, 3, 4, 30)
    # dim-3: [during incubation period, during symptomatic period, outside source (coming in infected)]
    # dim-2: [HCP_infected, patient_infected, HCP_recovered, patient_recovered]
    # dim-1: simulation period
    ############################################################################################################3
    # >>> B_transmission_route.shape
    # (500, 4, 3, 4, 30)
    # dim-2: [h->p, p->h, h->h, p->p]
    # dim-1: simulation period
    ############################################################################################################3
    # >>> B_population.shape
    # (500, 4, 3, 30)
    # dim-1: simulation period
    ############################################################################################################3
    # >>> B_generation_time.shape
    # (500, 4, 3)
    ############################################################################################################3
    # >>> B_R0.shape
    # (500, 4, 3)
    ############################################################################################################3

    xticklabels = ['M','T','W','Th','F','S','Su']*4+['M','T']
    Dtype_list = ["uni/uni(5%)", "uni/uni(35%)", "exp/exp(5%)", "exp/exp(35%)"]
    sus_array = np.load("dialysis/data/alpha_array.npy")
    # QC_list0 = ["QC:0.5", "QC:0.7"]
    QC_list0 = [r'$r_{VI}=0.5$', r'$r_{VI}=0.7$']
    simulation_period = B_n_inf_rec.shape[-1]
    n_repeat = B_n_inf_rec.shape[0]
    cum_infection_on = ["HCP", "Patient"]
    target_R0 = [2.0, 2.5, 3.0]

    ############################################################################################################3
    # Preprocess some of the result tables
    ############################################################################################################3
    B_H_infection = B_n_inf_rec[:,:,:,0,0,:] + B_n_inf_rec[:,:,:,1,0,:] + B_n_inf_rec[:,:,:,2,0,:] 
    B_P_infection = B_n_inf_rec[:,:,:,0,1,:] + B_n_inf_rec[:,:,:,1,1,:] + B_n_inf_rec[:,:,:,2,1,:] 
    B_T_infection = B_H_infection + B_P_infection
    B_T_cum_infection = np.cumsum(B_T_infection, axis=-1)
    B_T_cum_attack_rate = B_T_cum_infection / B_population

    H0_H_infection = H0_n_inf_rec[:,:,:,:,0,0,:] + H0_n_inf_rec[:,:,:,:,1,0,:] + H0_n_inf_rec[:,:,:,:,2,0,:] 
    H0_P_infection = H0_n_inf_rec[:,:,:,:,0,1,:] + H0_n_inf_rec[:,:,:,:,1,1,:] + H0_n_inf_rec[:,:,:,:,2,1,:] 
    H0_T_infection = H0_H_infection + H0_P_infection
    H0_T_cum_infection = np.cumsum(H0_T_infection, axis=-1)
    H0_T_cum_attack_rate = H0_T_cum_infection / H0_population

    H1_H_infection = H1_n_inf_rec[:,:,:,:,0,0,:] + H1_n_inf_rec[:,:,:,:,1,0,:] + H1_n_inf_rec[:,:,:,:,2,0,:] 
    H1_P_infection = H1_n_inf_rec[:,:,:,:,0,1,:] + H1_n_inf_rec[:,:,:,:,1,1,:] + H1_n_inf_rec[:,:,:,:,2,1,:] 
    H1_T_infection = H1_H_infection + H1_P_infection
    H1_T_cum_infection = np.cumsum(H1_T_infection, axis=-1)
    H1_T_cum_attack_rate = H1_T_cum_infection / H1_population

    P2_H_infection = P2_n_inf_rec[:,:,:,0,0,:] + P2_n_inf_rec[:,:,:,1,0,:] + P2_n_inf_rec[:,:,:,2,0,:] 
    P2_P_infection = P2_n_inf_rec[:,:,:,0,1,:] + P2_n_inf_rec[:,:,:,1,1,:] + P2_n_inf_rec[:,:,:,2,1,:] 
    P2_T_infection = P2_H_infection + P2_P_infection
    P2_T_cum_infection = np.cumsum(P2_T_infection, axis=-1)
    P2_T_cum_attack_rate = P2_T_cum_infection / P2_population

    H2P2_H_infection = H2P2_n_inf_rec[:,:,:,0,0,:] + H2P2_n_inf_rec[:,:,:,1,0,:] + H2P2_n_inf_rec[:,:,:,2,0,:] 
    H2P2_P_infection = H2P2_n_inf_rec[:,:,:,0,1,:] + H2P2_n_inf_rec[:,:,:,1,1,:] + H2P2_n_inf_rec[:,:,:,2,1,:] 
    H2P2_T_infection = H2P2_H_infection + H2P2_P_infection
    H2P2_T_cum_infection = np.cumsum(H2P2_T_infection, axis=-1)
    H2P2_T_cum_attack_rate = H2P2_T_cum_infection / H2P2_population

    H2P2v2_H_infection = H2P2v2_n_inf_rec[:,:,:,0,0,:] + H2P2v2_n_inf_rec[:,:,:,1,0,:] + H2P2v2_n_inf_rec[:,:,:,2,0,:] 
    H2P2v2_P_infection = H2P2v2_n_inf_rec[:,:,:,0,1,:] + H2P2v2_n_inf_rec[:,:,:,1,1,:] + H2P2v2_n_inf_rec[:,:,:,2,1,:] 
    H2P2v2_T_infection = H2P2v2_H_infection + H2P2v2_P_infection
    H2P2v2_T_cum_infection = np.cumsum(H2P2v2_T_infection, axis=-1)
    H2P2v2_T_cum_attack_rate = H2P2v2_T_cum_infection / H2P2v2_population

    N0_H_infection = N0_n_inf_rec[:,:,:,:,0,0,:] + N0_n_inf_rec[:,:,:,:,1,0,:] + N0_n_inf_rec[:,:,:,:,2,0,:] 
    N0_P_infection = N0_n_inf_rec[:,:,:,:,0,1,:] + N0_n_inf_rec[:,:,:,:,1,1,:] + N0_n_inf_rec[:,:,:,:,2,1,:] 
    N0_T_infection = N0_H_infection + N0_P_infection
    N0_T_cum_infection = np.cumsum(N0_T_infection, axis=-1)
    N0_T_cum_attack_rate = N0_T_cum_infection / N0_population

    N1_H_infection = N1_n_inf_rec[:,:,:,:,0,0,:] + N1_n_inf_rec[:,:,:,:,1,0,:] + N1_n_inf_rec[:,:,:,:,2,0,:] 
    N1_P_infection = N1_n_inf_rec[:,:,:,:,0,1,:] + N1_n_inf_rec[:,:,:,:,1,1,:] + N1_n_inf_rec[:,:,:,:,2,1,:] 
    N1_T_infection = N1_H_infection + N1_P_infection
    N1_T_cum_infection = np.cumsum(N1_T_infection, axis=-1)
    N1_T_cum_attack_rate = N1_T_cum_infection / N1_population
    
    H3P1_H_infection = H3P1_n_inf_rec[:,:,:,:,0,0,:] + H3P1_n_inf_rec[:,:,:,:,1,0,:] + H3P1_n_inf_rec[:,:,:,:,2,0,:] 
    H3P1_P_infection = H3P1_n_inf_rec[:,:,:,:,0,1,:] + H3P1_n_inf_rec[:,:,:,:,1,1,:] + H3P1_n_inf_rec[:,:,:,:,2,1,:] 
    H3P1_T_infection = H3P1_H_infection + H3P1_P_infection
    H3P1_T_cum_infection = np.cumsum(H3P1_T_infection, axis=-1)
    H3P1_T_cum_attack_rate = H3P1_T_cum_infection / H3P1_population

    Bp_H_infection = Bp_n_inf_rec[:,:,:,0,0,:] + Bp_n_inf_rec[:,:,:,1,0,:] + Bp_n_inf_rec[:,:,:,2,0,:] 
    Bp_P_infection = Bp_n_inf_rec[:,:,:,0,1,:] + Bp_n_inf_rec[:,:,:,1,1,:] + Bp_n_inf_rec[:,:,:,2,1,:] 
    Bp_T_infection = Bp_H_infection + Bp_P_infection
    Bp_T_cum_infection = np.cumsum(Bp_T_infection, axis=-1)
    Bp_T_cum_attack_rate = Bp_T_cum_infection / Bp_population

    Bpp_H_infection = Bpp_n_inf_rec[:,:,:,:,0,0,:] + Bpp_n_inf_rec[:,:,:,:,1,0,:] + Bpp_n_inf_rec[:,:,:,:,2,0,:] 
    Bpp_P_infection = Bpp_n_inf_rec[:,:,:,:,0,1,:] + Bpp_n_inf_rec[:,:,:,:,1,1,:] + Bpp_n_inf_rec[:,:,:,:,2,1,:] 
    Bpp_T_infection = Bpp_H_infection + Bpp_P_infection
    Bpp_T_cum_infection = np.cumsum(Bpp_T_infection, axis=-1)
    Bpp_T_cum_attack_rate = Bpp_T_cum_infection / Bpp_population

    Bppp_H_infection = Bppp_n_inf_rec[:,:,:,:,0,0,:] + Bppp_n_inf_rec[:,:,:,:,1,0,:] + Bppp_n_inf_rec[:,:,:,:,2,0,:] 
    Bppp_P_infection = Bppp_n_inf_rec[:,:,:,:,0,1,:] + Bppp_n_inf_rec[:,:,:,:,1,1,:] + Bppp_n_inf_rec[:,:,:,:,2,1,:] 
    Bppp_T_infection = Bppp_H_infection + Bppp_P_infection
    Bppp_T_cum_infection = np.cumsum(Bppp_T_infection, axis=-1)
    Bppp_T_cum_attack_rate = Bppp_T_cum_infection / Bppp_population

    H0_replaced_HCPs = np.mean(H0_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    H1_replaced_HCPs = np.mean(H1_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    H3P1_replaced_HCPs = np.mean(H3P1_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    Bpp_replaced_HCPs = np.mean(Bpp_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    Bppp_replaced_HCPs = np.mean(Bppp_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)

    ############################################################################################################3
    # Figures - Infection counts at the end of the simulation
    ############################################################################################################3
    # for i, D in enumerate(Dtype_list):
        # if i in [0,1,3]:
            # continue
        # for j, alpha in enumerate(sus_array[i]):
            # B_T_max = B_T_infection.sum(axis=-1)[:,i,j].max()
            # N0_T_max = N0_T_infection.sum(axis=-1)[:,i,j].max()
            # N0_rr100_T_max = N0_rr100_T_infection.sum(axis=-1)[:,i,j].max()
            # max_infection = max(B_T_max, N0_T_max, N0_rr100_T_max)

            # x_array = np.arange(1, max_infection + 1).astype(int)
            # y_array = np.zeros((x_array.shape)).astype(int)

            # _, y_temp = countplot(B_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p1,=plt.plot(x_array, y_array, label="Baseline", color="C0", zorder=4)

            # _, y_temp = countplot(N0_T_infection.sum(axis=-1)[:,i,j,0])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p2,=plt.plot(x_array, y_array, label="", color="C1", zorder=5)

            # _, y_temp = countplot(N0_T_infection.sum(axis=-1)[:,i,j,1])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p3,=plt.plot(x_array, y_array, label="", color="C2", zorder=6)

            # _, y_temp = countplot(N0_T_infection.sum(axis=-1)[:,i,j,2])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p4,=plt.plot(x_array, y_array, label="", color="C3", zorder=7)

            # _, y_temp = countplot(N0_rr100_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p5,=plt.plot(x_array, y_array, label="", color="C4", zorder=8)

            # # R0 = np.mean(B_R0, axis=0)[i,j]
            # R0 = target_R0[j]
            # plt.title("Total Infection Counts on 500 Repetitions\nSource: {}, D: {}, R0: {}".format(source, D, R0))
            # plt.xlabel("Number of infections for each simulation")
            # plt.legend(handles=[p1,p2,p3,p4,p5], loc='best')
            # plt.savefig("dialysis/tiff/plots/B_N0_countplot_D{}_R0{}_scenario{}.tif".format(i, R0, s))
            # plt.close()

    # for i, D in enumerate(Dtype_list):
        # if i in [0,1]:
            # continue
        # for j, alpha in enumerate(sus_array[i]):
            # B_T_max = B_T_infection.sum(axis=-1)[:,i,j].max()
            # B2p_T_max = B2p_T_infection.sum(axis=-1)[:,i,j].max()
            # B2pp_T_max = B2pp_T_infection.sum(axis=-1)[:,i,j].max()
            # B2ppH1_T_max = B2ppH1_T_infection.sum(axis=-1)[:,i,j].max()
            # H2P2_T_max = H2P2_T_infection.sum(axis=-1)[:,i,j].max()
            # H12P2_T_max = H12P2_T_infection.sum(axis=-1)[:,i,j].max()
            # max_infection = max(B_T_max, B2p_T_max, B2pp_T_max, B2ppH1_T_max)

            # x_array = np.arange(1, max_infection + 1).astype(int)
            # y_array = np.zeros((x_array.shape)).astype(int)

            # _, y_temp = countplot(B_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p1,=plt.plot(x_array, y_array, label="Baseline", color="C0", zorder=4)

            # _, y_temp = countplot(B2p_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p2,=plt.plot(x_array, y_array, label="Baseline+", color="C1", zorder=5)

            # _, y_temp = countplot(B2pp_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p3,=plt.plot(x_array, y_array, label="Baseline++ (HCP:+{:.1f})".format(B2pp_replaced_HCPs[i,j]), color="C2", zorder=6)

            # _, y_temp = countplot(B2ppH1_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p4,=plt.plot(x_array, y_array, label="AS Baseline++ (HCP:+{:.1f})".format(B2ppH1_replaced_HCPs[i,j]), color="C3", zorder=7)

            # _, y_temp = countplot(H2P2_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p5,=plt.plot(x_array, y_array, label="M P:surgical, H:N95", color="C4", zorder=8)

            # _, y_temp = countplot(H12P2_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p6,=plt.plot(x_array, y_array, label="AS M P:surgical, H:N95 (HCP:+{:.1f})".format(H12P2_replaced_HCPs[i,j,0]), color="C5", zorder=9)

            # # R0 = np.mean(B_R0, axis=0)[i,j]
            # R0 = target_R0[j]
            # # plt.title("Total Infection Counts on 500 Repetitions\nSource: {}, D: {}, R0: {}".format(source, D, R0))
            # plt.title("Simulation Counts by Total Infection over 500 Repetitions \nSource: {}, D: {}, R0: {}".format(source, D, R0))
            # plt.xlabel("Number of infections for each simulation")
            # plt.ylabel("Simulation count")
            # plt.legend(handles=[p1,p2,p3,p4,p5,p6], loc='best')
            # plt.savefig("dialysis/tiff/plots/countplot_D{}_R0{}_scenario{}.tif".format(i, R0, s))
            # plt.close()

    #######################################################################
    # Count plot for the paper
    #######################################################################
    # for i, D in enumerate(Dtype_list):
        # if i in [0,1]:
            # continue
        # for j, alpha in enumerate(sus_array[i]):
            # B_T_max = B_T_infection.sum(axis=-1)[:,i,j].max()
            # Bp_T_max = Bp_T_infection.sum(axis=-1)[:,i,j].max()
            # # early HCP replacement. k=1
            # Bpp_T_max = Bpp_T_infection[:,:,:,0,:].sum(axis=-1)[:,i,j].max()
            # Bppp_T_max = Bppp_T_infection[:,:,:,0,:].sum(axis=-1)[:,i,j].max()
            # # BppH1_T_max = BppH1_T_infection[:,:,:,2,:].sum(axis=-1)[:,i,j].max()
            # # max_infection = max(B_T_max, Bp_T_max, Bpp_T_max, BppH1_T_max)
            # max_infection = max(B_T_max, Bp_T_max, Bpp_T_max)

            # # x_array = np.arange(1, max_infection + 1).astype(int)
            # x_array = np.arange(1, max_infection + 1).astype(int) / max_infection
            # y_array = np.zeros((x_array.shape)).astype(int)

            # _, y_temp = countplot(B_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p1,=plt.plot(x_array, y_array, label="Baseline", color="C0", zorder=4)

            # _, y_temp = countplot(Bp_T_infection.sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p2,=plt.plot(x_array, y_array, label="Baseline+", color="C1", zorder=5)

            # _, y_temp = countplot(Bpp_T_infection[:,:,:,0,:].sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p3,=plt.plot(x_array, y_array, label="Baseline++ (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,0]), color="C2", zorder=6)

            # _, y_temp = countplot(Bppp_T_infection[:,:,:,0,:].sum(axis=-1)[:,i,j])
            # y_array = np.zeros((x_array.shape)).astype(int)
            # y_array[:y_temp.shape[0]] = y_temp
            # p4,=plt.plot(x_array, y_array, label="Baseline+++ (HCP:+{:.1f})".format(Bppp_replaced_HCPs[i,j,0]), color="C3", zorder=7)

            # # _, y_temp = countplot(BppH1_T_infection[:,:,:,2,:].sum(axis=-1)[:,i,j])
            # # y_array = np.zeros((x_array.shape)).astype(int)
            # # y_array[:y_temp.shape[0]] = y_temp
            # # p4,=plt.plot(x_array, y_array, label="AS Baseline++ (HCP:+{:.1f})".format(BppH1_replaced_HCPs[i,j,2]), color="C3", zorder=7)

            # # R0 = np.mean(B_R0, axis=0)[i,j]
            # R0 = target_R0[j]
            # # plt.title("Total Infection Counts on 500 Repetitions\nSource: {}, D: {}, R0: {}".format(source, D, R0))
            # plt.title("Simulation counts by attack rate over 500 Repetitions \nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            # # plt.xlabel("Number of infections for each simulation")
            # plt.xlabel("Attack rate")
            # plt.ylabel("Simulation count")
            # plt.legend(handles=[p1,p2,p3,p4], loc='best')
            # # plt.legend(handles=[p1,p2,p3,p4], loc='best')
            # plt.savefig("dialysis/tiff/plots/day{}/countplot_D{}_R0{}_scenario{}.tif".format(day, i, R0, s), dpi=300)
            # plt.close()

    # Similar to countplot, but ratio on x-axis
    for i, D in enumerate(Dtype_list):
        if i in [0,1]:
            continue
        for j, alpha in enumerate(sus_array[i]):
            B_ar = B_T_cum_attack_rate[:,i,j,-1]
            Bp_ar = Bp_T_cum_attack_rate[:,i,j,-1]
            Bpp_ar = Bpp_T_cum_attack_rate[:,i,j,0,-1]
            Bppp_ar = Bppp_T_cum_attack_rate[:,i,j,0,-1]

            n_bins = 20
            bins = np.linspace(0,1,n_bins)
            # plt.hist(B_attack_rates, bins, alpha=0.5, label="Baseline")
            # plt.hist(Bp_attack_rates, bins, alpha=0.5, label="Baseline+")
            # plt.hist(Bpp_attack_rates, bins, alpha=0.5, label="Baseline++ (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,0]))
            # plt.hist(Bppp_attack_rates, bins, alpha=0.5, label="Baseline+++ (HCP:+{:.1f})".format(Bppp_replaced_HCPs[i,j,0]))
            label_list = ["Baseline", "Baseline+", "Baseline++ (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,0]), "Baseline+++ (HCP:+{:.1f})".format(Bppp_replaced_HCPs[i,j,0])]
            plt.hist([B_ar, Bp_ar, Bpp_ar, Bppp_ar], bins, alpha=0.5, label=label_list)

            R0 = target_R0[j]
            plt.title("Simulation counts by attack rate over 500 Repetitions \nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            plt.xlabel("Attack rate")
            plt.ylabel("Simulation count")
            plt.ylim(0, 500)
            plt.legend(loc="best")
            plt.grid(which='major', axis='both',linestyle=':')
            # plt.grid(which='minor', axis='x',linestyle=':')
            # plt.set_yticks(range(n_bins)/n_bins, minor=True)
            # plt.yaxis.grid(True, which='minor')
            # plt.legend(handles=[p1,p2,p3,p4], loc='best')
            plt.savefig("dialysis/tiff/plots/day{}/histogram_D{}_R0{}_scenario{}.tif".format(day, i, R0, s), dpi=300)
            # plt.savefig("dialysis/tiff/plots/day{}/histogram_D{}_R0{}_scenario{}.tif".format(day, i, R0, s))
            plt.close()



    ############################################################################################################3
    # Figures - Social distancing
    ############################################################################################################3
    for i, D in enumerate(Dtype_list):
        if i in [0,1]:
            continue
        for j, alpha in enumerate(sus_array[i]):
            # Cum Patient Infection
            p1,=plt.plot(np.arange(1, simulation_period+1), np.mean(B_T_cum_attack_rate, axis=0)[i,j], label="Baseline", color="C0", zorder=4)
            p2,=plt.plot(np.arange(1, simulation_period+1), np.mean(N0_T_cum_attack_rate, axis=0)[i,j,0], label=r'$r_{SD}=0.25$', color="C1", zorder=5)
            p3,=plt.plot(np.arange(1, simulation_period+1), np.mean(N0_T_cum_attack_rate, axis=0)[i,j,1], label=r'$r_{SD}=0.50$', color="C2", zorder=6)
            p4,=plt.plot(np.arange(1, simulation_period+1), np.mean(N0_T_cum_attack_rate, axis=0)[i,j,2], label=r'$r_{SD}=0.75$', color="C3", zorder=7)
            p5,=plt.plot(np.arange(1, simulation_period+1), np.mean(N0_T_cum_attack_rate, axis=0)[i,j,3], label=r'$r_{SD}=1.00$', color="C4", zorder=8)

            # R0 = np.mean(B_R0, axis=0)[i,j]
            R0 = target_R0[j]
            plt.title("Mean cumulative attack rate (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            plt.xlabel("Time (in days)")
            plt.ylabel("Attack rate")
            plt.ylim(0,1)
            plt.legend(handles=[p1,p2,p3,p4,p5], loc='lower right')
            plt.xticks(range(1, 31), xticklabels)
            plt.grid(zorder=0)
            plt.savefig("dialysis/tiff/plots/day{}/B_N0_D{}_R0{}_scenario{}.tif".format(day, i, R0, s), dpi=300)
            plt.close()

    # ############################################################################################################3
    # # Figures - Move chairs apart
    # ############################################################################################################3
    for i, D in enumerate(Dtype_list):
        if i in [0,1]:
            continue
        for j, alpha in enumerate(sus_array[i]):
            # Cum Patient Infection
            p1,=plt.plot(np.arange(1, simulation_period+1), np.mean(B_T_cum_attack_rate, axis=0)[i,j], label="Baseline", color="C0", zorder=4)
            p2,=plt.plot(np.arange(1, simulation_period+1), np.mean(N1_T_cum_attack_rate, axis=0)[i,j,0], label=r'$r_{PS}=0.25$', color="C1", zorder=5)
            p3,=plt.plot(np.arange(1, simulation_period+1), np.mean(N1_T_cum_attack_rate, axis=0)[i,j,1], label=r'$r_{PS}=0.50$', color="C2", zorder=6)
            p4,=plt.plot(np.arange(1, simulation_period+1), np.mean(N1_T_cum_attack_rate, axis=0)[i,j,2], label=r'$r_{PS}=0.75$', color="C3", zorder=7)
            p5,=plt.plot(np.arange(1, simulation_period+1), np.mean(N1_T_cum_attack_rate, axis=0)[i,j,3], label=r'$r_{PS}=1.00$', color="C4", zorder=8)

            # R0 = np.mean(B_R0, axis=0)[i,j]
            R0 = target_R0[j]
            plt.title("Mean cumulative attack rate (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            plt.xlabel("Time (in days)")
            plt.ylabel("Attack rate")
            plt.ylim(0,1)
            plt.legend(handles=[p1,p2,p3,p4,p5], loc='lower right')
            plt.xticks(range(1, 31), xticklabels)
            plt.grid(zorder=0)
            plt.savefig("dialysis/tiff/plots/day{}/B_N1_D{}_R0{}_scenario{}.tif".format(day, i, R0, s), dpi=300)
            plt.close()

    # ############################################################################################################3
    # # Figures - B vs H0 vs H1
    # ############################################################################################################3
    for i, D in enumerate(Dtype_list):
        if i in [0,1]:
            continue
        for j, alpha in enumerate(sus_array[i]):
            # Cum Patient Infection
            p1,=plt.plot(np.arange(1, simulation_period+1), np.mean(B_T_cum_attack_rate, axis=0)[i,j], label="Baseline", color="C0", zorder=4)
            p2,=plt.plot(np.arange(1, simulation_period+1), np.mean(H0_T_cum_attack_rate, axis=0)[i,j,0], label="VI {} (HCP:+{:.1f})".format(QC_list0[0], H0_replaced_HCPs[i,j,0]), color="C1", zorder=5)
            p3,=plt.plot(np.arange(1, simulation_period+1), np.mean(H0_T_cum_attack_rate, axis=0)[i,j,1], label="VI {} (HCP:+{:.1f})".format(QC_list0[1], H0_replaced_HCPs[i,j,1]), color="C2", zorder=6)
            p4,=plt.plot(np.arange(1, simulation_period+1), np.mean(H1_T_cum_attack_rate, axis=0)[i,j,0], label="AS&CI (HCP:+{:.1f})".format(H1_replaced_HCPs[i,j,0]), color="C3", zorder=7)

            # R0 = np.mean(B_R0, axis=0)[i,j]
            R0 = target_R0[j]
            plt.title("Mean cumulative attack rate (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            plt.xlabel("Time (in days)")
            plt.ylabel("Attack rate")
            plt.ylim(0,1)
            plt.legend(handles=[p1,p2,p3,p4], loc='best')
            plt.xticks(range(1, 31), xticklabels)
            plt.grid(zorder=0)
            plt.savefig("dialysis/tiff/plots/day{}/B_H0_H1_D{}_R0{}_scenario{}.tif".format(day, i, R0, s), dpi=300)
            plt.close()

    # for i, D in enumerate(Dtype_list):
        # if i in [0,1]:
            # continue
        # for j, alpha in enumerate(sus_array[i]):
            # # Cum Patient Infection
            # p1,=plt.plot(np.arange(1, simulation_period+1), np.median(B_T_cum_attack_rate, axis=0)[i,j], label="Baseline", color="C0", zorder=4)
            # p2,=plt.plot(np.arange(1, simulation_period+1), np.median(H0_T_cum_attack_rate, axis=0)[i,j,0], label="VI {} (HCP:+{:.1f})".format(QC_list0[0], H0_replaced_HCPs[i,j,0]), color="C1", zorder=5)
            # p3,=plt.plot(np.arange(1, simulation_period+1), np.median(H0_T_cum_attack_rate, axis=0)[i,j,1], label="VI {} (HCP:+{:.1f})".format(QC_list0[1], H0_replaced_HCPs[i,j,1]), color="C2", zorder=6)
            # p4,=plt.plot(np.arange(1, simulation_period+1), np.median(H1_T_cum_attack_rate, axis=0)[i,j,0], label="AS&CI (HCP:+{:.1f})".format(H1_replaced_HCPs[i,j,0]), color="C3", zorder=7)

            # # R0 = np.median(B_R0, axis=0)[i,j]
            # R0 = target_R0[j]
            # plt.title("Median cumulative attack rate (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            # plt.xlabel("Time (in days)")
            # plt.ylabel("Attack rate")
            # plt.ylim(0,1)
            # plt.legend(handles=[p1,p2,p3,p4], loc='best')
            # plt.xticks(range(1, 31), xticklabels)
            # plt.grid(zorder=0)
            # plt.savefig("dialysis/tiff/plots/day{}/median_B_H0_H1_D{}_R0{}_scenario{}.tif".format(day, i, R0, s))
            # plt.close()

    # ############################################################################################################3
    # # Figures - Mask interventions
    # ############################################################################################################3
    for i, D in enumerate(Dtype_list):
        if i in [0,1]:
            continue
        for j, alpha in enumerate(sus_array[i]):
            # Cum Patient Infection
            p1,=plt.plot(np.arange(1, simulation_period+1), np.mean(B_T_cum_attack_rate, axis=0)[i,j], label="Baseline", color="C0", zorder=4)
            p2,=plt.plot(np.arange(1, simulation_period+1), np.mean(P2_T_cum_attack_rate, axis=0)[i,j], label="Patient:surgical, HCP:x", color="C1", zorder=5)
            p3,=plt.plot(np.arange(1, simulation_period+1), np.mean(H2P2v2_T_cum_attack_rate, axis=0)[i,j], label="Patient:surgical, HCP:surgical", color="C2", zorder=6)
            p4,=plt.plot(np.arange(1, simulation_period+1), np.mean(H2P2_T_cum_attack_rate, axis=0)[i,j], label="Patient:surgical, HCP:N95", color="C3", zorder=7)

            # R0 = np.mean(B_R0, axis=0)[i,j]
            R0 = target_R0[j]
            plt.title("Mean cumulative attack rate (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            plt.xlabel("Time (in days)")
            plt.ylabel("Attack rate")
            plt.ylim(0,1)
            plt.legend(handles=[p1,p2,p3,p4], loc='best')
            plt.xticks(range(1, 31), xticklabels)
            plt.grid(zorder=0)
            plt.savefig("dialysis/tiff/plots/day{}/B_P2_H2P2_D{}_R0{}_scenario{}.tif".format(day,i, R0, s), dpi=300)
            plt.close()

    # for i, D in enumerate(Dtype_list):
        # if i in [0,1]:
            # continue
        # for j, alpha in enumerate(sus_array[i]):
            # # Cum Patient Infection
            # p1,=plt.plot(np.arange(1, simulation_period+1), np.median(B_T_cum_attack_rate, axis=0)[i,j], label="Baseline", color="C0", zorder=4)
            # p2,=plt.plot(np.arange(1, simulation_period+1), np.median(P2_T_cum_attack_rate, axis=0)[i,j], label="M P:surgical, H:x", color="C1", zorder=5)
            # p3,=plt.plot(np.arange(1, simulation_period+1), np.median(H2P2v2_T_cum_attack_rate, axis=0)[i,j], label="M P:surgical, H:surgical", color="C2", zorder=6)
            # p4,=plt.plot(np.arange(1, simulation_period+1), np.median(H2P2_T_cum_attack_rate, axis=0)[i,j], label="M P:surgical, H:N95", color="C3", zorder=7)

            # # R0 = np.median(B_R0, axis=0)[i,j]
            # R0 = target_R0[j]
            # plt.title("Median cumulative attack rate (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            # plt.xlabel("Time (in days)")
            # plt.ylabel("Attack rate")
            # plt.ylim(0,1)
            # plt.legend(handles=[p1,p2,p3,p4], loc='best')
            # plt.xticks(range(1, 31), xticklabels)
            # plt.grid(zorder=0)
            # plt.savefig("dialysis/tiff/plots/day{}/median_B_P2_H2P2_D{}_R0{}_scenario{}.tif".format(day,i, R0, s))
            # plt.close()

    # ############################################################################################################3
    # # Figures - Patient isolation, early HCP replacement
    # ############################################################################################################3
    for i, D in enumerate(Dtype_list):
        if i in [0,1]:
            continue
        for j, alpha in enumerate(sus_array[i]):
            # Cum Patient Infection
            p1,=plt.plot(np.arange(1, simulation_period+1), np.mean(B_T_cum_attack_rate, axis=0)[i,j], label="Baseline", color="C0", zorder=4)
            p2,=plt.plot(np.arange(1, simulation_period+1), np.mean(H3P1_T_cum_attack_rate, axis=0)[i,j,0], label="k=1 (HCP:+{:.1f})".format(H3P1_replaced_HCPs[i,j,0]), color="C1", zorder=5)
            p3,=plt.plot(np.arange(1, simulation_period+1), np.mean(H3P1_T_cum_attack_rate, axis=0)[i,j,1], label="k=2 (HCP:+{:.1f})".format(H3P1_replaced_HCPs[i,j,1]), color="C2", zorder=6)
            p4,=plt.plot(np.arange(1, simulation_period+1), np.mean(H3P1_T_cum_attack_rate, axis=0)[i,j,2], label="k=3 (HCP:+{:.1f})".format(H3P1_replaced_HCPs[i,j,2]), color="C3", zorder=7)
            p5,=plt.plot(np.arange(1, simulation_period+1), np.mean(H3P1_T_cum_attack_rate, axis=0)[i,j,3], label="k=4 (HCP:+{:.1f})".format(H3P1_replaced_HCPs[i,j,3]), color="C4", zorder=8)
            p6,=plt.plot(np.arange(1, simulation_period+1), np.mean(H3P1_T_cum_attack_rate, axis=0)[i,j,4], label="k=5 (HCP:+{:.1f})".format(H3P1_replaced_HCPs[i,j,4]), color="C5", zorder=9)

            # R0 = np.mean(B_R0, axis=0)[i,j]
            R0 = target_R0[j]
            plt.title("Mean cumulative attack rate (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            plt.xlabel("Time (in days)")
            plt.ylabel("Attack rate")
            plt.ylim(0,1)
            plt.legend(handles=[p1,p2,p3,p4,p5,p6], loc='best')
            plt.xticks(range(1, 31), xticklabels)
            plt.grid(zorder=0)
            plt.savefig("dialysis/tiff/plots/day{}/B_H3P1_D{}_R0{}_scenario{}.tif".format(day,i, R0, s), dpi=300)
            plt.close()

    # ############################################################################################################3
    # # Figures - B vs Bp vs Bpp vs Bppp
    # ############################################################################################################3
    for i, D in enumerate(Dtype_list):
        if i in [0,1]:
            continue
        for j, alpha in enumerate(sus_array[i]):
            # Cum Patient Infection
            p1,=plt.plot(np.arange(1, simulation_period+1), np.mean(B_T_cum_attack_rate, axis=0)[i,j], label="Baseline", color="C0", zorder=4)
            p2,=plt.plot(np.arange(1, simulation_period+1), np.mean(Bp_T_cum_attack_rate, axis=0)[i,j], label="Baseline+", color="C1", zorder=5)
            p3,=plt.plot(np.arange(1, simulation_period+1), np.mean(Bpp_T_cum_attack_rate, axis=0)[i,j,0], label="Baseline++ (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,0]), color="C2", zorder=6)
            p4,=plt.plot(np.arange(1, simulation_period+1), np.mean(Bppp_T_cum_attack_rate, axis=0)[i,j,0], label="Baseline+++ (HCP:+{:.1f})".format(Bppp_replaced_HCPs[i,j,0]), color="C3", zorder=7)
            # p4,=plt.plot(np.arange(1, simulation_period+1), np.mean(Bpp_T_cum_attack_rate, axis=0)[i,j,1], label="Baseline++, k=2 (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,1]), color="C3", zorder=7)
            # p5,=plt.plot(np.arange(1, simulation_period+1), np.mean(Bpp_T_cum_attack_rate, axis=0)[i,j,2], label="Baseline++, k=3 (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,2]), color="C4", zorder=8)
            # p6,=plt.plot(np.arange(1, simulation_period+1), np.mean(Bpp_T_cum_attack_rate, axis=0)[i,j,3], label="Baseline++, k=4 (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,3]), color="C5", zorder=9)
            # p7,=plt.plot(np.arange(1, simulation_period+1), np.mean(Bpp_T_cum_attack_rate, axis=0)[i,j,4], label="Baseline++, k=5 (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,4]), color="C6", zorder=10)
            # p4,=plt.plot(np.arange(1, simulation_period+1), np.mean(BppH1_T_cum_attack_rate, axis=0)[i,j,0], label="AS Baseline++ (HCP:+{:.1f})".format(BppH1_replaced_HCPs[i,j,0]), color="C3", zorder=7)

            # R0 = np.mean(B_R0, axis=0)[i,j]
            R0 = target_R0[j]
            plt.title("Mean cumulative attack rate (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            plt.xlabel("Time (in days)")
            plt.ylabel("Attack rate")
            plt.ylim(0,1)
            plt.legend(handles=[p1,p2,p3,p4], loc='best')
            # plt.legend(handles=[p1,p2,p3,p4,p5,p6,p7], loc='best')
            plt.xticks(range(1, 31), xticklabels)
            plt.grid(zorder=0)
            plt.savefig("dialysis/tiff/plots/day{}/B_Bp_Bpp_Bppp_D{}_R0{}_scenario{}.tif".format(day,i, R0, s), dpi=300)
            plt.close()

    # ############################################################################################################3
    # # MEDIAN Figures - B vs Bp vs Bpp vs Bppp
    # ############################################################################################################3
    # for i, D in enumerate(Dtype_list):
        # if i in [0,1]:
            # continue
        # for j, alpha in enumerate(sus_array[i]):
            # # Cum Patient Infection
            # p1,=plt.plot(np.arange(1, simulation_period+1), np.median(B_T_cum_attack_rate, axis=0)[i,j], label="Baseline", color="C0", zorder=4)
            # p2,=plt.plot(np.arange(1, simulation_period+1), np.median(Bp_T_cum_attack_rate, axis=0)[i,j], label="Baseline+", color="C1", zorder=5)
            # p3,=plt.plot(np.arange(1, simulation_period+1), np.median(Bpp_T_cum_attack_rate, axis=0)[i,j,0], label="Baseline++ (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,0]), color="C2", zorder=6)
            # p4,=plt.plot(np.arange(1, simulation_period+1), np.median(Bppp_T_cum_attack_rate, axis=0)[i,j,0], label="Baseline+++ (HCP:+{:.1f})".format(Bppp_replaced_HCPs[i,j,0]), color="C3", zorder=7)
            # # p4,=plt.plot(np.arange(1, simulation_period+1), np.median(Bpp_T_cum_attack_rate, axis=0)[i,j,1], label="Baseline++, k=2 (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,1]), color="C3", zorder=7)
            # # p5,=plt.plot(np.arange(1, simulation_period+1), np.median(Bpp_T_cum_attack_rate, axis=0)[i,j,2], label="Baseline++, k=3 (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,2]), color="C4", zorder=8)
            # # p6,=plt.plot(np.arange(1, simulation_period+1), np.median(Bpp_T_cum_attack_rate, axis=0)[i,j,3], label="Baseline++, k=4 (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,3]), color="C5", zorder=9)
            # # p7,=plt.plot(np.arange(1, simulation_period+1), np.median(Bpp_T_cum_attack_rate, axis=0)[i,j,4], label="Baseline++, k=5 (HCP:+{:.1f})".format(Bpp_replaced_HCPs[i,j,4]), color="C6", zorder=10)
            # # p4,=plt.plot(np.arange(1, simulation_period+1), np.median(BppH1_T_cum_attack_rate, axis=0)[i,j,0], label="AS Baseline++ (HCP:+{:.1f})".format(BppH1_replaced_HCPs[i,j,0]), color="C3", zorder=7)

            # # R0 = np.median(B_R0, axis=0)[i,j]
            # R0 = target_R0[j]
            # plt.title("Median cumulative attack rate (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            # plt.xlabel("Time (in days)")
            # plt.ylabel("Attack rate")
            # plt.ylim(0,1)
            # plt.legend(handles=[p1,p2,p3,p4], loc='best')
            # # plt.legend(handles=[p1,p2,p3,p4,p5,p6,p7], loc='best')
            # plt.xticks(range(1, 31), xticklabels)
            # plt.grid(zorder=0)
            # plt.savefig("dialysis/tiff/plots/day{}/median_B_Bp_Bpp_Bppp_D{}_R0{}_scenario{}.tif".format(day,i, R0, s))
            # plt.close()

    ###########################################################################################################3
    # Figures - Transmission route over 30 days (B)
    ###########################################################################################################3
    for i, D in enumerate(Dtype_list):
        if i in [0,1]:
            continue
        for j, alpha in enumerate(sus_array[i]):
            # Cum Transmission route
            p1,=plt.plot(np.arange(1, simulation_period+1), np.cumsum(np.mean(B_transmission_route[:,:,:,0,:], axis=0)[i,j,:]), label="HCP->Patient", color="C0", zorder=4)
            p2,=plt.plot(np.arange(1, simulation_period+1), np.cumsum(np.mean(B_transmission_route[:,:,:,1,:], axis=0)[i,j,:]), label="Patient->HCP", color="C1", zorder=5)
            p3,=plt.plot(np.arange(1, simulation_period+1), np.cumsum(np.mean(B_transmission_route[:,:,:,2,:], axis=0)[i,j,:]), label="HCP->HCP", color="C2", zorder=6)
            p4,=plt.plot(np.arange(1, simulation_period+1), np.cumsum(np.mean(B_transmission_route[:,:,:,3,:], axis=0)[i,j,:]), label="Patient->Patient", color="C3", zorder=7)

            # R0 = np.mean(B_R0, axis=0)[i,j]
            R0 = target_R0[j]
            plt.title("Cumulative transmission route on Baseline simulation (30 days)\nSource: {}, shedding: {}, R0: {}".format(source, D, R0))
            plt.xlabel("Time (in days)")
            plt.ylabel("Infection count")
            plt.ylim(0, 30)
            plt.legend(handles=[p1,p4,p3,p2], loc='best')
            plt.xticks(range(1, 31), xticklabels)
            plt.grid(zorder=0)
            plt.savefig("dialysis/tiff/plots/day{}/transmission_route_D{}_R0{}_scenario{}.tif".format(day, i, R0, s), dpi=300)
            plt.close()

