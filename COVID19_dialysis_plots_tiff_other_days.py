"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: July, 2020

Description: This script generates figures on other long days

Figures are saved in `dialysis/tiff/plots/day{}` 
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
    contact_distance=6

    # Load Patient arrays
    npzfile = np.load("dialysis/contact_data/patient_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hpc_original = npzfile["hcw_patient_contact_arrays"]
    npzfile.close()

    simulation_period = hpc_original.shape[0]
    n_HCPs = hpc_original.shape[1]
    n_patients = hpc_original.shape[2]

    if s == 0:
        source = "patient"
    elif s == 1:
        source = "HCP"

    ############################################################################################################3
    # Read npzfile
    npzfile = np.load("dialysis/results/day{}/B_Bp_Bpp_Bppp_scenario{}.npz".format(day, s))
    B_n_inf_rec = npzfile["B_n_inf_rec"]
    B_transmission_route = npzfile["B_transmission_route"]
    B_population = npzfile["B_population"]
    B_R0 = npzfile["B_R0"]
    B_generation_time = npzfile["B_generation_time"]
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

    Bpp_replaced_HCPs = np.mean(Bpp_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    Bppp_replaced_HCPs = np.mean(Bppp_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)


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
            plt.legend(loc="best")
            plt.grid(which='major', axis='both',linestyle=':')
            # plt.grid(which='minor', axis='x',linestyle=':')
            # plt.set_yticks(range(n_bins)/n_bins, minor=True)
            # plt.yaxis.grid(True, which='minor')
            # plt.legend(handles=[p1,p2,p3,p4], loc='best')
            plt.savefig("dialysis/tiff/plots/day{}/histogram_D{}_R0{}_scenario{}.tif".format(day, i, R0, s), dpi=300)
            # plt.savefig("dialysis/tiff/plots/day{}/histogram_D{}_R0{}_scenario{}.tif".format(day, i, R0, s))
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
            plt.legend(handles=[p1,p4,p3,p2], loc='best')
            plt.xticks(range(1, 31), xticklabels)
            plt.grid(zorder=0)
            plt.savefig("dialysis/tiff/plots/day{}/transmission_route_D{}_R0{}_scenario{}.tif".format(day, i, R0, s), dpi=300)
            plt.close()

