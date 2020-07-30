"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: July, 2020

Description: This script prints numbers for paper

"""
import argparse
import numpy as np

def percentage_reduction(start_value, final_value):
    return 100 * (start_value - final_value) / start_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw cumulative incidence')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-s', '--scenario', type=int, default=0,
                        help= 'Infection source: scenario 0 = patient, scenario 1 = hcw')
    args = parser.parse_args()

    day = args.day
    s = args.scenario
    n_patients = 40
    n_HCPs = 11

    if s == 0:
        source = "patient"
    elif s == 1:
        source = "hcw"

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

    # npzfile = np.load("dialysis/results/day{}/final_N95_adj_patients_scenario{}.npz".format(day, s))
    # Bpppp_n_inf_rec = npzfile["Bpppp_n_inf_rec"]
    # Bpppp_transmission_route = npzfile["Bpppp_transmission_route"]
    # Bpppp_population = npzfile["Bpppp_population"]
    # Bpppp_R0 = npzfile["Bpppp_R0"]
    # Bpppp_generation_time = npzfile["Bpppp_generation_time"]
    # npzfile.close()
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

    # Bpppp_H_infection = Bpppp_n_inf_rec[:,:,:,:,0,0,:] + Bpppp_n_inf_rec[:,:,:,:,1,0,:] + Bpppp_n_inf_rec[:,:,:,:,2,0,:] 
    # Bpppp_P_infection = Bpppp_n_inf_rec[:,:,:,:,0,1,:] + Bpppp_n_inf_rec[:,:,:,:,1,1,:] + Bpppp_n_inf_rec[:,:,:,:,2,1,:] 
    # Bpppp_T_infection = Bpppp_H_infection + Bpppp_P_infection
    # Bpppp_T_cum_infection = np.cumsum(Bpppp_T_infection, axis=-1)
    # Bpppp_T_cum_attack_rate = Bpppp_T_cum_infection / Bpppp_population

    H0_replaced_HCPs = np.mean(H0_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    H1_replaced_HCPs = np.mean(H1_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    H3P1_replaced_HCPs = np.mean(H3P1_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    Bpp_replaced_HCPs = np.mean(Bpp_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    Bppp_replaced_HCPs = np.mean(Bppp_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
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
    # dim-2: [hcw_infected, patient_infected, hcw_recovered, patient_recovered]
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

    np.set_printoptions(1)
    n_simulations = B_T_cum_infection.shape[0]
    DM_idx = 3 # this is exp/exp 35% model
    R0_idx = 2 # this is R0 of 3

    print()
    print("*"*20)
    print("Abstract")
    print("\nattack rate B {:.1f}".format(100*np.mean(B_T_cum_attack_rate[:,DM_idx,R0_idx,-1])))
    print("attack rate Bppp {:.1f}".format(100*np.mean(Bppp_T_cum_attack_rate[:,DM_idx,R0_idx,0,-1])))
    start_value = 100*np.mean(B_T_cum_attack_rate[:,DM_idx,R0_idx,-1])
    final_value = 100*np.mean(Bppp_T_cum_attack_rate[:,DM_idx,R0_idx,0,-1])
    print("\npercentage reduction in attack rate in Bppp: {:.1f}".format(percentage_reduction(start_value, final_value)))

    Bppp_T_cum_infection = Bppp_T_cum_infection.astype(int)
    print("Likelihood of no additional infection: {:.1f}".format(100 * (Bppp_T_cum_infection[:,3,2,0,-1] == 1).sum() / n_simulations))

    print()
    print("*"*20)
    print("Result")

    B_TR = B_transmission_route[:,DM_idx,R0_idx,:,:].sum(axis=(0,-1))
    print("transmission route ratio: {}".format(100 * B_TR / B_TR.sum()))
    print("h->p, p->h, h->h, p->p")

    print("\nSelf-isolation, active surveillance")
    print("[exp/exp 5%]")
    print("attack rate H1 on R0 = 2  : {:.1f}".format(100*np.mean(H1_T_cum_attack_rate[:,2,0,0,-1], axis=0)))
    print("attack rate H1 on R0 = 2.5: {:.1f}".format(100*np.mean(H1_T_cum_attack_rate[:,2,1,0,-1], axis=0)))
    print("attack rate H1 on R0 = 3  : {:.1f}".format(100*np.mean(H1_T_cum_attack_rate[:,2,2,0,-1], axis=0)))
    print("npercentage reduction in attack rate in H1 on R0 = 2  : {:.1f}".format(percentage_reduction(100*np.mean(B_T_cum_attack_rate[:,2,0,-1]), 100*np.mean(H1_T_cum_attack_rate[:,2,0,0,-1], axis=0))))
    print("npercentage reduction in attack rate in H1 on R0 = 2.5: {:.1f}".format(percentage_reduction(100*np.mean(B_T_cum_attack_rate[:,2,1,-1]), 100*np.mean(H1_T_cum_attack_rate[:,2,1,0,-1], axis=0))))
    print("npercentage reduction in attack rate in H1 on R0 = 3  : {:.1f}".format(percentage_reduction(100*np.mean(B_T_cum_attack_rate[:,2,2,-1]), 100*np.mean(H1_T_cum_attack_rate[:,2,2,0,-1], axis=0))))

    print("\nmask")
    print("attack rate P2 {:.1f}".format(100*np.mean(P2_T_cum_attack_rate[:,DM_idx,R0_idx,-1])))
    print("attack rate H2P2v2 {:.1f}".format(100*np.mean(H2P2v2_T_cum_attack_rate[:,DM_idx,R0_idx,-1])))
    print("attack rate H2P2 {:.1f}".format(100*np.mean(H2P2_T_cum_attack_rate[:,DM_idx,R0_idx,-1])))
    
    print("\nMoving chairs apart")
    print("attack rate N1 {}".format(100*np.mean(N1_T_cum_attack_rate[:,DM_idx,R0_idx,:,-1], axis=0)))
    
    print("\npatient isolation, early replacement")
    print("attack rate H3P1 {}".format(100*np.mean(H3P1_T_cum_attack_rate[:,DM_idx,R0_idx,:,-1], axis=0)))
    print("Early replacement k HCPs: 1, 2, 3, 4, 5")
    
    print("\nBaseline+ Baseline++ Baseline+++")
    print("attack rate Baseline+ {:.1f}".format(100*np.mean(Bp_T_cum_attack_rate[:,DM_idx,R0_idx,-1], axis=0)))
    print("attack rate Baseline++, k=1 {:.1f}".format(100*np.mean(Bpp_T_cum_attack_rate[:,DM_idx,R0_idx,0,-1], axis=0)))
    print("attack rate Baseline+++, k=1 {:.1f}".format(100*np.mean(Bppp_T_cum_attack_rate[:,DM_idx,R0_idx,0,-1], axis=0)))
    
    # print()
    # print("Discussion")
    # print("N95 to 2 adj patients for 2 weeks")
    # print("rows: exp/exp(5%), exp/exp(35%), cols: R0 2, 2.5, 3")
    # print("{}".format(np.mean(Bpppp_T_cum_attack_rate[:,2:,:,0,-1], axis=0) * 100))

    B_end_infection = B_T_infection[:,DM_idx,R0_idx,:].sum(axis=-1)
    Bp_end_infection = Bp_T_infection[:,DM_idx,R0_idx,:].sum(axis=-1)
    Bpp_end_infection = Bpp_T_infection[:,DM_idx,R0_idx,0,:].sum(axis=-1)
    Bppp_end_infection = Bppp_T_infection[:,DM_idx,R0_idx,0,:].sum(axis=-1)
    # print("B: 51 infection at the end: {} ({})".format((B_end_infection==51).sum(), (B_end_infection==51).sum() / n_repeat))
    # print("B: 50 infection at the end: {} ({})".format((B_end_infection==50).sum(), (B_end_infection==50).sum() / n_repeat))

    # B2p_end_infection = B2p_T_infection[:,2,2,:].sum(axis=-1)
    # print("B2+: 1 infection at the end: {} ({})".format((B2p_end_infection==1).sum(), (B2p_end_infection==1).sum() / n_repeat))
    # print("B2+: >= 39 infection at the end: {} ({})".format((B2p_end_infection>=39).sum(), (B2p_end_infection>=39).sum() / n_repeat))

    # B2pp_end_infection = B2pp_T_infection[:,2,2,:].sum(axis=-1)
    # B2ppH1_end_infection = B2ppH1_T_infection[:,2,2,:].sum(axis=-1)

    # H2P2_end_infection = H2P2_T_infection[:,2,2,:].sum(axis=-1)
    # H12P2_end_infection = H12P2_T_infection[:,2,2,:].sum(axis=-1)

