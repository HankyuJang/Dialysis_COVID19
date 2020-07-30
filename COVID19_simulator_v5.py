"""
COVID19 SIMULATOR

Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: June, 2020 

Description: This simulation class simulates COVID19 on given contact network

Each individual has its own incubation period and symptomatic period

difference from v3: Bpp now has one additional intervention that uses limited supply of N95 respirators.
difference from v4: H3P1 is different that the intervention is imposed on the 'first symptomatic agent'.

On the first day, 
either one patient in morning_patients array gets infected 
or one hcw in morning_hcws array gets infected
"""
import numpy as np
import random as rd
import copy

class Simulation:
    def __init__(self,
            W = 5,
            T = 7,
            inf = 1,
            sus = 0.3,
            QC = 1,
            asymp_rate = 0.2,
            asymp_shedding = 0.5,
            QS = 6,
            QT = 14,
            Dtype = 0,
            community_attack_rate = 0.0005,
            k = 1,
            mask_efficacy = np.array([0.4, 0.4, 0.93]),
            intervention = None,
            hcw_hcw_contact = None,
            hcw_patient_contact = None,
            patient_patient_contact = None,
            morning_patients = None,
            morning_hcws = None,
            verbose = False):

        self.W = W
        self.T = T
        self.inf = inf
        self.sus = sus
        self.QC = QC
        self.asymp_rate = asymp_rate
        self.asymp_shedding = asymp_shedding
        self.QS = QS
        self.QT = QT
        self.Dtype = Dtype
        self.intervention = intervention
        self.community_attack_rate = community_attack_rate
        self.k = k
        self.mask_efficacy = mask_efficacy
        self.hcw_hcw_contact = hcw_hcw_contact
        self.hcw_patient_contact = hcw_patient_contact
        self.patient_patient_contact = patient_patient_contact
        self.morning_patients = morning_patients
        self.morning_hcws = morning_hcws
        self.num_isolation_room = 1

        self.simulation_period = self.hcw_patient_contact.shape[0]
        self.n_hcw = self.hcw_patient_contact.shape[1]
        self.n_patient = self.hcw_patient_contact.shape[2]
        self.max_time = self.hcw_patient_contact.shape[3]

        self.hcw_W, self.hcw_T, self.patient_W, self.patient_T = self.draw_W_T()
        self.hcw_asymptomatic, self.patient_asymptomatic = self.get_asymptomatic()
        self.original_hcw_W = copy.deepcopy(self.hcw_W)
        self.original_hcw_T = copy.deepcopy(self.hcw_T)

        # Having negative status values means people are not infectious
        self.hcw_status = np.zeros((self.n_hcw)).astype(int) - 1
        self.patient_status = np.zeros((self.n_patient)).astype(int) - 1
        # Having negative status values means the hcw have not quarantined
        # Dim0: remaining quarantine days
        # Dim1: the HCW's disease status
        self.hcw_quarantine_status = np.zeros((2, self.n_hcw)).astype(int) - 1

        # dim0: [during incubation period, during symptomatic period, outside source (coming in infected)]
        # dim1: [hcw_infected, patient_infected, hcw_recovered, patient_recovered]
        self.n_inf_rec = np.zeros((3, 4, self.simulation_period)).astype(int)
        # dim0: [h->p, p->h, h->h, p->p]
        self.transmission_route = np.zeros((4, self.simulation_period)).astype(int)
        
        self.hcw_infection_source = -1
        self.patient_infection_source = -1

        self.hcw_daily_shedding, self.patient_daily_shedding = self.get_D()
        self.adjust_hcw_patient_daily_shedding()
        # We keep a separate version of shedding profiles, as hcw replacements changes the shedding profile
        # of hcw at the same index in the hcw_status.
        self.original_hcw_daily_shedding = copy.deepcopy(self.hcw_daily_shedding)

        self.hcw_recovered = np.zeros((self.n_hcw)).astype(bool)
        self.patient_recovered = np.zeros((self.n_patient)).astype(bool)

        self.population = np.zeros((self.simulation_period)).astype(int) + self.n_hcw + self.n_patient
        # self.total_population = self.n_hcw + self.n_patient
        self.attack_rate_array = np.zeros((self.simulation_period))
        self.attack_rate = 0
        self.R0 = 0
        self.generation_time = 0

        # Efficacy of mask. reduces inf
        self.hcw_mask_efficacy = self.mask_efficacy[0]
        self.patient_mask_efficacy = self.mask_efficacy[1]
        self.N95_efficacy = self.mask_efficacy[2]
        # self.patient_wearing_mask = np.zeros((self.n_patient)).astype(bool)
        self.hcw_bad_replacement = np.zeros((self.n_hcw)).astype(bool)

        self.N95_days_count = -1
        self.flag_PI = False

    def print_hcw_W_T(self):
        print(self.hcw_W, self.hcw_T, self.hcw_asymptomatic)

    def replacement_hcw_W_T_shedding(self, idx, W, T):
        self.hcw_W[idx] = W
        self.hcw_T[idx] = T
        self.hcw_daily_shedding[idx] = self.get_daily_shedding(W, T) * self.inf * self.sus 
        self.hcw_asymptomatic[idx] = np.random.random() < self.asymp_rate
        if self.hcw_asymptomatic[idx]:
            self.hcw_daily_shedding[idx] = self.hcw_daily_shedding[idx] * self.asymp_shedding

    # When isolated hcw comes back to unit, we retrieve original hcw's profiles
    def retrieve_original_hcw_W_T_shedding(self, idx):
        self.hcw_W[idx] = self.original_hcw_W[idx]
        self.hcw_T[idx] = self.original_hcw_T[idx]
        self.hcw_daily_shedding[idx] = self.original_hcw_daily_shedding[idx]

    def draw_W_T(self):
        hcw_W = np.random.geometric(p=1/self.W, size=self.n_hcw)
        hcw_T = np.random.geometric(p=1/self.T, size=self.n_hcw)
        patient_W = np.random.geometric(p=1/self.W, size=self.n_patient)
        patient_T = np.random.geometric(p=1/self.T, size=self.n_patient)
        return hcw_W, hcw_T, patient_W, patient_T
    
    def draw_W_T_for_replacement(self):
        W = np.random.geometric(p=1/self.W, size=1)[0]
        T = np.random.geometric(p=1/self.T, size=1)[0]
        return W, T

    def hcw_replacement(self, idx, day):
        self.population[day:] += 1
        # Update hcw profiles with the replacement's
        W, T = self.draw_W_T_for_replacement()
        self.replacement_hcw_W_T_shedding(idx, W, T)
        # Replacement is based on current community attack rate
        if rd.random() < self.community_attack_rate:
            self.hcw_status[idx] = W + T - 1
            self.add_infection_count("outside_source", "hcw", day)
        else:
            self.hcw_status[idx] = -1 # replacement is susceptible

    def get_asymptomatic(self):
        hcw_asymptomatic = np.random.random(self.n_hcw) < self.asymp_rate
        patient_asymptomatic = np.random.random(self.n_patient) < self.asymp_rate
        return hcw_asymptomatic, patient_asymptomatic

    def get_daily_shedding(self, W, T):
        daily_shedding = np.zeros((W + T))
        if self.Dtype == 2:
            daily_shedding[T-1] = 1
            # Infectivity during incubation period
            for idx in range(T, W + T):
                daily_shedding[idx] = 1/7.7 * daily_shedding[idx-1]
            # Infectivity during infectious period
            for idx in range(T-2, -1, -1):
                daily_shedding[idx] = 1/1.5 * daily_shedding[idx+1]
        # exp/exp: 35% asymptomatic spread
        elif self.Dtype == 3:
            daily_shedding[T-1] = 1
            # Infectivity during incubation period
            for idx in range(T, W + T):
                daily_shedding[idx] = 1/1.592 * daily_shedding[idx-1]
            # Infectivity during infectious period
            for idx in range(T-2, -1, -1):
                daily_shedding[idx] = 1/1.5 * daily_shedding[idx+1]
        return daily_shedding

    # Disease model: note that you need to change exp/exp based on W and T
    def get_D(self):
        # hcw_daily_shedding is a list of numpy arrays
        hcw_daily_shedding = []
        patient_daily_shedding = []
        for h in range(self.n_hcw):
            daily_shedding = self.get_daily_shedding(self.hcw_W[h], self.hcw_T[h])
            hcw_daily_shedding.append(daily_shedding)
        for p in range(self.n_patient):
            daily_shedding = self.get_daily_shedding(self.patient_W[p], self.patient_T[p])
            patient_daily_shedding.append(daily_shedding)

        return hcw_daily_shedding, patient_daily_shedding

    def adjust_hcw_patient_daily_shedding(self):
        # Multiply alpha (scaling parameter). alpha substitutes sus in this simulator. inf=1 (no effect)
        for h in range(self.n_hcw):
            self.hcw_daily_shedding[h] = self.hcw_daily_shedding[h] * self.inf * self.sus 
        for p in range(self.n_patient):
            self.patient_daily_shedding[p] = self.patient_daily_shedding[p] * self.inf * self.sus 
        # reduce shedding of asymptomatic agents
        for h in range(self.n_hcw):
            if self.hcw_asymptomatic[h]:
                self.hcw_daily_shedding[h] = self.hcw_daily_shedding[h] * self.asymp_shedding
        for p in range(self.n_patient):
            if self.patient_asymptomatic[p]:
                self.patient_daily_shedding[p] = self.patient_daily_shedding[p] * self.asymp_shedding

    def hcw_daily_replacement(self):
        pass

    def update_status(self, who, idx):
        if who == "patient":
            self.patient_status[idx] = self.patient_W[idx] + self.patient_T[idx] - 1
        elif who == "hcw":
            self.hcw_status[idx] = self.hcw_W[idx] + self.hcw_T[idx] - 1

    def add_infection_count(self, when, who, day):
        if when == "incubation":
            i = 0
        elif when == "after_incubation":
            i = 1
        elif when == "outside_source":
            i = 2
        if who == "hcw":
            j = 0
        elif who == "patient":
            j = 1
        self.n_inf_rec[i,j,day] += 1

    def add_recover_count(self, who, day):
        if who == "hcw":
            j = 2
        elif who == "patient":
            j = 3
        self.n_inf_rec[:,j,day] += 1

    def add_transmission_route(self, source_who, target_who, day):
        if source_who == "hcw" and target_who == "patient":
            i = 0
        elif source_who == "patient" and target_who == "hcw":
            i = 1
        elif source_who == "hcw" and target_who == "hcw":
            i = 2
        elif source_who == "patient" and target_who == "patient":
            i = 3
        self.transmission_route[i, day] += 1

    def mask_patients(self):
        for p in range(self.n_patient):
            self.patient_daily_shedding[p] *= (1 - self.patient_mask_efficacy)

    def mask_hcws(self):
        for h in range(self.n_hcw):
            self.hcw_daily_shedding[h] *= (1 - self.hcw_mask_efficacy)

    def N95_hcws(self):
        for h in range(self.n_hcw):
            self.hcw_daily_shedding[h] *= (1 - self.N95_efficacy)

    def unmask_patients(self):
        for p in range(self.n_patient):
            self.patient_daily_shedding[p] /= (1 - self.patient_mask_efficacy)

    def unmask_hcws(self):
        for h in range(self.n_hcw):
            self.hcw_daily_shedding[h] /= (1 - self.hcw_mask_efficacy)

    def unmask_N95_hcws(self):
        for h in range(self.n_hcw):
            self.hcw_daily_shedding[h] /= (1 - self.N95_efficacy)

    def mask_one_hcw(self, h):
        self.hcw_daily_shedding[h] *= (1 - self.hcw_mask_efficacy)

    def N95_one_hcw(self, h):
        self.hcw_daily_shedding[h] *= (1 - self.N95_efficacy)

    # hcw voluntarily isolates themselves the day after the symptoms
    def hcw_voluntary_isolation(self, idx, day):
        if (self.hcw_status[idx] == self.hcw_T[idx] - 2) and (rd.random() < self.QC):
            # 1. Replace the hcw if the hcw is original. Update hcw_quarantine_status
            if self.hcw_quarantine_status[0,idx] < 0:
                self.hcw_quarantine_status[0,idx] = self.QT - 1
                self.hcw_quarantine_status[1,idx] = self.hcw_status[idx]
            # 2. Replace if the hcw is substitute, and previously quarantineed hcw has not recovered
            # Do not update hcw_quarantine_status this case
            self.hcw_replacement(idx, day)

    def hcw_active_surveillance(self, idx, day):
        if self.hcw_status[idx] == self.hcw_T[idx] - 1:
            # 1. Replace the hcw if the hcw is original. Update hcw_quarantine_status
            if self.hcw_quarantine_status[0,idx] < 0:
                self.hcw_quarantine_status[0,idx] = self.QT - 1
                self.hcw_quarantine_status[1,idx] = self.hcw_status[idx]
            # 2. Replace if the hcw is substitute, and previously quarantineed hcw has not recovered
            # Do not update hcw_quarantine_status this case
            self.hcw_replacement(idx, day)
            # self.print_hcw_W_T()

    def patient_isolation_hcw_early_replacement(self, day, p):
        # Get top k hcws
        hcw_patient_contact_sum = self.hcw_patient_contact[:self.patient_W[p],:,:,:].sum(axis=(0,-1))
        contact_with_source = hcw_patient_contact_sum[:,p]
        top_k_hcws = np.argpartition(contact_with_source, -self.k)[-self.k:]
        # Isolate the source patient (remove all contacts with this patient)
        self.hcw_patient_contact[:,:,p,:] = 0
        self.patient_patient_contact[:,:,p,:] = 0
        self.patient_patient_contact[:,p,:,:] = 0

        for h in top_k_hcws:
            # If h is not exposed, it's a bad move. Record this HCW as this HCw will return as susceptible
            if self.hcw_status[h] < 0:
                self.hcw_bad_replacement[h] = True
            if self.hcw_quarantine_status[0,h] < 0:
                self.hcw_quarantine_status[0,h] = self.QT - 1
                self.hcw_quarantine_status[1,h] = self.hcw_status[h]
            self.hcw_replacement(h, day)
            # surgical mask the replacement
            if self.intervention[0, 2]:
                self.mask_one_hcw(h)

    def transmission(self, source_who, source_idx, target_who, target_idx, day):
        if source_who == "hcw" and target_who == "hcw":
            # If source is infected and target is susceptible
            if self.hcw_status[source_idx] >= 0 and self.hcw_status[target_idx] < 0:
                if rd.random() < self.hcw_daily_shedding[source_idx][self.hcw_status[source_idx]]:
                    self.hcw_status[target_idx] = self.hcw_W[target_idx] + self.hcw_T[target_idx] - 1
                    # Transmission during incubation period
                    if self.hcw_status[source_idx] >= self.hcw_T[source_idx]:
                        self.add_infection_count("incubation", "hcw", day)
                    else:
                        self.add_infection_count("after_incubation", "hcw", day)
                    self.add_transmission_route("hcw", "hcw", day)
                    if source_idx == self.hcw_infection_source:
                        self.R0 += 1
                        self.generation_time += day

        elif source_who == "hcw" and target_who == "patient":
            # If source is infected and target is susceptible
            if self.hcw_status[source_idx] >= 0 and self.patient_status[target_idx] < 0:
                if rd.random() < self.hcw_daily_shedding[source_idx][self.hcw_status[source_idx]]:
                    self.patient_status[target_idx] = self.patient_W[target_idx] + self.patient_T[target_idx] - 1
                    # Transmission during incubation period
                    if self.hcw_status[source_idx] >= self.hcw_T[source_idx]:
                        self.add_infection_count("incubation", "patient", day)
                    else:
                        self.add_infection_count("after_incubation", "patient", day)
                    self.add_transmission_route("hcw", "patient", day)
                    if source_idx == self.hcw_infection_source:
                        self.R0 += 1
                        self.generation_time += day

        elif source_who == "patient" and target_who == "hcw":
            # If source is infected and target is susceptible
            if self.patient_status[source_idx] >= 0 and self.hcw_status[target_idx] < 0:
                if rd.random() < self.patient_daily_shedding[source_idx][self.patient_status[source_idx]]:
                    self.hcw_status[target_idx] = self.hcw_W[target_idx] + self.hcw_T[target_idx] - 1
                    # Transmission during incubation period
                    if self.patient_status[source_idx] >= self.patient_T[source_idx]:
                        self.add_infection_count("incubation", "hcw", day)
                    else:
                        self.add_infection_count("after_incubation", "hcw", day)
                    self.add_transmission_route("patient", "hcw", day)
                    if source_idx == self.patient_infection_source:
                        self.R0 += 1
                        self.generation_time += day

        elif source_who == "patient" and target_who == "patient":
            # If source is infected and target is susceptible
            if self.patient_status[source_idx] >= 0 and self.patient_status[target_idx] < 0:
                if rd.random() < self.patient_daily_shedding[source_idx][self.patient_status[source_idx]]:
                    self.patient_status[target_idx] = self.patient_W[target_idx] + self.patient_T[target_idx] - 1
                    # Transmission during incubation period
                    if self.patient_status[source_idx] >= self.patient_T[source_idx]:
                        self.add_infection_count("incubation", "patient", day)
                    else:
                        self.add_infection_count("after_incubation", "patient", day)
                    self.add_transmission_route("patient", "patient", day)
                    if source_idx == self.patient_infection_source:
                        self.R0 += 1
                        self.generation_time += day

    def any_symptomatic_agents(self):
        pass

    def simulate(self):
        # self.print_hcw_W_T()
        ###################################################################
        # Things to be done at the start of the simulation
        ###################################################################
        # Infect one morning patient in the first day
        if self.morning_patients.size > 0:
            # self.n_inf_rec[2,1,0] += 1
            self.add_infection_count("outside_source", "patient", 0)
            self.patient_infection_source = np.random.choice(self.morning_patients)
            # self.patient_status[self.patient_infection_source] = self.W + self.T - 1
            self.update_status("patient", self.patient_infection_source)
            self.hcw_infection_source = -1
            self.infection_source_W = self.patient_W[self.patient_infection_source]
        # Infect one morning hcw in the first day
        elif self.morning_hcws.size > 0:
            # self.n_inf_rec[2,0,0] += 1
            self.add_infection_count("outside_source", "hcw", 0)
            self.hcw_infection_source = np.random.choice(self.morning_hcws)
            # self.hcw_status[self.hcw_infection_source] = self.W + self.T - 1
            self.update_status("hcw", self.hcw_infection_source)
            self.patient_infection_source = -1
            self.infection_source_W = self.hcw_W[self.hcw_infection_source]

        if self.intervention[0,3] and self.intervention[1,1]:
            self.flag_PI = True

        for d in range(self.simulation_period):
            # print(d)
            ###################################################################
            # Things to be done at the start of each day - Interventions
            ###################################################################
            # H3P1: As soon as the infection source start shedding (if the source is not asymptomatic)
            # If source: patient, isolate patient and replace top k hcws
            # If source: hcw, isolate top patient and replace top k hcws (including the source hcw)
            # if self.intervention[0,3] and self.intervention[1,1] and d == self.infection_source_W:
                # self.patient_isolation_hcw_early_replacement(d)
                # self.N95_days_count = 14

            # Intervention on HCW: HCW presenteeism (self-quarantine)
            if self.intervention[0, 0]:
                for h in range(self.n_hcw):
                    # nothing happens if the hcw is asymptomatic
                    if self.hcw_asymptomatic[h]:
                        continue
                    self.hcw_voluntary_isolation(h, d)

            # Intervention on HCW: Active surveilence. Measure temperature before working
            if self.intervention[0, 1]:
                for h in range(self.n_hcw):
                    # nothing happens if the hcw is asymptomatic
                    if self.hcw_asymptomatic[h]:
                        continue
                    self.hcw_active_surveillance(h, d)

            # P2: Masks on patients (everyone, all the time)
            if self.intervention[1, 2]:
                self.mask_patients()

            # H2: Masks on hcws (everyone, all the time)
            if self.intervention[2, 2] and self.N95_days_count > 0:
                self.N95_hcws()
            elif self.intervention[0, 2]:
                self.mask_hcws()

            # if self.intervention[2, 2] and self.N95_days_count < 0 and self.any_symptomatic_agents():
                # self.N95_days_count = 14
                # self.unmask_hcws()
                # self.N95_hcws()

            # if self.intervention[2, 2] and self.N95_days_count == 0:
                # self.unmask_N95_hcws()
                # self.mask_hcws()

            for t in range(self.max_time):
                # H3P1
                # As soon as a symptomatic patient is detected, replace that patient as well as top k hcws.
                if self.flag_PI:
                    # if any symptomatic patient is in the unit at that time
                    hcw_patient_pairs = np.transpose(self.hcw_patient_contact[d,:,:,t].nonzero())
                    for h, p in hcw_patient_pairs:
                        if self.patient_status[p] == (self.patient_T[p] - 1) and not self.patient_asymptomatic[p]:
                            # patient isolation, hcw replacement
                            self.patient_isolation_hcw_early_replacement(d, p)
                            self.flag_PI = False
                            # at this point of time, all hcws in the unit are wearing surgical masks
                            if self.intervention[2, 2]:
                                # start N95 counter, unmask HCWs and the N95s on the HCWs
                                self.N95_days_count = 14
                                self.unmask_hcws()
                                self.N95_hcws()
                            break

                #########################################################################################
                # Start transmission based on contacts
                #########################################################################################
                # Any hcw-hcw contacts?
                hcw_hcw_pairs = np.transpose(self.hcw_hcw_contact[d,:,:,t].nonzero())
                for h1, h2 in hcw_hcw_pairs:
                    # No transmission happens if either of them have recovered
                    if self.hcw_recovered[h1] or self.hcw_recovered[h2]:
                        continue
                    # Disease flow: h1 -> h2
                    self.transmission("hcw", h1, "hcw", h2, d)
                    # Disease flow: h2 -> h1
                    self.transmission("hcw", h2, "hcw", h1, d)

                            
                # Any hcw-patient contacts?
                hcw_patient_pairs = np.transpose(self.hcw_patient_contact[d,:,:,t].nonzero())
                for h, p in hcw_patient_pairs:
                    # No transmission happens if either of them have recovered
                    if self.hcw_recovered[h] or self.patient_recovered[p]:
                        continue
                    # Disease flow: h -> p
                    self.transmission("hcw", h, "patient", p, d)
                    # Disease flow: p -> h
                    self.transmission("patient", p, "hcw", h, d)

                # Any patient-patient contacts?
                patient_patient_pairs = np.transpose(self.patient_patient_contact[d,:,:,t].nonzero())
                for p1, p2 in patient_patient_pairs:
                    if self.patient_recovered[p1] or self.patient_recovered[p2]:
                        continue
                    # Disease flow: p1 -> p2
                    self.transmission("patient", p1, "patient", p2, d)
                    # Disease flow: p2 -> p1
                    self.transmission("patient", p2, "patient", p1, d)

            ###################################################################
            # Things to be done at the end of each day
            ###################################################################
            # P2: Masks off patients (everyone, all the time)
            if self.intervention[1, 2]:
                self.unmask_patients()

            # H2: Masks off hcws (everyone, all the time)
            if self.intervention[2, 2] and self.N95_days_count > 0:
                self.unmask_N95_hcws()
            elif self.intervention[0, 2]:
                self.unmask_hcws()

            self.N95_days_count -= 1
            # Any hcw recovered? (original hcws, not substitutes)
            for h in range(self.n_hcw):
                # if hcw_status[h] == 0, it means it was h's last infectious day
                if self.hcw_status[h] == 0: # status of hcws in the unit
                    self.add_recover_count("hcw", d)
                    self.hcw_recovered[h] = True
                # Quaranteened hcw comes back after QT days. Comes back to work the next day
                if self.hcw_quarantine_status[0,h] == 0: # status of hcws in quarantine (last day of quarantine)
                    self.retrieve_original_hcw_W_T_shedding(h)
                    if self.hcw_bad_replacement[h]: # if quarantined hcw was not exposed, return as susceptible
                        self.hcw_recovered[h] = False
                        self.hcw_status[h] = 0 # susceptible (as it reduces by 1 at the end of the day)
                    elif self.hcw_quarantine_status[1,h] >= 1: # if shedding doesn't end today
                        self.hcw_recovered[h] = False
                        self.hcw_status[h] = self.hcw_quarantine_status[1,h] # infectious
                    else: # HCW is recovered
                        self.add_recover_count("hcw", d)
                        self.hcw_recovered[h] = True
                        self.hcw_status[h] = 0

            # Any patient recovered?
            for p in range(self.n_patient):
                if self.patient_status[p] == 0: # status of patients in the unit
                    self.add_recover_count("patient", d)
                    self.patient_recovered[p] = True

            # Reduce the infected days by 1 for everyone
            self.hcw_status -= 1
            self.patient_status -= 1
            self.hcw_quarantine_status -= 1
        ###################################################################
        # Things to be done at the end of the simulation
        ###################################################################
