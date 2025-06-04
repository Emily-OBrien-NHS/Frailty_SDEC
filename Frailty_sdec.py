import simpy
import random
import math
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

class default_params():
    run_name = 'Frailty Model'
    #run times and iterations
    run_time = 20160#525600
    run_days = int(run_time/(60*24))
    iterations = 3#10
    occ_sample_time = 60
    #FSDEC Opening Hours
    FSDEC_open = 8
    FSDEC_stop_accept = 18
    FSDEC_close = 20
    FSDEC_open_days = [0, 1, 2, 3, 4]
    #inter arrival time (based on arrivals per day)
    mean_ED_to_FSDEC = 7 / (FSDEC_stop_accept - FSDEC_open)
    mean_ED_to_SSU = 180 #Need this number
    mean_GP_to_FSDEC = 2.8 / (FSDEC_stop_accept - FSDEC_open)
    #LoS
    mean_FSDEC_los = 10*60
    mean_SSU_los = 36*60
    max_SSU_los = 72*60
    #resources
    no_FSDEC_beds = 10
    no_SSU_beds = 14
    #Probability splits
    FSDEC_open_to_SSU = 0.08
    FSDEC_close_to_SSU = 0.5
    #lists for storing results
    pat_res = []
    occ_res = []

class spawn_patient:
    def __init__(self, p_id, ED_to_FSDEC_prob, FSDEC_to_SSU_prob):
        self.id = p_id
        #Record arrival mode
        self.arrival = ''
        #Record probabilities
        self.ED_to_FSDEC = (True if random.uniform(0,1) <= ED_to_FSDEC_prob
                            else False)
        self.FSDEC_to_SSU = (True if random.uniform(0,1) <= FSDEC_to_SSU_prob
                            else False)
        #Journey string
        self.journey_string = ''
        #recrord timings
        self.FSDEC_wait_start_time = np.nan
        self.FSDEC_admitted_time = np.nan
        self.FSDEC_leave_time = np.nan
        self.SSU_wait_start_time = np.nan
        self.SSU_admitted_time = np.nan
        self.SSU_leave_time = np.nan

class frailty_model:
    def __init__(self, run_number, input_params):
        self.patient_results = []
        self.mru_occupancy_results = []
        #start environment, set patient counter to 0 and set run number
        self.env = simpy.Environment()
        self.input_params = input_params
        self.patient_counter = 0
        self.run_number = run_number
        #establish resources
        self.FSDEC_bed = simpy.Resource(self.env,
                                        capacity=input_params.no_FSDEC_beds)
        self.SSU_bed = simpy.Resource(self.env,
                                      capacity=input_params.no_FSDEC_beds)
    
    ########################ARRIVALS################################
    def ED_arrivals(self):
        while True:
            #up patient counter and spawn a new walk-in patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter,
                              self.input_params.ED_to_FSDEC,
                              self.input_params.FSDEC_to_SSU)
            p.arrival = 'ED arrival'
            p.journey_string += 'ED > '
            #begin patient ED process
            self.env.process(self.frailty_journey(p))
            #randomly sample the time until the next patient arrival
            sampled_interarrival = random.expovariate(1.0
                                                / self.input_params.mean_ED_arr)
            yield self.env.timeout(sampled_interarrival)
    
    def GP_arrivals(self):
        while True:
            #up patient counter and spawn a new walk-in patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter,
                              self.input_params.ED_to_FSDEC,
                              self.input_params.FSDEC_to_SSU)
            p.arrival = 'GP arrival'
            p.journey_string += 'GP > '
            #begin patient ED process
            self.env.process(self.frailty_journey(p))
            #randomly sample the time until the next patient arrival
            sampled_interarrival = random.expovariate(1.0
                                                      / self.input_params.mean_GP_arr)
            yield self.env.timeout(sampled_interarrival)

    #################### CLOSE FSDEC ###########################
    def close_FSDEC(self):
        while self.env.now():
            x=9
    ################## FRAILTY JOURNEY #########################

    def frailty_journey(self, patient):
        #Enter FSDEC uf patient is GP arrival or ED to FSDEC
        if patient.arrival == 'GP arrival' or patient.ED_to_FSDEC:
            patient.journey_string += 'FSDEC > '
            patient.FSDEC_wait_start_time = self.env.now
            #request FSDEC bed
            with self.FSDEC_bed.request() as req:
                yield req
                patient.FSDEC_admitted_time = self.env.now
                sampled_FSDEC_time = random.expovariate(1.0
                                            / self.input_params.mean_FSDEC_los) 
                yield self.env.timeout(sampled_FSDEC_time)
            patient.FSDEC_leave_time = self.env.now
            #If patient does not continue on to SSU, then this is their final step
            if not patient.FSDEC_to_SSU:
                patient.journey_string += 'Out'
                self.store_patient_results(patient)
            #FSDEC journey
        #Patients who are not ED to FSDEC go straight to SSU.  Patients who are
        #FSDEC_to_SSU complete the above but are not sent home, so go through SSU.
        if (not patient.ED_to_FSDEC) or (patient.FSDEC_to_SSU):
            patient.journey_string += 'SSU > '
            patient.SSU_wait_start_time = self.env.now
            #Patient goes through SSU
            with self.SSU_bed.request() as req:
                yield req
                patient.SSU_admitted_time = self.env.now
                sampled_SSU_time = min(random.expovariate(1.0
                                            / self.input_params.mean_SSU_los),
                                       self.input_params.max_SSU_los)
                yield self.env.timeout(sampled_SSU_time)
            patient.SSU_leave_time = self.env.now
            #Patient leaves the model.
            patient.journey_string += 'Out'
            self.store_patient_results(patient)

###################RECORD RESULTS####################
    def store_patient_results(self, patient):
        self.patient_results.append([self.run_number, patient.id,
                                     patient.arrival,
                                     patient.ED_to_FSDEC,
                                     patient.FSDEC_to_SSU,
                                     patient.journey_string,
                                     patient.FSDEC_wait_start_time,
                                     patient.FSDEC_admitted_time,
                                     patient.FSDEC_leave_time,
                                     patient.SSU_wait_start_time,
                                     patient.SSU_admitted_time,
                                     patient.SSU_leave_time])
    
    def store_occupancy(self):
        while True:
            self.mru_occupancy_results.append([self.run_number,
                                               self.FSDEC_bed._env.now,
                                               self.FSDEC_bed.count,
                                               len(self.FSDEC_bed.queue),
                                               self.SSU_bed._env.now,
                                               self.SSU_bed.count,
                                               len(self.SSU_bed.queue)])
            yield self.env.timeout(self.input_params.occ_sample_time)
########################RUN#######################
    def run(self):
        self.env.process(self.ED_arrivals())
        self.env.process(self.GP_arrivals())
        self.env.process(self.store_occupancy())
        self.env.run(until = self.input_params.run_time)
        default_params.pat_res += self.patient_results
        default_params.occ_res += self.mru_occupancy_results
        return self.patient_results, self.mru_occupancy_results

def export_results(run_days, pat_results, occ_results):
    patient_df = pd.DataFrame(pat_results,
                              columns=['Run', 'Patient ID', 'Arrival Method',
                                       'ED to FSDEC', 'FSDEC to SSU', 'Journey',
                                       'FSDEC Wait Start Time',
                                       'FSDEC Arrival Time', 'FSDEC Leave Time',
                                       'SSU Wait Start Time',
                                       'SSU Arrival Time', 'SSU Leave Time'])
    patient_df['Simulation Arrival Time'] = (patient_df['FSDEC Wait Start Time']
                                             .fillna(patient_df['SSU Wait Start Time']))
    patient_df['Simulation Arrival Day'] = pd.cut(
                           patient_df['Simulation Arrival Time'], bins=run_days,
                           labels=np.linspace(1, run_days, run_days))
    patient_df['Wait for FSDEC Bed Time'] = (patient_df['FSDEC Arrival Time']
                                           - patient_df['FSDEC Wait Start Time'])
    patient_df['Wait for SSU Bed Time'] = (patient_df['SSU Arrival Time']
                                           - patient_df['SSU Wait Start Time'])
    patient_df['Simulation Leave Time'] = (patient_df['SSU Leave Time']
                                             .fillna(patient_df['FSDEC Leave Time']))
    patient_df['Simulation Leave Day'] = pd.cut(
                                    patient_df['Simulation Leave Time'], bins=run_days,
                                    labels=np.linspace(1, run_days, run_days))
    patient_df['Simulation Leave Hour'] = (patient_df['Simulation Leave Time']
                                           / 60).round().astype(int)

    
    occupancy_df = pd.DataFrame(occ_results,
                                columns=['Run', 'FSDEC Time', 'FSDEC Occupancy',
                                'FSDEC Bed Queue', 'SSU Time', 'SSU Occupancy',
                                'SSU Bed Queue'])
    occupancy_df['day'] = pd.cut(occupancy_df['FSDEC Time'], bins=run_days,
                                 labels=np.linspace(1, run_days, run_days))
    return patient_df, occupancy_df

def run_the_model(input_params):
    #run the model for the number of iterations specified
    #for run in stqdm(range(input_params.iterations), desc='Simulation progress...'):
    for run in range(input_params.iterations):
        print(f"Run {run+1} of {input_params.iterations}")
        model = frailty_model(run, input_params)
        model.run()
    patient_df, occ_df = export_results(input_params.run_days,
                                        input_params.pat_res,
                                        input_params.occ_res)
    return patient_df, occ_df

###############################################################################
#Run and save results
pat, occ = run_the_model(default_params)
pat.to_csv(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/Full Results/patients - {default_params.run_name}.csv')
occ.to_csv(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/Full Results/occupancy - {default_params.run_name}.csv')

# ####MRU leavers plot
# font_size = 24
# MRU_discharges = (pat.groupby(['Run', 'Simulation Leave Hour'], as_index=False)
#                   ['Patient ID'].count()
#                   .groupby('Simulation Leave Hour').mean()
#                   ['Patient ID'])
# MRU_discharges.columns = ['Hour', 'Patients Leaving MRU']
# daily_av = MRU_discharges.groupby((MRU_discharges.index/24).round()).mean()
# daily_av.index = daily_av.index * 24

# fig, axs = plt.subplots(figsize=(25, 10))
# axs.plot(MRU_discharges.index, MRU_discharges, color='grey', alpha=0.3, label='Hourly Leavers')
# axs.plot(daily_av.index, daily_av, 'r-', label='Daily Average Hourly Leavers')
# axs.set_title(f'Patients Leaving MRU per Hour - {default_params.run_name}', fontsize=font_size)
# axs.set_xlabel('Hour', fontsize=font_size)
# axs.set_ylabel('Patients Leaveing MRU', fontsize=font_size)
# axs.legend()
# axs.tick_params(axis='both',  which='major', labelsize=font_size)
# plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/Patients Leaving MRU - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
# plt.close()

# ####Occupancy plot
# mean_occupancy = (occ.groupby(['Run', 'Time'], as_index=False)
#                  [['ED Occupancy', 'MRU Occupancy']].mean().groupby('Time')
#                  [['ED Occupancy', 'MRU Occupancy']].mean()).round().astype(int)
# mean_occupancy.columns = ['Mean ED Occupancy', 'Mean MRU Occupancy']
# max_occupancy = (occ.groupby(['Run', 'Time'], as_index=False)
#                  [['ED Occupancy', 'MRU Occupancy']].max().groupby('Time')
#                  [['ED Occupancy', 'MRU Occupancy']].max())
# max_occupancy.columns = ['Max ED', 'Max MRU']
# min_occupancy = (occ.groupby(['Run', 'Time'], as_index=False)
#                  [['ED Occupancy', 'MRU Occupancy']].min().groupby('Time')
#                  [['ED Occupancy', 'MRU Occupancy']].min())
# min_occupancy.columns = ['Min ED', 'Min MRU']
# occupancy = min_occupancy.join(mean_occupancy).join(max_occupancy)

# ####ED
# fig, axs = plt.subplots(figsize=(25, 10))
# axs.plot(occupancy.index, occupancy['Mean ED Occupancy'], '-r')
# axs.fill_between(occupancy.index, occupancy['Min ED'], occupancy['Max ED'], color='grey', alpha=0.2)
# axs.set_title(f'Average Number of Patients in ED to go to MRU - {default_params.run_name}', fontsize=font_size)
# axs.set_xlabel('Time (Mins)', fontsize=font_size)
# axs.set_ylabel('No. Patients', fontsize=font_size)
# axs.tick_params(axis='both',  which='major', labelsize=font_size)
# plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/ED Occupancy - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
# plt.close()

# #MRU
# fig, axs = plt.subplots(figsize=(25, 10))
# axs.plot(occupancy.index, occupancy['Mean MRU Occupancy'], '-r')
# axs.fill_between(occupancy.index, occupancy['Min MRU'], occupancy['Max MRU'], color='grey', alpha=0.2)
# axs.set_title(f'Average Number of Patients in MRU - {default_params.run_name}', fontsize=font_size)
# axs.set_xlabel('Time (Mins)', fontsize=font_size)
# axs.set_ylabel('No. Patients', fontsize=font_size)
# axs.tick_params(axis='both',  which='major', labelsize=font_size)
# plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/MRU Occupancy - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
# plt.close()

# #### MRU Bed queue
# queue = occ.groupby('day')['MRU Bed Queue'].mean()
# fig, axs = plt.subplots(figsize=(25, 10))
# axs.plot(queue.index, queue)
# axs.set_title(f'Average Number of Patients in MRU Queue - {default_params.run_name}', fontsize=font_size)
# axs.set_xlabel('Simulation Day', fontsize=font_size)
# axs.set_ylabel('No. Patients', fontsize=font_size)
# axs.tick_params(axis='both',  which='major', labelsize=font_size)
# plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/Average Number of Patients in MRU Queue - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
# plt.close()

# #### MRU Bed Wait Time
# wait_for_bed = pat.groupby('Simulation Arrival Day')['Wait for MRU Bed Time'].mean() / 60
# fig, axs = plt.subplots(figsize=(25, 10))
# axs.plot(wait_for_bed.index, wait_for_bed)
# axs.set_title(f'Average Hours Waiting for MRU Bed - {default_params.run_name}', fontsize=font_size)
# axs.set_xlabel('Simulation Day', fontsize=font_size)
# axs.set_ylabel('Hours Waited', fontsize=font_size)
# axs.tick_params(axis='both',  which='major', labelsize=font_size)
# plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/MRU model/Results/Average Wait for Bed Time - {default_params.run_name}.svg', bbox_inches='tight', dpi=1200)
# plt.close()