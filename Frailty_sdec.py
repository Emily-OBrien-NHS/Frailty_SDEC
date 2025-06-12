import simpy
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import re

class default_params():
    run_name = 'Frailty Model'
    #run times and iterations
    run_time = 525600
    run_days = int(run_time/(60*24))
    iterations = 10
    occ_sample_time = 60
    #FSDEC Opening Hours (to change closure days and hours, edit close FSDEC
    #function)
    FSDEC_open_time = 8
    FSDEC_stop_accept = 18
    FSDEC_close_time = 20
    FSDEC_day_of_week = [0, 1, 2, 3, 4]
    FSDEC_accept_mins = (FSDEC_stop_accept - FSDEC_open_time)*60
    #inter arrival time (based on arrivals per day)
    realtime_engine = create_engine('mssql+pyodbc://@dwrealtime/RealTimeReporting?'\
                           'trusted_connection=yes&driver=ODBC+Driver+17'\
                               '+for+SQL+Server')
    FSDEC_arrivals_SQL = """SET NOCOUNT ON
    declare @startdatetime as datetime
    set @startdatetime = '21-jun-2023 00:00:00'

    ----ED attendances
    select LengthOfStay,
    AdmitPrvspRefno ,
    IsNewAttendance,
    AttendanceType,
    NCAttendanceId,
    dischargedatetime,
    departurereadydatetime
    into #ed
    from [cl3-data].[DataWarehouse].[ED].[vw_EDAttendance] 
    where DischargeDateTime is not NULL --get only discharged patients
    and ArrivalDateTime between DATEADD(day,-5,@startdatetime) and DATEADD(ss,-1,DATEADD(dd, DATEDIFF(dd, 0, GETDATE()), 0))--midnight yesterday)
    
    select distinct bstay.FKInpatientSpellId, --distinct removes duplicates for different beds
    bstay.WardStayStart,
    bstay.WardStayEnd,
    admitted_to_inpatient_ward = case when bsw.WardCode is not NULL then 1 else 0 end,
    discharging_ward = case when bstay.WardStayEnd = bstay.Discharged then 1 else 0 end,
    [DeparturePathway] = case when bsw.WardCode is not NULL then 'Admitted'
                                when wstay.WardStayCode in ('RK950AAU','RK950AAU01') then 'Medical SDEC'
                                when left(wstay.WardStayCode,5) in ('RK922','RK935','RK901','RK959') then 'Community Ward'
                                when wstay.WardStayCode = 'RK9VW01' then 'Virtual Ward'
                                when (bstay.WardStayEnd = bstay.Discharged) or (wstay.WardStayCode is NULL and wstaycurr.Discharged is not NULL) then 'Discharged'
                                when wstay.WardStayCode = 'RK950101' then 'Discharge Lounge'
                                when wstay.WardStayCode like 'RK950%' then 'Admitted'
                                else 'Other' end,
    bstay.IPMPrvspRefno,
    wstay.WardStayCode as NextWard,
    wstay.WardStayDescription as NextWardDesc,
    wstaycurr.Admitted,
    wstaycurr.Discharged,
    wstaycurr.Admet,
    wstaycurr.PatCl,
    wstaycurr.HospitalNumber,
    ed.dischargedatetime as EDDischargeDateTime,
    ed.departureReadyDateTime as EDClinicallyReadyToProceedDttm
    into  #afu_spells
    from RealTimeReporting.PCM.vw_bedstay bstay--using bedstay as this gets rid of dvt clinic
    left join RealTimeReporting.PCM.vw_WardStay wstay on wstay.FKInpatientSpellId = bstay.FKInpatientSpellId
            and wstay.WardStayStart = bstay.WardStayEnd
    left join RealTimeReporting.PCM.vw_WardStay wstaycurr on wstaycurr.FKInpatientSpellId = bstay.FKInpatientSpellId
            and wstaycurr.WardStayEnd = bstay.WardStayEnd
    ---find if the patient was previously in an inpatient ward (boarding overnight)
    left join [RealTimeReporting].[dbo].[covid_bed_state_wards] bsw on bsw.WardCode = wstay.WardStayCode
    left join #ed ed on ed.[AdmitPrvspRefno] = bstay.IPMPrvspRefno and ed.IsNewAttendance = 'Y'
    and ed.AttendanceType = '1' and NCAttendanceId <> '1204655'--remove repeated attendance under same admission prvsp refno
    where bstay.WardStayCode = 'RK950AFU'
    and bstay.WardStayStart between @startdatetime and DATEADD(ss,-1,DATEADD(dd, DATEDIFF(dd, 0, GETDATE()), 0))--midnight yesterday) --only retrieve data after the switchover date of 21 JUN 2023

    ---Grouping up by when the patient is discharged from ED
    select EDDischargeDateTime, EDClinicallyReadyToProceedDttm, WardStayStart, Admet
    from #afu_spells afu
    left join realtimereporting.Reference.[Date] dte
    on dte.[Date] = afu.EDDischargeDateTime """
    FSDEC_arrivals = pd.read_sql(FSDEC_arrivals_SQL, realtime_engine)
    FSDEC_arrivals['ArrivalDtm'] = (FSDEC_arrivals['EDClinicallyReadyToProceedDttm']
                                     .fillna(FSDEC_arrivals['WardStayStart']))
    FSDEC_arrivals = FSDEC_arrivals.sort_values(by='ArrivalDtm')
    FSDEC_arrivals['DoW'] = FSDEC_arrivals['ArrivalDtm'].dt.dayofweek
    FSDEC_arrivals['HoD'] = FSDEC_arrivals['ArrivalDtm'].dt.hour
    #All days and hours
    days = []
    hours = []
    for day in list(range(0, 7)):
        for hour in list(range(0, 24)):
            days.append(day)
            hours.append(hour)
    all_times = pd.DataFrame({'DoW':days, 'HoD':hours})
    #Get ED to FSDEC Inter Arrivals
    ED_to_FSDEC_arrivals = FSDEC_arrivals.loc[~FSDEC_arrivals['EDDischargeDateTime'].isna()].copy()
    ED_to_FSDEC_arrivals['InterArr'] = (ED_to_FSDEC_arrivals['ArrivalDtm'].diff()
                                        / pd.Timedelta(minutes=1)).shift(-1)
    ED_to_FSDEC_arrivals = (all_times.merge(ED_to_FSDEC_arrivals
                                     .groupby(['DoW', 'HoD'], as_index=False)
                                     ['InterArr'].mean().round(),
                                     on=['DoW', 'HoD'], how='left')
                                     .interpolate(limit_direction='both')
                                     .astype(int))
    #Get GP/other to FSDEC Arrivals
    GP_to_FSDEC_arrivals = FSDEC_arrivals.loc[FSDEC_arrivals['EDDischargeDateTime'].isna()].copy()
    GP_to_FSDEC_arrivals['InterArr'] = (GP_to_FSDEC_arrivals['ArrivalDtm'].diff()
                                        / pd.Timedelta(minutes=1)).shift(-1)
    GP_to_FSDEC_arrivals = (all_times.merge(GP_to_FSDEC_arrivals
                                     .groupby(['DoW', 'HoD'], as_index=False)
                                     ['InterArr'].mean().round(),
                                     on=['DoW', 'HoD'], how='left')
                                     .interpolate(limit_direction='both')
                                     .astype(int))
    #ED to SSU arrivals
    mean_ED_to_SSU = 420 #Need this number
    #LoS
    mean_FSDEC_los = 10*60
    max_FSDEC_los = 12*60 #only open 12 hours, longer than this is not possible.
    min_FSDEC_los = 60
    mean_SSU_los = 36*60
    max_SSU_los = 72*60
    #resources
    no_FSDEC_beds = 10
    no_SSU_beds = 14
    #Probability splits
    FSDEC_to_SSU = 0.08
    FSDEC_close_to_SSU = 0.5
    #lists for storing results
    pat_res = []
    occ_res = []

class spawn_patient:
    def __init__(self, p_id, FSDEC_to_SSU_prob, 
                 FSDEC_close_to_SSU_prob):
        self.id = p_id
        #Record arrival mode
        self.arrival = ''
        #Record probabilities
        self.FSDEC_to_SSU = (True if random.uniform(0,1)
                             <= FSDEC_to_SSU_prob
                             else False)
        self.FSDEC_close_to_SSU = (True if random.uniform(0,1)
                                   <= FSDEC_close_to_SSU_prob
                                   else False)
        self.SSU_priority = 0
        #Journey string
        self.journey_string = ''
        #recrord timings
        self.FSDEC_wait_start_time = np.nan
        self.FSDEC_admitted_time = np.nan
        self.FSDEC_sampled_time = np.nan
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
        self.FSDEC_bed = simpy.PriorityResource(self.env,
                                        capacity=input_params.no_FSDEC_beds)
        self.SSU_bed = simpy.PriorityResource(self.env,
                                      capacity=input_params.no_SSU_beds)
    ######################MODEL TIME AND FSDEC OPEN#############################
    def model_time(self, time):
        #Work out what time it is and if FSDEC is closed or not.
        day = math.floor(time / (24*60))
        day_of_week = day % 7
        #If day 0, hour is time / 60, otherwise it is the remainder time once
        #divided by number of days
        hour = math.floor((time % (day*(24*60)) if day != 0 else time) / 60)
        return day, day_of_week, hour
    
    def FSDEC_open(self, day, day_of_week, hour):
        FSDEC_accepting = ((day_of_week in self.input_params.FSDEC_day_of_week)
                           and ((hour >= self.input_params.FSDEC_open_time)
                              and (hour < self.input_params.FSDEC_stop_accept)))
        FSDEC_open = ((day_of_week in self.input_params.FSDEC_day_of_week)
                           and ((hour >= self.input_params.FSDEC_open_time)
                               and (hour < self.input_params.FSDEC_close_time)))
        return FSDEC_accepting, FSDEC_open
    ###########################ARRIVALS##################################
    def ED_to_SSU_arrivals(self):
        yield self.env.timeout(1)
        while True:
            #up patient counter and spawn a new SSU patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter,
                              self.input_params.FSDEC_to_SSU,
                              self.input_params.FSDEC_close_to_SSU)
            p.arrival = 'ED to SSU arrival'
            p.journey_string += 'ED > '
            #begin patient to FSDEC process
            self.env.process(self.frailty_journey(p))
            #randomly sample the time until the next patient arrival
            sampled_interarrival = round(random.expovariate(1.0
                                        / self.input_params.mean_ED_to_SSU))
            yield self.env.timeout(sampled_interarrival)

    def ED_to_FSDEC_arrivals(self):
        yield self.env.timeout(1)
        while True:
            #up patient counter and spawn a new walk-in patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter,
                                 self.input_params.FSDEC_to_SSU,
                                 self.input_params.FSDEC_close_to_SSU)
            p.arrival = 'ED to FSDEC arrival'
            p.journey_string += 'ED > '
            #begin patient to FSDEC process
            self.env.process(self.frailty_journey(p))
            #randomly sample the time until the next patient arrival based on
            #current time
            day, day_of_week, hour = self.model_time(self.env.now)
            inter_time = self.input_params.ED_to_FSDEC_arrivals.loc[
                  (self.input_params.ED_to_FSDEC_arrivals['DoW'] == day_of_week)
                & (self.input_params.ED_to_FSDEC_arrivals['HoD'] == hour),
                  'InterArr'].iloc[0]
            sampled_interarrival = round(random.expovariate(1.0 / inter_time))
            yield self.env.timeout(sampled_interarrival)
    
    def GP_to_FSDEC_arrivals(self):
        yield self.env.timeout(1)
        while True:
            #up patient counter and spawn a new patient 
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter,
                                  self.input_params.FSDEC_to_SSU,
                                  self.input_params.FSDEC_close_to_SSU)
            p.arrival = 'GP arrival'
            p.journey_string += 'GP > '
            #begin patient Frailty process
            self.env.process(self.frailty_journey(p))
            #randomly sample the time until the next patient arrival based on
            #current time
            day, day_of_week, hour = self.model_time(self.env.now)
            inter_time = self.input_params.GP_to_FSDEC_arrivals.loc[
                  (self.input_params.GP_to_FSDEC_arrivals['DoW'] == day_of_week)
                & (self.input_params.GP_to_FSDEC_arrivals['HoD'] == hour),
                  'InterArr'].iloc[0]
            sampled_interarrival = round(random.expovariate(1.0 / inter_time))
            yield self.env.timeout(sampled_interarrival)

    ######################### CLOSE FSDEC ################################
    def close_FSDEC(self):
        while True:
            #Work out what time it is and if FSDEC is closed or not.
            time = self.env.now
            day, day_of_week, hour = self.model_time(time)
            #First day,wait until 8pm then close
            if time == 0:
                time_closed = self.input_params.FSDEC_open_time
                time_out = self.input_params.FSDEC_close_time
            else:
                #If week day, close for 12 hour overnight and time out process
                # until next day
                if day_of_week < 4:
                    time_closed = 12
                    time_out = 24
                #If friday, close for weekend and time out until next close on
                # monday evening.
                else:
                    time_closed = 60
                    time_out = 72
            #Take away all the beds for the close time to siulate the unit being
            #closed.
            for _ in range(self.input_params.no_FSDEC_beds):
                self.env.process(self.fill_FSDEC(time_closed * 60))
               # yield self.env.timeout(0)
            #Timout for timeout period until next closure. 
            yield self.env.timeout(time_out * 60)
    
    def fill_FSDEC(self, time_closed):
        #If close time, take away all the FSDEC beds
        with self.FSDEC_bed.request(priority=-1) as req:
            yield req
            yield self.env.timeout(time_closed)

    ######################## FRAILTY JOURNEY #############################

    def frailty_journey(self, patient):
        #Enter FSDEC if patient is GP arrival or ED to FSDEC
        if (patient.arrival == 'GP arrival') or (patient.arrival == 'ED to FSDEC arrival'):
            patient.journey_string += 'FSDEC > '
            patient.FSDEC_wait_start_time = self.env.now
            #request FSDEC bed
            with self.FSDEC_bed.request() as req:
                yield req
                patient.FSDEC_admitted_time = self.env.now
                day, day_of_week, hour = self.model_time(patient.FSDEC_admitted_time)
                #Get sampled FSDEC time, resample if over maximum
                sampled_FSDEC_time = round(random.expovariate(1.0
                                           / self.input_params.mean_FSDEC_los))
                while ((sampled_FSDEC_time < self.input_params.min_FSDEC_los)
                    or (sampled_FSDEC_time > self.input_params.max_FSDEC_los)):
                    sampled_FSDEC_time = round(random.expovariate(1.0
                                           / self.input_params.mean_FSDEC_los))
                patient.FSDEC_sampled_time = sampled_FSDEC_time
                #Check if FSDEC will still be open when the patient leaves,
                #Else kick out to SSU or leave model.
                pat_leave = patient.FSDEC_admitted_time + sampled_FSDEC_time
                leave_day, leave_day_of_week, leave_hour = self.model_time(pat_leave)
                if not self.FSDEC_open(leave_day, leave_day_of_week, leave_hour)[1]:
                    patient.journey_string += 'KICKOUT > '
                    next_close = ((day * 60 * 24)
                                 + (self.input_params.FSDEC_close_time * 60))
                    sampled_FSDEC_time = max(next_close - patient.FSDEC_admitted_time - 1, 0)
                    #If patient is kicked out, their chance of going to
                    # SSU increases, with higher priortiy.
                    patient.FSDEC_to_SSU = patient.FSDEC_close_to_SSU
                    patient.SSU_priority = -1
                yield self.env.timeout(sampled_FSDEC_time)
            patient.FSDEC_leave_time = self.env.now
            #If patient does not continue on to SSU, then this is their final step
            if not patient.FSDEC_to_SSU:
                patient.journey_string += 'Out'
                self.store_patient_results(patient)

        #Patients who go straight from ED to SSU OR Patients who are
        #FSDEC_to_SSU complete the above but are not sent home, so go through SSU.
        if (patient.arrival == 'ED to SSU arrival') or (patient.FSDEC_to_SSU):
            patient.journey_string += 'SSU > '
            patient.SSU_wait_start_time = self.env.now
            #Patient goes through SSU
            with self.SSU_bed.request(priority = patient.SSU_priority) as req:
                yield req
                patient.SSU_admitted_time = self.env.now
                sampled_SSU_time = round(random.expovariate(1.0
                                            / self.input_params.mean_SSU_los))
                #Resample SSU time until less thamn the max.
                while sampled_SSU_time > self.input_params.max_SSU_los:
                    sampled_SSU_time = round(random.expovariate(1.0
                                            / self.input_params.mean_SSU_los))
                yield self.env.timeout(sampled_SSU_time)
            patient.SSU_leave_time = self.env.now
            #Patient leaves the model.
            patient.journey_string += 'Out'
            self.store_patient_results(patient)

###################RECORD RESULTS####################
    def store_patient_results(self, patient):
        self.patient_results.append([self.run_number, patient.id,
                                     patient.arrival,
                                     patient.FSDEC_to_SSU,
                                     patient.FSDEC_close_to_SSU,
                                     patient.journey_string,
                                     patient.FSDEC_wait_start_time,
                                     patient.FSDEC_admitted_time,
                                     patient.FSDEC_sampled_time,
                                     patient.FSDEC_leave_time,
                                     patient.SSU_wait_start_time,
                                     patient.SSU_admitted_time,
                                     patient.SSU_leave_time])
    
    def store_occupancy(self):
        while True:
            day, day_of_week, hour = self.model_time(self.env.now)
            FSDEC_open = self.FSDEC_open(day, day_of_week, hour)[1]
            self.mru_occupancy_results.append([self.run_number,
                                               self.FSDEC_bed._env.now,
                                               (self.FSDEC_bed.count
                                                     if FSDEC_open else np.nan),
                                               len(self.FSDEC_bed.queue),
                                               self.SSU_bed._env.now,
                                               self.SSU_bed.count,
                                               len(self.SSU_bed.queue)])
            yield self.env.timeout(self.input_params.occ_sample_time)
########################RUN#######################
    def run(self):
        self.env.process(self.ED_to_SSU_arrivals())
        self.env.process(self.ED_to_FSDEC_arrivals())
        self.env.process(self.GP_to_FSDEC_arrivals())
        self.env.process(self.close_FSDEC())
        self.env.process(self.store_occupancy())
        self.env.run(until = self.input_params.run_time)
        default_params.pat_res += self.patient_results
        default_params.occ_res += self.mru_occupancy_results
        return self.patient_results, self.mru_occupancy_results

def export_results(run_days, pat_results, occ_results):
    ####################Patient Table
    patient_df = pd.DataFrame(pat_results,
                 columns=['Run', 'Patient ID', 'Arrival Method',
                          'FSDEC to SSU', 'FSDEC close to SSU', 'Journey',
                          'FSDEC Wait Start Time', 'FSDEC Arrival Time', 'FSDEC Original LoS',
                          'FSDEC Leave Time', 'SSU Wait Start Time',
                          'SSU Arrival Time', 'SSU Leave Time'])
    #####Arrivals
    patient_df['Simulation Arrival Time'] = (patient_df['FSDEC Wait Start Time']
                                    .fillna(patient_df['SSU Wait Start Time']))
    patient_df['Simulation Arrival Day'] = pd.cut(
                           patient_df['Simulation Arrival Time'], bins=run_days,
                           labels=np.linspace(1, run_days, run_days))
    patient_df['Simulation Arrival Hour'] =(
                                        (patient_df['Simulation Arrival Time']
                                         / 60) % 24).apply(np.floor)
    #####FSDEC
    patient_df['Wait for FSDEC Bed Time'] = (patient_df['FSDEC Arrival Time']
                                        - patient_df['FSDEC Wait Start Time'])
    patient_df['FSDEC Arrival Hour'] = (
                                   (patient_df['FSDEC Arrival Time'] / 60) % 24
                                   ).apply(np.floor)
    patient_df['FSDEC Arrival Day'] = (patient_df['FSDEC Arrival Time']
    // (24*60))
    #Have SSU entry time in case they queue to get in there.
    patient_df['FSDEC Actual Leave Time'] = np.where(patient_df['Journey']
                                                    .str.contains('FSDEC > SSU')
                                        | patient_df['Journey']
                                         .str.contains('FSDEC > KICKOUT > SSU'),
                                        patient_df['SSU Arrival Time'],
                                        patient_df['FSDEC Leave Time']) 
    patient_df['FSDEC Leave Hour'] = ((patient_df['FSDEC Actual Leave Time']
                                       / 60) % 24).apply(np.floor)
    patient_df['FSDEC Leave Day'] = (patient_df['FSDEC Actual Leave Time']
                                    // (24*60))
    patient_df['FSDEC LoS'] = (patient_df['FSDEC Actual Leave Time']
                               - patient_df['FSDEC Arrival Time'])
    #####SSU
    patient_df['Wait for SSU Bed Time'] = (patient_df['SSU Arrival Time']
                                           - patient_df['SSU Wait Start Time'])
    patient_df['SSU Arrival Hour'] = (
                                     (patient_df['SSU Arrival Time'] / 60) % 24
                                     ).apply(np.floor)
    patient_df['SSU Arrival Day'] = patient_df['SSU Arrival Time'] // (24*60)
    patient_df['SSU LoS'] = (patient_df['SSU Leave Time']
                             - patient_df['SSU Arrival Time'])
    patient_df['SSU Leave Hour'] = ((patient_df['SSU Leave Time']
                                       / 60) % 24).apply(np.floor)
    patient_df['SSU Leave Day'] = (patient_df['SSU Leave Time']
                                    // (24*60))
    #####Leaving
    patient_df['Simulation Leave Time'] = (patient_df['SSU Leave Time']
                                        .fillna(patient_df['FSDEC Leave Time']))
    patient_df['Simulation Leave Day'] = pd.cut(
                                      patient_df['Simulation Leave Time'],
                                      bins=run_days,
                                      labels=np.linspace(1, run_days, run_days))
    patient_df['Simulation Leave Hour'] = (patient_df['Simulation Leave Time']
                                           / 60).round().astype(int)

    ####################Occupancy Table
    occupancy_df = pd.DataFrame(occ_results,
                                columns=['Run', 'FSDEC Time', 'FSDEC Occupancy',
                                'FSDEC Bed Queue', 'SSU Time', 'SSU Occupancy',
                                'SSU Bed Queue'])
    occupancy_df['day'] = pd.cut(occupancy_df['FSDEC Time'], bins=run_days,
                                 labels=np.linspace(1, run_days, run_days))
    occupancy_df['hour'] = (occupancy_df['FSDEC Time'] / 60) % 24

    return patient_df, occupancy_df

def run_the_model(input_params):
    #run the model for the number of iterations specified
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

#Lists of weeks days for plots
DoW_7 = np.array([1, 2, 3, 4, 5, 6, 7]).astype(str)
DoW_df = pd.DataFrame({'DoW':[1, 2, 3, 4, 5, 6, 7]})

#####LoS Plot
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Length of Stay')
ax1.set_title('FSDEC')
ax1.hist(pat['FSDEC LoS'], 100)
ax1.yaxis.set_tick_params(labelleft=False)
ax1.set_yticks([])
ax2.set_title('SSU')
ax2.set_xlabel('Minutes')
ax2.hist(pat['SSU LoS'], 100)
ax2.yaxis.set_tick_params(labelleft=False)
ax2.set_yticks([])
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - LoS.png',
            bbox_inches='tight', dpi=1200)
plt.close()

#####Occupancy Hour of Day plot
# 25th and 75th Percentiles
def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)
#Metrics by hour of day
occ_metrics = (occ.groupby('hour', as_index=False)
               [['FSDEC Occupancy', 'SSU Occupancy']]
               .agg({'FSDEC Occupancy':['min', q25, 'mean', q75, 'max'],
                     'SSU Occupancy':['min', q25, 'mean', q75, 'max']}))
#plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Occupancy by Hour of Day', fontsize=24)
FSDEC_metrics = occ_metrics['FSDEC Occupancy']
ax1.plot(occ_metrics['hour'], FSDEC_metrics['mean'].fillna(0), '-r', label='Mean')
ax1.fill_between(occ_metrics['hour'], FSDEC_metrics['min'].fillna(0), FSDEC_metrics['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
ax1.fill_between(occ_metrics['hour'], FSDEC_metrics['q25'].fillna(0), FSDEC_metrics['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
ax1.set_title('FSDEC', fontsize=18)
ax1.set_xlabel('Hour of Day', fontsize=18)
ax1.set_ylabel('No. Beds Occupied', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
SSU_metrics = occ_metrics['SSU Occupancy']
ax2.plot(occ_metrics['hour'], SSU_metrics['mean'], '-r')
ax2.fill_between(occ_metrics['hour'], SSU_metrics['min'], SSU_metrics['max'], color='grey', alpha=0.2)
ax2.fill_between(occ_metrics['hour'], SSU_metrics['q25'], SSU_metrics['q75'], color='black', alpha=0.2)
ax2.set_title('SSU', fontsize=18)
ax2.set_xlabel('Hour of Day', fontsize=18)
ax2.tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - Hourly Occupancy.png',
            bbox_inches='tight', dpi=1200)
plt.close()

#####Overall Occupancy plot?
occ['DoW'] = ((occ['day'].astype(int)-1) % 7) + 1
#Metrics by day of week
occ_metrics = (occ.groupby('DoW', as_index=False)
               [['FSDEC Occupancy', 'SSU Occupancy']]
               .agg({'FSDEC Occupancy':['min', q25, 'mean', q75, 'max'],
                     'SSU Occupancy':['min', q25, 'mean', q75, 'max']}))
#plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Occupancy by Day of Week', fontsize=24)
FSDEC_metrics = occ_metrics['FSDEC Occupancy']
ax1.plot(occ_metrics['DoW'], FSDEC_metrics['mean'].fillna(0), '-r', label='Mean')
ax1.fill_between(occ_metrics['DoW'], FSDEC_metrics['min'].fillna(0), FSDEC_metrics['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
ax1.fill_between(occ_metrics['DoW'], FSDEC_metrics['q25'].fillna(0), FSDEC_metrics['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
ax1.set_title('FSDEC', fontsize=18)
ax1.set_xlabel('Day of Week', fontsize=18)
ax1.set_ylabel('No. Beds Occupied', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
SSU_metrics = occ_metrics['SSU Occupancy']
ax2.plot(occ_metrics['DoW'], SSU_metrics['mean'], '-r')
ax2.fill_between(occ_metrics['DoW'], SSU_metrics['min'], SSU_metrics['max'], color='grey', alpha=0.2)
ax2.fill_between(occ_metrics['DoW'], SSU_metrics['q25'], SSU_metrics['q75'], color='black', alpha=0.2)
ax2.set_title('SSU', fontsize=18)
ax2.set_xlabel('Day of Week', fontsize=18)
ax2.tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - Day of Week Occupancy.png',
            bbox_inches='tight', dpi=1200)
plt.close()

#####FSDEC Admissions and Discharges from ED and GP
adm_metrics = pat.groupby(['Run', 'Arrival Method', 'FSDEC Arrival Day'], as_index=False)['Journey'].count()
adm_metrics['DoW'] = ((adm_metrics['FSDEC Arrival Day'].astype(int)) % 7) + 1
adm_metrics = adm_metrics.groupby(['Arrival Method', 'DoW'], as_index=False).agg({'Journey':['min', q25, 'mean', q75, 'max']})
adm_metrics.columns = [col[0] if col[1] == '' else col[1] for col in adm_metrics.columns]
ED_arr = DoW_df.merge(adm_metrics.loc[adm_metrics['Arrival Method'] == 'ED to FSDEC arrival'].copy(), on='DoW', how='left').fillna(0)
GP_arr = DoW_df.merge(adm_metrics.loc[adm_metrics['Arrival Method'] == 'GP arrival'].copy(), on='DoW', how='left').fillna(0)

dis_metrics = pat.groupby(['Run', 'Arrival Method', 'FSDEC Leave Day'], as_index=False)['Journey'].count()
dis_metrics['DoW'] = ((dis_metrics['FSDEC Leave Day'].astype(int)) % 7) + 1
dis_metrics = dis_metrics.groupby(['Arrival Method', 'DoW'], as_index=False).agg({'Journey':['min', q25, 'mean', q75, 'max']})
dis_metrics.columns = [col[0] if col[1] == '' else col[1] for col in dis_metrics.columns]
ED_dis = DoW_df.merge(dis_metrics.loc[dis_metrics['Arrival Method'] == 'ED to FSDEC arrival'].copy(), on='DoW', how='left').fillna(0)
GP_dis = DoW_df.merge(dis_metrics.loc[dis_metrics['Arrival Method'] == 'GP arrival'].copy(), on='DoW', how='left').fillna(0)

fig, axs = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle('FSDEC Day of Week Average Arrivals and Discharges by Arrival Mode', fontsize=24)
axs[0, 0].plot(DoW_7, ED_arr['mean'].fillna(0), '-r', label='Mean')
axs[0, 0].fill_between(DoW_7, ED_arr['min'].fillna(0), ED_arr['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
axs[0, 0].fill_between(DoW_7, ED_arr['q25'].fillna(0), ED_arr['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
axs[0, 0].set_title('Arrivals - ED', fontsize=18)
axs[0, 0].set_ylabel('No. Arrivals', fontsize=16)
axs[0, 0].tick_params(axis='both',  which='major', labelsize=18)
axs[0, 0].legend(fontsize=18)

axs[1, 0].plot(DoW_7, GP_arr['mean'], '-r')
axs[1, 0].fill_between(DoW_7, GP_arr['min'], GP_arr['max'], color='grey', alpha=0.2)
axs[1, 0].fill_between(DoW_7, GP_arr['q25'], GP_arr['q75'], color='black', alpha=0.2)
axs[1, 0].set_title('Arrivals - GP', fontsize=18)
axs[1, 0].set_ylabel('No. Arrivals', fontsize=16)
axs[1, 0].set_xlabel('Day of Week', fontsize=16)
axs[1, 0].tick_params(axis='both',  which='major', labelsize=18)

axs[0, 1].plot(DoW_7, ED_dis['mean'].fillna(0), '-r', label='Mean')
axs[0, 1].fill_between(DoW_7, ED_dis['min'].fillna(0), ED_dis['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
axs[0, 1].fill_between(DoW_7, ED_dis['q25'].fillna(0), ED_dis['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
axs[0, 1].set_title('Discharges - ED arrival', fontsize=18)
axs[0, 1].set_ylabel('No. Discharges', fontsize=16)
axs[0, 1].tick_params(axis='both',  which='major', labelsize=18)

axs[1, 1].plot(DoW_7, GP_dis['mean'], '-r')
axs[1, 1].fill_between(DoW_7, GP_dis['min'], GP_dis['max'], color='grey', alpha=0.2)
axs[1, 1].fill_between(DoW_7, GP_dis['q25'], GP_dis['q75'], color='black', alpha=0.2)
axs[1, 1].set_title('Discharges - GP arrival', fontsize=18)
axs[1, 1].set_ylabel('No. Discharges', fontsize=16)
axs[1, 1].set_xlabel('Day of Week', fontsize=16)
axs[1, 1].tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - FSDEC Arr and Dis.png',
            bbox_inches='tight', dpi=1200)
plt.close()

#####SSU Admissions and Discharges from ED and FSDEC
#Get arrival to SSU method
results = []
for journey in pat['Journey'].tolist():
    match = re.search(r'(?:^|\>\s*)(\w+)\s*(?=\>\s*KICKOUT\s*\>\s*SSU|\>\s*SSU)',
                      journey)
    if match:
        results.append(match.group(1))
    else:
        results.append(None)
pat['SSU Arrival Method'] = results
#filter to SSU arrivals only
SSU_pat = pat.loc[~pat['SSU Arrival Method'].isna()]

adm_metrics = SSU_pat.groupby(['Run', 'SSU Arrival Method', 'SSU Arrival Day'], as_index=False)['Journey'].count()
adm_metrics['DoW'] = ((adm_metrics['SSU Arrival Day'].astype(int)) % 7) + 1
adm_metrics = adm_metrics.groupby(['SSU Arrival Method', 'DoW'], as_index=False).agg({'Journey':['min', q25, 'mean', q75, 'max']})
adm_metrics.columns = [col[0] if col[1] == '' else col[1] for col in adm_metrics.columns]
ED_arr = DoW_df.merge(adm_metrics.loc[adm_metrics['SSU Arrival Method'] == 'ED'].copy(), on='DoW', how='left').fillna(0)
FSDEC_arr = DoW_df.merge(adm_metrics.loc[adm_metrics['SSU Arrival Method'] == 'FSDEC'].copy(), on='DoW', how='left').fillna(0)

dis_metrics = SSU_pat.groupby(['Run', 'SSU Arrival Method', 'SSU Leave Day'], as_index=False)['Journey'].count()
dis_metrics['DoW'] = ((dis_metrics['SSU Leave Day'].astype(int)) % 7) + 1
dis_metrics = dis_metrics.groupby(['SSU Arrival Method', 'DoW'], as_index=False).agg({'Journey':['min', q25, 'mean', q75, 'max']})
dis_metrics.columns = [col[0] if col[1] == '' else col[1] for col in dis_metrics.columns]
ED_dis = DoW_df.merge(dis_metrics.loc[dis_metrics['SSU Arrival Method'] == 'ED'].copy(), on='DoW', how='left').fillna(0)
FSDEC_dis = DoW_df.merge(dis_metrics.loc[dis_metrics['SSU Arrival Method'] == 'FSDEC'].copy(), on='DoW', how='left').fillna(0)

fig, axs = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle('SSU Day of Week Average Arrivals and Discharges by Arrival Mode', fontsize=24)
axs[0, 0].plot(DoW_7, ED_arr['mean'].fillna(0), '-r', label='Mean')
axs[0, 0].fill_between(DoW_7, ED_arr['min'].fillna(0), ED_arr['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
axs[0, 0].fill_between(DoW_7, ED_arr['q25'].fillna(0), ED_arr['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
axs[0, 0].set_title('Arrivals - ED', fontsize=18)
axs[0, 0].set_ylabel('No. Arrivals', fontsize=16)
axs[0, 0].tick_params(axis='both',  which='major', labelsize=18)
axs[0, 0].legend(fontsize=18)

axs[1, 0].plot(DoW_7, FSDEC_arr['mean'], '-r')
axs[1, 0].fill_between(DoW_7, FSDEC_arr['min'], FSDEC_arr['max'], color='grey', alpha=0.2)
axs[1, 0].fill_between(DoW_7, FSDEC_arr['q25'], FSDEC_arr['q75'], color='black', alpha=0.2)
axs[1, 0].set_title('Arrivals - FSDEC', fontsize=18)
axs[1, 0].set_ylabel('No. Arrivals', fontsize=16)
axs[1, 0].set_xlabel('Day of Week', fontsize=16)
axs[1, 0].tick_params(axis='both',  which='major', labelsize=18)

axs[0, 1].plot(DoW_7, ED_dis['mean'].fillna(0), '-r', label='Mean')
axs[0, 1].fill_between(DoW_7, ED_dis['min'].fillna(0), ED_dis['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
axs[0, 1].fill_between(DoW_7, ED_dis['q25'].fillna(0), ED_dis['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
axs[0, 1].set_title('Discharges - ED arrival', fontsize=18)
axs[0, 1].set_ylabel('No. Discharges', fontsize=16)
axs[0, 1].tick_params(axis='both',  which='major', labelsize=18)

axs[1, 1].plot(DoW_7, FSDEC_dis['mean'], '-r')
axs[1, 1].fill_between(DoW_7, FSDEC_dis['min'], FSDEC_dis['max'], color='grey', alpha=0.2)
axs[1, 1].fill_between(DoW_7, FSDEC_dis['q25'], FSDEC_dis['q75'], color='black', alpha=0.2)
axs[1, 1].set_title('Discharges - FSDEC arrival', fontsize=18)
axs[1, 1].set_ylabel('No. Discharges', fontsize=16)
axs[1, 1].set_xlabel('Day of Week', fontsize=16)
axs[1, 1].tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - SSU Arr and Dis.png',
            bbox_inches='tight', dpi=1200)
plt.close()

#####Average wait time per day
wait_time = pat.groupby(['Simulation Arrival Day'], observed=False).agg(
                    {'Wait for FSDEC Bed Time':['min', q25, 'mean', q75, 'max'],
                     'Wait for SSU Bed Time':['min', q25, 'mean', q75, 'max']})

#plot
fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
fig.suptitle('Average Bed Wait Time', fontsize=24)
FSDEC_metrics = wait_time['Wait for FSDEC Bed Time']
ax1.plot(FSDEC_metrics.index.tolist(), FSDEC_metrics['mean'].fillna(0), '-r', label='Mean')
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['min'].fillna(0), FSDEC_metrics['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['q25'].fillna(0), FSDEC_metrics['q75'].fillna(0), color='black', alpha=0.2, label='10th-90th Percentile')
ax1.set_title('FSDEC', fontsize=18)
ax1.set_ylabel('Mean Minutes Waited', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
SSU_metrics = wait_time['Wait for SSU Bed Time']
ax2.plot(FSDEC_metrics.index.tolist(), SSU_metrics['mean'], '-r')
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['min'], SSU_metrics['max'], color='grey', alpha=0.2)
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['q25'], SSU_metrics['q75'], color='black', alpha=0.2)
ax2.set_title('SSU', fontsize=18)
ax2.set_xlabel('Simulation Arrival Day', fontsize=18)
ax2.set_ylabel('Mean Minutes Waited', fontsize=18)
ax2.tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - Bed Wait Times.png',
            bbox_inches='tight', dpi=1200)
plt.close()

#####Average Wait time by DoW
pat['Simulation Arrival DoW'] = ((pat['Simulation Arrival Day'].astype(int)) % 7).replace({0:7})
wait_time = pat.groupby(['Simulation Arrival DoW'], observed=False).agg(
                    {'Wait for FSDEC Bed Time':['min', q25, 'mean', q75, 'max'],
                     'Wait for SSU Bed Time':['min', q25, 'mean', q75, 'max']})

#plot
fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
fig.suptitle('Average Bed Wait Time', fontsize=24)
FSDEC_metrics = wait_time['Wait for FSDEC Bed Time'].fillna(0)
ax1.plot(FSDEC_metrics.index.tolist(), FSDEC_metrics['mean'].fillna(0), '-r', label='Mean')
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['min'].fillna(0), FSDEC_metrics['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['q25'].fillna(0), FSDEC_metrics['q75'].fillna(0), color='black', alpha=0.2, label='10th-90th Percentile')
ax1.set_title('FSDEC', fontsize=18)
ax1.set_ylabel('Mean Minutes Waited', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
SSU_metrics = wait_time['Wait for SSU Bed Time']
ax2.plot(FSDEC_metrics.index.tolist(), SSU_metrics['mean'], '-r')
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['min'], SSU_metrics['max'], color='grey', alpha=0.2)
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['q25'], SSU_metrics['q75'], color='black', alpha=0.2)
ax2.set_title('SSU', fontsize=18)
ax2.set_xlabel('Simulation Arrival Day', fontsize=18)
ax2.set_ylabel('Mean Minutes Waited', fontsize=18)
ax2.tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - Bed Wait Times DoW.png',
            bbox_inches='tight', dpi=1200)
plt.close()

#####Queue by day
queue = occ.groupby('day', observed=False).agg({'FSDEC Bed Queue':['min', q25, 'mean', q75, 'max'],
                     'SSU Bed Queue':['min', q25, 'mean', q75, 'max']})
#plot
fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
fig.suptitle('Average Number of Patients Waiting for a Bed by Day', fontsize=24)
FSDEC_metrics = queue['FSDEC Bed Queue']
ax1.plot(FSDEC_metrics.index.tolist(), FSDEC_metrics['mean'].fillna(0), '-r', label='Mean')
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['min'].fillna(0), FSDEC_metrics['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['q25'].fillna(0), FSDEC_metrics['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
ax1.set_title('FSDEC', fontsize=18)
ax1.set_ylabel('No. Patients Waiting', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
SSU_metrics = queue['SSU Bed Queue']
ax2.plot(FSDEC_metrics.index.tolist(), SSU_metrics['mean'], '-r')
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['min'], SSU_metrics['max'], color='grey', alpha=0.2)
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['q25'], SSU_metrics['q75'], color='black', alpha=0.2)
ax2.set_title('SSU', fontsize=18)
ax2.set_ylabel('No. Patients Waiting', fontsize=18)
ax2.set_xlabel('Simulation Arrival Day', fontsize=18)
ax2.tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - Bed Queue.png',
            bbox_inches='tight', dpi=1200)
plt.close()

#####Queue by day of week
queue = occ.groupby('DoW').agg({'FSDEC Bed Queue':['min', q25, 'mean', q75, 'max'],
                     'SSU Bed Queue':['min', q25, 'mean', q75, 'max']})
#plot
fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
fig.suptitle('Average Number of Patients Waiting for a Bed by Day of Week', fontsize=24)
FSDEC_metrics = queue['FSDEC Bed Queue']
ax1.plot(FSDEC_metrics.index.tolist(), FSDEC_metrics['mean'].fillna(0), '-r', label='Mean')
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['min'].fillna(0), FSDEC_metrics['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['q25'].fillna(0), FSDEC_metrics['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
ax1.set_title('FSDEC', fontsize=18)
ax1.set_ylabel('No. Patients Waiting', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
SSU_metrics = queue['SSU Bed Queue']
ax2.plot(FSDEC_metrics.index.tolist(), SSU_metrics['mean'], '-r')
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['min'], SSU_metrics['max'], color='grey', alpha=0.2)
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['q25'], SSU_metrics['q75'], color='black', alpha=0.2)
ax2.set_title('SSU', fontsize=18)
ax2.set_ylabel('No. Patients Waiting', fontsize=18)
ax2.set_xlabel('Day of Week', fontsize=18)
ax2.tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - Bed Queue DoW.png',
            bbox_inches='tight', dpi=1200)
plt.close()