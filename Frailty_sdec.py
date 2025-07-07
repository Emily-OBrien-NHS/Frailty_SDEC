import simpy
import random
import math
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import re

class default_params():
    run_name = 'Frailty Model v3 - 10 FSDEC Beds, 7 Days Open'
    #Set to true to print things for debugging.
    printouts = False
    #run times and iterations
    run_time = 525600
    run_days = int(run_time/(60*24))
    iterations = 10
    occ_sample_time = 60
    #FSDEC Opening Hours and days (to change closure days and hours, edit close
    # FSDEC function)
    FSDEC_weekend_close = False
    FSDEC_open_time = 8
    FSDEC_stop_accept = 16#18
    FSDEC_close_time = 20
    FSDEC_day_of_week = ([0, 1, 2, 3, 4] if FSDEC_weekend_close
                         else [0, 1, 2, 3, 4, 5, 6])
    FSDEC_accept_mins = (FSDEC_stop_accept - FSDEC_open_time)*60
    #inter arrival time (based on actual data - uncomment code in generators
    #to put this back in)
    # realtime_engine = create_engine('mssql+pyodbc://@dwrealtime/RealTimeReporting?'\
    #                     'trusted_connection=yes&driver=ODBC+Driver+17'\
    #                         '+for+SQL+Server')
    FSDEC_arrivals_SQL = """SET NOCOUNT ON
    declare @startdatetime as datetime
    set @startdatetime = '01-oct-2024 00:00:00'

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
    # FSDEC_arrivals = pd.read_sql(FSDEC_arrivals_SQL, realtime_engine)
    # FSDEC_arrivals['ArrivalDtm'] = (FSDEC_arrivals['EDClinicallyReadyToProceedDttm']
    #                                 .fillna(FSDEC_arrivals['WardStayStart']))
    # FSDEC_arrivals = FSDEC_arrivals.sort_values(by='ArrivalDtm')
    # FSDEC_arrivals['DoW'] = FSDEC_arrivals['ArrivalDtm'].dt.dayofweek
    # FSDEC_arrivals['HoD'] = FSDEC_arrivals['ArrivalDtm'].dt.hour
    # #All days and hours
    # days = []
    # hours = []
    # for day in list(range(0, 7)):
    #     for hour in list(range(0, 24)):
    #         days.append(day)
    #         hours.append(hour)
    # all_times = pd.DataFrame({'DoW':days, 'HoD':hours})
    # #Get ED to FSDEC Inter Arrivals
    # ED_to_FSDEC_arrivals = FSDEC_arrivals.loc[~FSDEC_arrivals['EDDischargeDateTime'].isna()].copy()
    # ED_to_FSDEC_arrivals['InterArr'] = (ED_to_FSDEC_arrivals['ArrivalDtm'].diff()
    #                                     / pd.Timedelta(minutes=1)).shift(-1)
    # ED_to_FSDEC_arrivals = (all_times.merge(ED_to_FSDEC_arrivals
    #                                 .groupby(['DoW', 'HoD'], as_index=False)
    #                                 ['InterArr'].mean().round(),
    #                                 on=['DoW', 'HoD'], how='left')
    #                                 .interpolate(limit_direction='both')
    #                                 .astype(int))

    # #Get GP/other to FSDEC Arrivals
    # GP_to_FSDEC_arrivals = FSDEC_arrivals.loc[FSDEC_arrivals['EDDischargeDateTime'].isna()].copy()
    # GP_to_FSDEC_arrivals['InterArr'] = (GP_to_FSDEC_arrivals['ArrivalDtm'].diff()
    #                                     / pd.Timedelta(minutes=1)).shift(-1)
    # GP_to_FSDEC_arrivals = (all_times.merge(GP_to_FSDEC_arrivals
    #                                 .groupby(['DoW', 'HoD'], as_index=False)
    #                                 ['InterArr'].mean().round(),
    #                                 on=['DoW', 'HoD'], how='left')
    #                                 .interpolate(limit_direction='both')
    #                                 .astype(int))

    #Inter arrival times based on rough number
    ED_to_FSDEC_daily_arrivals = 11
    GP_to_FSDEC_daily_arrivals = 4
    ED_to_SSU_daily_arrivals = 2.5
    mean_ED_to_FSDEC = (24*60) / ED_to_FSDEC_daily_arrivals
    mean_GP_to_FSDEC = (24*60) / GP_to_FSDEC_daily_arrivals
    mean_ED_to_SSU = (24*60) / ED_to_SSU_daily_arrivals
    #Length of Stay
    mean_FSDEC_los = 8 * 60
    max_FSDEC_los = 12 * 60 #only open 12 hours, longer than this is not possible.
    min_FSDEC_los = 4 * 60
    mean_SSU_los = 36 * 60
    max_SSU_los = 72 * 60
    min_SSU_los = 14 * 60
    #resources
    no_FSDEC_beds = 10
    no_SSU_beds = 15
    #Probability splits
    FSDEC_to_SSU = 0.08
    FSDEC_close_to_SSU = 0.5
    #lists for storing results
    pat_res = []
    occ_res = []

class spawn_patient:
    def __init__(self, p_id, FSDEC_to_SSU_prob, 
                 FSDEC_close_to_SSU_prob):
        #patient id
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
        #List to manually monitor FSDEC queue
        self.FSDEC_queue = []
        #Set up lists to record results
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
        self.SSU_bed = simpy.PriorityResource(self.env,
                                      capacity=input_params.no_SSU_beds)
        #Start FSDEC accepting condition event
        self.FSDEC_is_open = False
        self.FSDEC_accepting = False
        self.FSDEC_accepting_event = self.env.event()

    ######################MODEL TIME AND FSDEC OPEN#############################
    def model_time(self, time):
        #Work out what day and time it is in the model.
        day = math.floor(time / (24*60))
        day_of_week = day % 7
        #If day 0, hour is time / 60, otherwise it is the remainder time once
        #divided by number of days
        hour = math.floor((time % (day*(24*60)) if day != 0 else time) / 60)
        return day, day_of_week, hour
    
    def FSDEC_open(self, day, day_of_week, hour):
        #Function to work out if FSDEC is open and/or accepting at any time
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
            # day, day_of_week, hour = self.model_time(self.env.now)
            # inter_time = self.input_params.ED_to_FSDEC_arrivals.loc[
            #       (self.input_params.ED_to_FSDEC_arrivals['DoW'] == day_of_week)
            #     & (self.input_params.ED_to_FSDEC_arrivals['HoD'] == hour),
            #       'InterArr'].iloc[0]
            sampled_interarrival = round(random.expovariate(1.0 / self.input_params.mean_ED_to_FSDEC))
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
            # day, day_of_week, hour = self.model_time(self.env.now)
            # inter_time = self.input_params.GP_to_FSDEC_arrivals.loc[
            #       (self.input_params.GP_to_FSDEC_arrivals['DoW'] == day_of_week)
            #     & (self.input_params.GP_to_FSDEC_arrivals['HoD'] == hour),
            #       'InterArr'].iloc[0]
            sampled_interarrival = round(random.expovariate(1.0 / self.input_params.mean_GP_to_FSDEC))
            yield self.env.timeout(sampled_interarrival)

    ######################### CLOSE FSDEC ################################
    def create_FSDEC_accepting_event(self):
        #Event to monitor if FSDEC is open or closed
        if self.FSDEC_accepting:
            if not self.FSDEC_accepting_event.triggered:
                self.FSDEC_accepting_event.succeed()
        else:
        # If FSDEC_accepting is triggered, and we are now closing FSDEC,
        # replace with a new untriggered event
            if self.FSDEC_accepting_event.triggered:
                self.FSDEC_accepting_event = self.env.event()

    def close_FSDEC(self):
        #Manage FSDEC opening and closing schedule
        while True:
            #Get current model time
            current_time = self.env.now
            day, day_of_week, hour = self.model_time(current_time)
            
            # Check current status
            should_be_accepting, should_be_open = self.FSDEC_open(
                                                       day, day_of_week, hour)
            
            #Open FSDEC
            if should_be_accepting and not self.FSDEC_accepting:
                # Time to open FSDEC and start accepting patients
                if self.input_params.printouts:
                    print(f'!!!!!FSDEC START ACCEPTING AT {current_time}!!!!!')
                self.FSDEC_accepting = True
                self.FSDEC_is_open = True
                self.create_FSDEC_accepting_event()
            
            #Stop accepting to FSDEC
            elif not should_be_accepting and self.FSDEC_accepting:
                # Time to stop accepting new patients (6pm)
                if self.input_params.printouts:
                    print(f'!!!!!FSDEC STOP ACCEPTING AT {current_time}!!!!!')
                self.FSDEC_accepting = False
                self.create_FSDEC_accepting_event()  # Create new event but don't trigger
            
            #Close FSDEC
            elif not should_be_open and self.FSDEC_is_open:
                # Time to close completely (8pm) - claim all beds
                if self.input_params.printouts:
                    print(f'!!!!!FSDEC CLOSING AT {current_time}!!!!!')
                self.FSDEC_is_open = False
                self.FSDEC_accepting = False
                
                #If FSDEC closes for the weekend, use this logic
                if self.input_params.FSDEC_weekend_close:
                    # Calculate how long to stay closed
                    if day_of_week < 4:  # Monday-Thursday, reopen next morning
                        close_duration = (24 - self.input_params.FSDEC_close_time + 
                                        self.input_params.FSDEC_open_time) * 60
                    else:  # Friday, closed for weekend
                        close_duration = (72 - self.input_params.FSDEC_close_time + 
                                        self.input_params.FSDEC_open_time) * 60
                else: #close until next day
                    close_duration = (24 - self.input_params.FSDEC_close_time + 
                                        self.input_params.FSDEC_open_time) * 60
                
                #time out until next open
                if self.input_params.printouts:
                    print(f'!!!!!FSDEC CLOSED!!!!!')
                yield self.env.timeout(close_duration) 

                #FSDEC Re-opening
                if self.input_params.printouts:
                    print(f'!!!!!FSDEC REOPENING AT {self.env.now}!!!!!')
                    print(f'!!Time: {self.env.now}, Queue length: {len(self.FSDEC_bed.queue)}, shop occupancy: {self.FSDEC_bed.count}')
                self.FSDEC_is_open = True
                self.FSDEC_accepting = True
                self.create_FSDEC_accepting_event()
            
            # Check again in 60 minutes
            yield self.env.timeout(60)


    ######################## FRAILTY JOURNEY #############################

    def frailty_journey(self, patient):
        #Enter FSDEC if patient is GP arrival or ED to FSDEC
        if (patient.arrival == 'GP arrival') or (patient.arrival == 'ED to FSDEC arrival'):
            #Record step in patient journey and they time they did it
            patient.journey_string += 'FSDEC > '
            time = self.env.now
            patient.FSDEC_wait_start_time = time

            #Add patient to manual queue
            self.FSDEC_queue.append(patient)

            # Keep trying to get a bed until successful
            bed_secured = False
            req = None
            while not bed_secured:
                #If FSDEC is not accepting, patient waits until accepting 
                #event is triggered.
                if not self.FSDEC_accepting:
                    if self.input_params.printouts:
                        print(f'**patient {patient.id} waiting for FSDEC to start accepting at {self.env.now}')
                    yield self.FSDEC_accepting_event

                # FSDEC is now accepting, make a bed request
                req = self.FSDEC_bed.request()
                time = self.env.now
                day, day_of_week, hour = self.model_time(time)

                #Work out the time until FSDEC stops accepting for the day
                next_stop_accept_day = (day + 1 if hour 
                                        >= self.input_params.FSDEC_stop_accept
                                        else day)
                next_stop_accept = ((next_stop_accept_day * 60 * 24)
                                        + (self.input_params.FSDEC_stop_accept
                                        * 60))
                time_to_stop_accept = max((next_stop_accept - time), 1)

                if self.input_params.printouts:
                    print(f'??patient {patient.id} has requested bed at {time}, {time_to_stop_accept} until FSDEC stops accepting at {time + time_to_stop_accept}')

                #Patient either gets bed or timesout if past stop accept time
                bed_or_timeout = yield req | self.env.timeout(time_to_stop_accept)

                #If patientn got the bed before FSDEC stops accepting, they
                #leave the while loop and continue their journey
                if req in bed_or_timeout:
                    bed_secured = True
                    if self.input_params.printouts:
                        print(f'++patient {patient.id} got FSDEC bed at {self.env.now}, req status: {req.triggered}')
                #If FSDEC stops accepting before patient gets a bed, cancel
                #the request and try again
                else:
                    # Timed out, cancel request and wait for next opening
                    if self.input_params.printouts:
                        print(f'**patient {patient.id} timed out waiting for bed at {self.env.now}, trying again')
                    if not req.triggered:
                        req.cancel()

            #remove from manual queue as patient has a bed
            self.FSDEC_queue.remove(patient)
            #Record time
            patient.FSDEC_admitted_time = self.env.now
            day, day_of_week, hour = self.model_time(patient.FSDEC_admitted_time)
            #Get sampled FSDEC time
            sampled_FSDEC_time = round(random.expovariate(1.0
                                        / self.input_params.mean_FSDEC_los))
            #resample if not between max and min values
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

            # Release the bed and record patient leave time
            self.FSDEC_bed.release(req)
            if self.input_params.printouts:
                print(f'--patient {patient.id} leaving FSDEC bed at {self.env.now}')
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
                    
                #Get sampled SSU time
                sampled_SSU_time = round(random.expovariate(1.0
                                            / self.input_params.mean_SSU_los))
                #resample if not between max and min values
                while ((sampled_SSU_time < self.input_params.min_SSU_los)
                    or (sampled_SSU_time > self.input_params.max_SSU_los)):
                    sampled_SSU_time = round(random.expovariate(1.0
                                            / self.input_params.mean_SSU_los))
                patient.SSU_sampled_time = sampled_SSU_time


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
        yield self.env.timeout(1)
        while True:
            day, day_of_week, hour = self.model_time(self.env.now)
            FSDEC_open = self.FSDEC_open(day, day_of_week, hour)[1]
            self.mru_occupancy_results.append([self.run_number,
                                               self.FSDEC_bed._env.now - 1,
                                               (self.FSDEC_bed.count
                                                    if FSDEC_open else np.nan),
                                               len(self.FSDEC_queue),
                                               self.SSU_bed._env.now - 1,
                                               self.SSU_bed.count,
                                               len(self.SSU_bed.queue)])
            
            pats = [pat.id for pat in self.FSDEC_queue]
            if self.input_params.printouts:
                print(f'^^Time {self.env.now}, patients in FSDEC queue {pats}')
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

def time_to_day_and_hour(col):
    day = col  // (24*60)
    hour = ((col / 60) % 24).apply(np.floor)
    return pd.DataFrame({'Day':day, 'Hour': hour})

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
    patient_df[['Simulation Arrival Day',
                'Simulation Arrival Hour']] = time_to_day_and_hour(
                                              patient_df['Simulation Arrival Time'])

    #####FSDEC
    patient_df['Wait for FSDEC Bed Time'] = (patient_df['FSDEC Arrival Time']
                                        - patient_df['FSDEC Wait Start Time'])
    patient_df[['FSDEC Arrival Day',
                'FSDEC Arrival Hour']] = time_to_day_and_hour(
                                         patient_df['FSDEC Arrival Time'])
    #Have SSU entry time in case they queue to get in there.
    patient_df['FSDEC Actual Leave Time'] = np.where(patient_df['Journey']
                                                    .str.contains('FSDEC > SSU')
                                        | patient_df['Journey']
                                         .str.contains('FSDEC > KICKOUT > SSU'),
                                        patient_df['SSU Arrival Time'],
                                        patient_df['FSDEC Leave Time'])
    patient_df[['FSDEC Leave Day',
                'FSDEC Leave Hour']] = time_to_day_and_hour(
                                         patient_df['FSDEC Actual Leave Time'])
    patient_df['FSDEC LoS'] = (patient_df['FSDEC Actual Leave Time']
                               - patient_df['FSDEC Arrival Time'])
    #####SSU
    patient_df['Wait for SSU Bed Time'] = (patient_df['SSU Arrival Time']
                                           - patient_df['SSU Wait Start Time'])
    patient_df[['SSU Arrival Day',
                'SSU Arrival Hour']] = time_to_day_and_hour(
                                       patient_df['SSU Arrival Time'])
    patient_df[['SSU Leave Day',
                'SSU Leave Hour']] = time_to_day_and_hour(
                                       patient_df['SSU Leave Time'])
    patient_df['SSU LoS'] = (patient_df['SSU Leave Time']
                             - patient_df['SSU Arrival Time'])
    
    #####Leaving
    patient_df['Simulation Leave Time'] = (patient_df['SSU Leave Time']
                                        .fillna(patient_df['FSDEC Leave Time']))
    patient_df[['Simulation Leave Day',
                'Simulation Leave Hour']] = time_to_day_and_hour(
                                       patient_df['Simulation Leave Time'])

    ####################Occupancy Table
    occupancy_df = pd.DataFrame(occ_results,
                                columns=['Run', 'FSDEC Time', 'FSDEC Occupancy',
                                'FSDEC Bed Queue', #'FSDEC Bed Queue 2', 
                                'SSU Time', 'SSU Occupancy',
                                'SSU Bed Queue'])
    
    occupancy_df[['day', 'hour']] = time_to_day_and_hour(
                                    occupancy_df['FSDEC Time'])

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

#Summry table
#Throughput
FSDEC_av_throughput = pat.groupby(['FSDEC Leave Day', 'Run'])['Patient ID'].count().mean()
SSU_av_throughput = pat.groupby(['SSU Leave Day', 'Run'])['Patient ID'].count().mean()
#FSEC Queue
FSDEC_av_wait = pat['Wait for FSDEC Bed Time'].mean()
FSDEC_av_queue = occ['FSDEC Bed Queue'].mean()
#SSU Queue
SSU_av_wait = pat['Wait for SSU Bed Time'].mean()
SSU_av_queue = occ['SSU Bed Queue'].mean()
summary = pd.DataFrame({'Scenario' : [default_params.run_name],
                        'FSDEC Mean Daily Throughput' : [FSDEC_av_throughput],
                        'FSDEC Mean Bed Wait Time' : [FSDEC_av_wait],
                        'FSDEC Mean Queue Length' : [FSDEC_av_queue],
                        'SSU Mean Daily Throughput' : [FSDEC_av_throughput],
                        'SSU Mean Bed Wait Time' : [SSU_av_wait],
                        'SSU Mean Queue Length' : [SSU_av_queue]}).round(2)
summary.to_csv(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/Full Results/Summary - {default_params.run_name}.csv')

#Lists of weeks days for plots
DoW_7 = np.array([1, 2, 3, 4, 5, 6, 7]).astype(str)
DoW_df = pd.DataFrame({'DoW':[1, 2, 3, 4, 5, 6, 7]})

# 25th and 75th Percentiles functions 
def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)

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

#####Overall Occupancy plot
#Metrics by day of week
occ_metrics = (occ.groupby('day', observed=False, as_index=False)
               [['FSDEC Occupancy', 'SSU Occupancy']]
               .agg({'FSDEC Occupancy':['min', q25, 'mean', q75, 'max'],
                     'SSU Occupancy':['min', q25, 'mean', q75, 'max']}))
#plot
fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
fig.suptitle('Occupancy by Day', fontsize=24)
FSDEC_metrics = occ_metrics['FSDEC Occupancy']
ax1.plot(occ_metrics['day'], FSDEC_metrics['mean'].fillna(0), '-r', label='Mean')
ax1.fill_between(occ_metrics['day'], FSDEC_metrics['min'].fillna(0), FSDEC_metrics['max'].fillna(0), color='grey', alpha=0.2, label='Min-Max')
ax1.fill_between(occ_metrics['day'], FSDEC_metrics['q25'].fillna(0), FSDEC_metrics['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
ax1.set_title('FSDEC', fontsize=18)
ax1.set_ylabel('No. Beds Occupied', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
SSU_metrics = occ_metrics['SSU Occupancy']
ax2.plot(occ_metrics['day'], SSU_metrics['mean'], '-r')
ax2.fill_between(occ_metrics['day'], SSU_metrics['min'], SSU_metrics['max'], color='grey', alpha=0.2)
ax2.fill_between(occ_metrics['day'], SSU_metrics['q25'], SSU_metrics['q75'], color='black', alpha=0.2)
ax2.set_title('SSU', fontsize=18)
ax2.set_xlabel('Day of Model', fontsize=18)
ax2.tick_params(axis='both',  which='major', labelsize=18)
fig.tight_layout()
plt.savefig(f'C:/Users/obriene/Projects/Discrete Event Simulation/Frailty SDEC/Results/{default_params.run_name} - Occupancy.png',
            bbox_inches='tight', dpi=1200)
plt.close()

#####Occupancy Hour of Day plot
#Metrics by hour of day
occ_metrics = (occ.groupby('hour', observed=False,  as_index=False)
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

######Occupancy by DoW
occ['DoW'] = ((occ['day'].astype(int)) % 7) + 1
#Metrics by day of week
occ_metrics = (occ.groupby('DoW', observed=False,  as_index=False)
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
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['q25'].fillna(0), FSDEC_metrics['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
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
ax1.fill_between(FSDEC_metrics.index.tolist(), FSDEC_metrics['q25'].fillna(0), FSDEC_metrics['q75'].fillna(0), color='black', alpha=0.2, label='LQ-UQ')
ax1.set_title('FSDEC', fontsize=18)
ax1.set_ylabel('Mean Minutes Waited', fontsize=18)
ax1.tick_params(axis='both',  which='major', labelsize=18)
ax1.legend(fontsize=18)
SSU_metrics = wait_time['Wait for SSU Bed Time']
ax2.plot(FSDEC_metrics.index.tolist(), SSU_metrics['mean'], '-r')
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['min'], SSU_metrics['max'], color='grey', alpha=0.2)
ax2.fill_between(FSDEC_metrics.index.tolist(), SSU_metrics['q25'], SSU_metrics['q75'], color='black', alpha=0.2)
ax2.set_title('SSU', fontsize=18)
ax2.set_xlabel('Day of Week', fontsize=18)
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

x=5