
# load survey data

from geopy.distance import distance
import numpy as np
import pandas as pd
import datetime

import json

from scipy.interpolate import interp1d

from src.processing.scripts.gpr.utils.processing_utils import read_segy, cut_profile #internal
from src.processing.scripts.gpr.utils.gpr_configs import ProcessingConfig, GPRtypeConfig
#from .processing_utils import read_segy, cut_profile #internal
#from .Process_GPR_configs import ProcessingConfig, GPRtypeConfig


#data object for storing our gpr survey data
class ProjectGPRdata(object):

    def __init__(self, 
                 config_processing,
                 config_gpr,
                 dfPos: pd.DataFrame,
                 distVecList: list,
                 gprDataList: list) -> None:

        self.config = config_processing
        self.config_gpr = config_gpr
        #list of segment-data (.npy) dim ntraces, nsamples
        self.df_Pos = dfPos

        # from created .pos-file
        self.distVecList = distVecList
        self.gprDataList = gprDataList

        nrSeg1 = len(distVecList)
        nrSeg2 = len(gprDataList)

        if nrSeg1 != nrSeg2:
            print("===> WARNING: number distVecs and associated GPR profiles do not match!")
            #should also check dims of each array
        else:
            self.numberOfSegments = nrSeg1

        self.lat_list = list()
        self.lon_list = list()


    @classmethod
    def from_filepath(cls, filepath):
        # Initialize an instance of the class using data from a file
        data = np.load(filepath, allow_pickle=True)
        
        config_processing = json.loads(data['config'].item())
        config_gpr = json.loads(data['config_gpr'].item())
        
        # Load the DataFrame from a CSV file
        dfPos_columns = data['df_Pos_columns']
        dfPos_data = data['df_Pos']
        dfPos = pd.DataFrame(data=dfPos_data, columns=dfPos_columns)
        #dfPos = pd.read_csv(data['df_Pos'], index_col=0)
        #dfPos = pd.DataFrame(data['df_Pos']) # .item())
        
        distVecList = [data[arr_name] for arr_name in data.files if arr_name.startswith('distVec_array_')]
        gprDataList = [data[arr_name] for arr_name in data.files if arr_name.startswith('gprProfile_array_')]

        #processing_config = ProcessingConfig(**config_settings)

        return cls(config_processing, config_gpr, dfPos, distVecList, gprDataList)

    def getTimeSampleVector(self, segment_number = None):

        dt = self.config.dt if hasattr(self.config, 'dt') else self.config["dt"] if isinstance(self.config, dict) else 100/512
        resample = self.config.resample if hasattr(self.config, 'resample') else self.config["resample"] if isinstance(self.config, dict) else 1

        #data for first segment
        isegUse  = 0 or (segment_number-1)
        nsamples,ntraces = self.gprDataList[isegUse].shape

        time_ns = [(i * dt/resample) for i in range(0, nsamples)]

        return time_ns

    def getListSegmentsIncluded(self):
        '''
        In some cases, only a selection of segments are included in the data.
        This routine returns the number of segments included and a list of
        the segments (note: segment number starts at 1 ->).
        Example: segments stored [3,5,7,9,11], i.e. 5 segments in total
        '''
        distinct_segment_numbers = self.df_Pos['Segment-number'].unique().tolist()
        return distinct_segment_numbers


    def getDataForSegment(self, segment_number):
        '''Return segment data and info for given segment number
        @args:
        - segment_number : target segment (integer between 1 and number of segments)
        
        @returns:
        - segment_GPRdata : 
        - segment_distVec : 
        - segment_dfPos   : 
        '''
        
        #in case that project has been saved with only a selection of segments
        distinct_segment_numbers = self.df_Pos['Segment-number'].unique().tolist()

        numberOfSegmentsIncluded = len(distinct_segment_numbers)


        #if segment_number > self.numberOfSegments:
        #    print(' ==> Not a valid segment number!')
        #    print(' select a segment between 1 and ' + str(self.numberOfSegments))
        #    return
        #if segment_number > numberOfSegmentsIncluded:
        if segment_number not in distinct_segment_numbers:
            print(' ==> Not a valid segment number! ' + str(segment_number))
            print(' select a segment number in this list:')
            print(distinct_segment_numbers)
            return
        else:

            #slice data-frame
            indices = self.df_Pos.index[self.df_Pos['Segment-number'] == segment_number].tolist()
            segment_dfPos = self.df_Pos[self.df_Pos['Segment-number'] == segment_number]

            #extract data for current segment only
            segment_nr_use = distinct_segment_numbers.index(segment_number)

            segment_GPRdata = self.gprDataList[segment_nr_use]
            segment_distVec = self.distVecList[segment_nr_use]
            #segment_GPRdata = self.gprDataList[segment_number-1]
            #segment_distVec = self.distVecList[segment_number-1]


        return segment_GPRdata, segment_distVec, segment_dfPos
    
    def writeMLfile(self, filname = None):
        '''
        Create specific txt file from dataframe.
        Format for columns:
        
        segm-nr    lat    lon    tracenr

        '''
        fname = filname or 'output.txt'
        
        for iseg in range(self.numberOfSegments):
            nsamples,ntraces = self.gprDataList[iseg].shape
            traceNumbers = np.arange(0,ntraces)
            if iseg == 0:
                traceNrList = traceNumbers
            else:
                traceNrList = np.concatenate((traceNrList,traceNumbers))
        columns =['Segment-number','Latitude','Longitude']
        dfNew = self.df_Pos[columns].copy()
        dfNew['TraceNr'] = traceNrList

        dfNew.to_csv(fname, header=False, index=False, sep='\t')

        return dfNew


    def to_filepath(self, filepath):
        # Save the class data to an .npz file
        # Save the DataFrame to a CSV file
        #self.df_Pos.to_csv('df_Pos.csv', index=True)
        
        np.savez(filepath, config=json.dumps(self.config, default=lambda o: o.__dict__), \
                 config_gpr = json.dumps(self.config_gpr, default=lambda o: o.__dict__), \
                 df_Pos=self.df_Pos.to_numpy(), df_Pos_columns= self.df_Pos.columns.to_list(), \
                    **{'distVec_array_{}'.format(i): arr for i, arr in enumerate(self.distVecList)}, \
                    **{'gprProfile_array_{}'.format(i): arr for i, arr in enumerate(self.gprDataList)})

        print('Survey data stored in:')
        print(filepath)

    @staticmethod
    def get_topo_corrected_profile_depth_migrated(profile_migrated, alt_data, dt_ns, resample):

        from src.processing.scripts.gpr.utils.processing_utils import calculate_depthshift, correction_topo_in_depth
        
        dt = dt_ns

        c = 30 # cm/ns - EM wave velocity in the air

        dz = 0.01*c * dt /resample #meter

        nsamps, ntraces = profile_migrated.shape

        #time_ns = np.arange(0.,dt*nsamps, dt)
        depth_vector = np.arange(0., dz*nsamps, dz)


        depth_vector_shifted, depth_shift_max_vector,depth_GPR_to_surface = calculate_depthshift(alt_data,dz,depth_vector)
        profile_shifted = correction_topo_in_depth(depth_vector_shifted,depth_shift_max_vector,profile_migrated,\
                                                   depth_GPR_to_surface,dz)

        return profile_shifted


    def get_TOPO_corrected_profiles_depth_migrated(self,segmentId = None, profile_segm_list = None, \
                                                   dt_ns = None, resample_use = None):
        #todo: need velocity in air...check details
        '''
        Adjust data profile for relative height of flight (from starting location).
        Use gpr flight height (or digital elevation model (DEM)) to correct profile.

        If not stated: Operates on the segments assigned to this object.
        Note: Expect profile(s) to be depth migrated (in meter).

        Returns list of topo-adjusted profiles/segments in depth.
        '''

        from src.processing.scripts.gpr.utils.processing_utils import calculate_depthshift, correction_topo_in_depth

        profile_segms =  profile_segm_list or self.gprDataList

        dt = (self.config.dt if hasattr(self.config, 'dt') else self.config["dt"] if isinstance(self.config, dict) else 100/512) or dt_ns
        resample = (self.config.resample if hasattr(self.config, 'resample') else self.config["resample"] if isinstance(self.config, dict) else 1) or resample_use

        c = 30 # cm/ns - EM wave velocity in the air

        dz = 0.01*c * dt /resample #meter


        alt_data_list = list()
        TWT_vector_list = list()

        nsamps, ntraces = profile_segms[0].shape

        #time_ns = np.arange(0.,dt*nsamps, dt)
        depth_vector = np.arange(0., dz*nsamps, dz)

        distinct_segment_numbers = self.df_Pos['Segment-number'].unique().tolist()

        numberOfSegmentsIncluded = len(distinct_segment_numbers)

        if segmentId == None:
            profile_shifted_list = list()
            #get all segments
            #for iseg in range(self.numberOfSegments):
            for iseg in range(numberOfSegmentsIncluded):
                #slice data-frame
                #indices = self.df_Pos.index[self.df_Pos['Segment-number'] == iseg+1].tolist()
                isegUse = distinct_segment_numbers[iseg]
                segment_dfPos = self.df_Pos[self.df_Pos['Segment-number'] == isegUse]
                #segment_dfPos = self.df_Pos[self.df_Pos['Segment-number'] == iseg+1]
                height_data = segment_dfPos["ALT:Altitude"].values
                #alt_data = segment_dfPos["Altitude RTK"].values
                alt_data = segment_dfPos["Altitude"].values
                alt_data_list.append(alt_data)

                depth_vector_shifted, depth_shift_max_vector = calculate_depthshift(alt_data,dz,depth_vector)
                profile_shifted = correction_topo_in_depth(depth_vector_shifted,depth_shift_max_vector,profile_segms[iseg],\
                                                           alt_data,dz)
                profile_shifted_list.append(profile_shifted)

            return profile_shifted_list, alt_data_list

        else:
            #slice data-frame
            #indices = self.df_Pos.index[self.df_Pos['Segment-number'] == segmentId].tolist()
            segment_dfPos = self.df_Pos[self.df_Pos['Segment-number'] == segmentId]
            alt_data = segment_dfPos["ALT:Altitude"].values

            #calc TWT (TWT_GPR_to_surface_tmp)
            segIdx = distinct_segment_numbers.index(segmentId)
            depth_vector_shifted, depth_shift_max_vector = calculate_depthshift(alt_data,dz,depth_vector)
            profile_shifted = correction_topo_in_depth(depth_vector_shifted,depth_shift_max_vector,profile_segms[segIdx],\
                                                           alt_data,dz)

            return profile_shifted,alt_data

      
    
    def get_TOPO_corrected_profiles(self, segmentId = None, profile_segm_list = None, dt_ns = None, resample_use = None):
        '''
        Adjust data profile for relative height of flight (from starting location).
        Use two-way traveltime (TWT) to correct profile.

        If not stated: Operates on the segments assigned to this object.

        Returns list of topo-adjusted profiles/segments.
        '''

        from src.processing.scripts.gpr.utils.processing_utils import calculate_timeshift,correction_topo

        profile_segms =  profile_segm_list or self.gprDataList

        dt = (self.config.dt if hasattr(self.config, 'dt') else self.config["dt"] if isinstance(self.config, dict) else 100/512) or dt_ns
        resample = (self.config.resample if hasattr(self.config, 'resample') else self.config["resample"] if isinstance(self.config, dict) else 1) or resample_use

        c = 30 # cm/ns - EM wave velocity in the air

        alt_data_list = list()
        TWT_vector_list = list()

        nsamps, ntraces = profile_segms[0].shape

        time_ns = np.arange(0.,dt*nsamps, dt)

        distinct_segment_numbers = self.df_Pos['Segment-number'].unique().tolist()

        numberOfSegmentsIncluded = len(distinct_segment_numbers)

        if segmentId == None:
            profile_shifted_list = list()
            #get all segments
            #for iseg in range(self.numberOfSegments):
            for iseg in range(numberOfSegmentsIncluded):
                #slice data-frame
                #indices = self.df_Pos.index[self.df_Pos['Segment-number'] == iseg+1].tolist()
                isegUse = distinct_segment_numbers[iseg]
                segment_dfPos = self.df_Pos[self.df_Pos['Segment-number'] == isegUse]
                #segment_dfPos = self.df_Pos[self.df_Pos['Segment-number'] == iseg+1]
                height_data = segment_dfPos["ALT:Altitude"].values
                #alt_data = segment_dfPos["Altitude RTK"].values
                alt_data = segment_dfPos["Altitude"].values
                alt_data_list.append(alt_data)

                #calc TWT (TWT_GPR_to_surface_tmp)
                TWT_vector = np.abs(2 * (np.abs(np.max(np.array(alt_data)) - np.array(alt_data))) *100 / c)
                #TWT_vector = np.flip(TWT_vector)
                TWT_vector_list.append(TWT_vector)
                time_vector_shifted,time_shift_max_vector = calculate_timeshift(TWT_vector,dt,resample,time_ns)
                profile_shifted = correction_topo(time_vector_shifted,time_shift_max_vector,profile_segms[iseg],TWT_vector,dt,resample)
                profile_shifted_list.append(profile_shifted)

            return profile_shifted_list, TWT_vector_list, alt_data_list

        else:
            #slice data-frame
            #indices = self.df_Pos.index[self.df_Pos['Segment-number'] == segmentId].tolist()
            segment_dfPos = self.df_Pos[self.df_Pos['Segment-number'] == segmentId]
            alt_data = segment_dfPos["ALT:Altitude"].values

            #calc TWT (TWT_GPR_to_surface_tmp)
            segIdx = distinct_segment_numbers.index(segmentId)
            TWT_vector = np.abs(2 * (np.abs(np.max(np.array(alt_data)) - np.array(alt_data))) *100 / c)
            time_vector_shifted,time_shift_max_vector = calculate_timeshift(TWT_vector,dt,resample,time_ns)
            profile_shifted = correction_topo(time_vector_shifted,time_shift_max_vector,profile_segms[segIdx],TWT_vector,dt,resample)
            #profile_shifted = correction_topo(time_vector_shifted,time_shift_max_vector,profile_segms[segmentId-1],TWT_vector,dt,resample)

            return profile_shifted,TWT_vector,alt_data

        

        

class LoadRawSurveyData(object):
    '''
    Expect survey with naming convention:
    caseDate = "YYYY-MM-DD-HH-MM-SS"

    A survey contains (in addition to mission file *.kml) these files:

    - caseDate-system.log     (Mission log file)
    - caseDate-pos.csv        (Mission position file, altimeter, gps data, trace timestamps, etc...)
    - caseDate-gpr.SGY        (GPR raw data)

    return:
      gpr data split into segments with equal spacing (dx).
      The data is returned as ProjectGPRdata-object for each survey
      that is part of this flight (caseDate)
    '''

    def __init__(self, globalpath: str, caseDate: str) -> None:
        
        #globalpath = 'C:/code.sintef.no/geodrones/GPR_processing/data/'
        self.caseDate = caseDate #"2023-03-01-15-11-52"

        self.log_filename = globalpath + caseDate + '-system.log'
        self.sgy_filename = globalpath + caseDate + '-gpr.SGY'
        self.csv_filename = globalpath + caseDate + '-position.csv'


        self._segments_gpr_data_list = [] #segments_data - REDUNDANT
        self._segments_distanceVec_list = [] #segments_distanceVec - REDUNDANT

        self._list_of_ProjectGPRdata = [] # ProjectGPRdata - object

        self.number_of_surveys_included = 1 #default. should always be one survey included!

        self.trace_spacing = 0.05 #m
        self.trace_time_cut = -1


        #create data-folder
        # - data/project_name/raw_data
        # - data/project_name/processed_data

        #steps

        # 1) Read system file + find indices for waypoints (based date-format, time-stamp)
        # 2) Read position file and find trace-sections (indices) that correspond to current mission (way points)
        # 3) Read position file and extract GPS data and flight height
        # 3a) Map pos-data for each way point segment + calc real distance for each of the segment profiles
        # 4) Read GPR data and cut data according way points for current mission
        # 5) Interpolate data onto equal trace spacing of x m


    def _read_system_file(self, getOnlyGPRconfig = False):
        '''
        synopsis: read system file (.log) and return array of time-stamps for waypoints 
                for a given flight/mission.
                If only getting type of GPR: getOnlyGPRconfig = True (default: False)

        ex-line: [10:02:24.707] Waypoint index set to 0
                [10:02:24.707] [TF] Going to waypoint #1
                ...
                [10:05:10.600] [TF] Going to waypoint #6
                [10:05:10.618] "Change mode from MoveToWaypoint to TurnBeforeMove"
                [10:05:10.619] Waypoint index set to 6
                [10:05:13.099] "Change mode from TurnBeforeMove to MoveToWaypoint"
                [10:06:03.000] Destination reached
                [10:06:03.001] [TF] Last waypoint reached'''
        
        filename = self.log_filename

        waypointsArray = []
        waypointsDateArray = []

        survey_waypoints_list = []
        survey_waypointsDate_list = []

        type_GPR = ""

        number_of_surveys = 1

        survey_cnt = 0

        str_dummy = ""

        with open(filename, 'r') as file:

            if getOnlyGPRconfig==False:
                lines = file.readlines()
                for line in lines:
                    # Splitting the line into timestamp and text
                    parts = line.strip().split('] ')
                    if len(parts) > 1:
                        # Split the line into timestamp and text
                        timestamp, text = line.split(']', 1)
                        
                        # Remove the leading '[' character from the timestamp
                        timestamp = timestamp[1:]
                        
                        # Convert the timestamp to a datetime object
                        timestamp_obj = datetime.datetime.strptime(timestamp, '%H:%M:%S.%f')

                        if "PAYLOADS/RADSYS_ZGPR=true" in text:
                            type_GPR = "RADSYS_ZGPR"
                        elif "PAYLOADS/RADSYS_ZOND=true" in text:
                            type_GPR = "RADSYS_ZOND"
                        
                                                
                        # Check if the text contains "[TF] Going to waypoint #"
                        if "[TF] Going to waypoint #" in text:

                            str_dummy = "[TF] Going to waypoint #"

                            if "[TF] Going to waypoint #1\n" in text:
                                #new survey within same file!
                                survey_cnt = survey_cnt + 1

                                if survey_cnt > 1:

                                    survey_waypoints_list.append(waypointsArray)
                                    survey_waypointsDate_list.append(waypointsDateArray)

                                    #reset for next survey
                                    waypointsArray = []
                                    waypointsDateArray = []

                            
                            waypointsDateArray.append(timestamp_obj)
                            waypointsArray.append(timestamp)
                        
                            # Print the timestamp and text
                            print('Survey #',survey_cnt)
                            print('Timestamp:', timestamp_obj)
                            print('Timestamp:', timestamp)
                            print('Text:', text.strip())

                            

                        if "[TF] Last waypoint reached" in text:

                            str_dummy = "[TF] Last waypoint reached"

                            waypointsDateArray.append(timestamp_obj)
                            waypointsArray.append(timestamp)
                        
                            # Print the timestamp and text
                            print('Survey #',survey_cnt)
                            print('TimesObj:', timestamp_obj)
                            print('Timestamp:', timestamp)
                            print('Text:', text.strip())

                            #checking if there are more surveys included in file!
                            # NB: May not enter "last waypoint"! 
                            #     should check for index number start at 1 again
                            #survey_cnt = survey_cnt + 1                     
                            #survey_waypoints_list.append(waypointsArray)
                            #survey_waypointsDate_list.append(waypointsDateArray)
                            # #reset for next survey
                            #waypointsArray = []
                            #waypointsDateArray = []

                            #return waypointsArray, waypointsDateArray, type_GPR
            
            else:
                lines = file.readlines()
                for line in lines:
                    # Splitting the line into timestamp and text
                    parts = line.strip().split('] ')
                    if len(parts) > 1:
                        # Split the line into timestamp and text
                        timestamp, text = line.split(']', 1)
                        
                        # Remove the leading '[' character from the timestamp
                        timestamp = timestamp[1:]
                        
                        # Convert the timestamp to a datetime object
                        timestamp_obj = datetime.datetime.strptime(timestamp, '%H:%M:%S.%f')

                        if "PAYLOADS/RADSYS_ZGPR=true" in text:
                            type_GPR = "RADSYS_ZGPR"
                        elif "PAYLOADS/RADSYS_ZOND=true" in text:
                            type_GPR = "RADSYS_ZOND"
                            
                        #return waypointsArray, waypointsDateArray, type_GPR
                        return survey_waypoints_list, survey_waypointsDate_list, type_GPR
   
        #if last data entry is last waypoint (some cases it doesn't reach last waypoint)
        #if "[TF] Last waypoint reached" in str_dummy:
        survey_waypoints_list.append(waypointsArray)
        survey_waypointsDate_list.append(waypointsDateArray)


        #return waypointsArray, waypointsDateArray, type_GPR        
        return survey_waypoints_list, survey_waypointsDate_list, type_GPR        
    


    def _read_position_file(self, waypoint_list):
        '''
        Synopsis: Read the position file (.csv), then use the timestamps provided in
                the waypoint_list - array to slice the position data for each
                of the waypoint segments for a flight/mission.

                Note: it finds the closest time (if not matching)
       
        @args:
        - waypoint_list: list of timestamps for each segment (dim: number of segments)

        @returns: 
        - timesArr: timestamps  (list)
        - traces  : trace nr for the waypoints (list)
        - idxArr  : indices for dataframe (csv) for waypoints (list)
        '''
        file_path = self.csv_filename

        if self.typeGPR == 'RADSYS_ZOND':
            gprTypeTxt = 'GPR:Trace'
        else: 
            gprTypeTxt = 'zGPR:Trace'
    
        def find_closest_element(df_in, check_value, gprTypeTxt):
            # Convert check_value to pandas Timestamp for comparison
            check_time = pd.to_datetime(check_value, format='%H:%M:%S.%f')

            df = df_in.copy()

            # Convert the column containing time data to pandas Timestamp
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f')
            #df.loc[:, 'Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f')

            # Calculate the time difference between each element and the check_value
            df['time_diff'] = abs((df['Time'] - pd.Timestamp.min).dt.total_seconds() - (check_time - pd.Timestamp.min).total_seconds())
            #df.loc[:,'time_diff'] = abs((df['Time'] - pd.Timestamp.min).dt.total_seconds() - (check_time - pd.Timestamp.min).total_seconds())

            # Find the index of the element with the minimum time difference
            closest_index = df['time_diff'].idxmin()
            # Get the closest element value
            closest_time = df.loc[closest_index, 'Time']
            closest_trace = df.loc[closest_index, gprTypeTxt]
            #check if closest_trace is a nan (in the new csv file the trace nr occur at interval of 3)
            # if np.isnan(closest_trace):
            #     closest_trace = df.loc[closest_index+1, gprTypeTxt]
            #     if np.isnan(closest_trace):
            #         closest_trace = df.loc[closest_index-1, gprTypeTxt]
            #         if np.isnan(closest_trace):
            #             closest_trace = df.loc[closest_index-2, gprTypeTxt]


            return closest_time, closest_trace, closest_index
        
        #create data frame object of position data
        df = pd.read_csv(file_path)

        #check if csv is old version or new standard
        if 'ALT:ID' in df.columns: #new version
            if self.typeGPR == 'RADSYS_ZOND':
                #gprTypeTxt = 'GPR:Trace'
                # Filter rows where 'zGPR:Traces' has values (non-null and non-zero)
                df_filtered = df[df['GPR:Trace'].notnull() & (df['GPR:Trace'] != 0)]
            else: 
                #gprTypeTxt = 'zGPR:Trace'            
                # Filter rows where 'zGPR:Traces' has values (non-null and non-zero)
                df_filtered = df[df['zGPR:Trace'].notnull() & (df['zGPR:Trace'] != 0)]

                dfUse = df_filtered
        else:
            dfUse = df


        
        timesArr = []
        traces = []
        idxArr = []
        for elem in waypoint_list:
            timestamp, traceidx, idx = find_closest_element(dfUse, elem,gprTypeTxt)
            timesArr.append(timestamp)
            traces.append(int(traceidx))
            idxArr.append(idx)

        return timesArr,traces,idxArr
    

    def _read_segy_file(self,targetTraces, time_cut):
        '''
        Synopsis: read mission gpr data (segy - file). Extract only gpr data for
                the segments defined by mission waypoints, defined by input 
                <targetTraces> array containinig corresponding trace id number.
                If <time_cut> = -1, no cutting in time domain, else only extract
                data samples between 0 - time_cut.

        Returns list of segment gpr data + corresponding list of trace numbers 
        '''
        sgy_filename = self.sgy_filename

        full_raw_profile,nbr_tr,n_samples,samples_interval,dt,time_ns,traces,h,bf,tf = read_segy(sgy_filename)
        
        segments_gpr_data = list()
        segments_gpr_tracenrs = list()
        for i in range(len(targetTraces)-1):
            #cut traces
            trace_start = targetTraces[i]
            trace_end   = targetTraces[i+1]
            index_cut,traces_cut,n_samples,time_ns_cut,nbr_tr,profile_segm = \
                cut_profile(time_cut,dt,trace_start,trace_end,traces,full_raw_profile)
            segments_gpr_data.append(profile_segm)
            segments_gpr_tracenrs.append(traces_cut)

        return segments_gpr_data, segments_gpr_tracenrs, dt #,n_samples

    
    def _calculate_distance_vector(self,gprCoords):
        '''
        Based on gprCoords of segment profile, calculate distance between traces and
        accumulated distance along profile.
        @arg:
        - gprCoord : list of gprCoords for segment profile. 
                     - gprCoords[0][traceIdx]:  latitude for trace = traceIdx
                     - gprCoords[1][traceIdx]:  longitude for trace = traceIdx

        @return
        - distance_vector : accumulated distance in meter from first trace (for each trace)
        - dx_vector       : distance in meter between neighbouring trace

        '''
        nrOfCoords = gprCoords[0].shape[0]
        x_coords = []
        y_coords = []
        dx_vector = []
        distance_vector = []
        distance_vector.append(0.0) #first point
        #for trace_idx, coords in enumerate(gprCoords):
        for trace_idx in range(nrOfCoords):
            #x_coords.append(trace_idx * trace_spacing)
            #y_coords.append([point[2] for point in trace])

            if trace_idx > 0:
                startCoord = (gprCoords[0][trace_idx - 1], gprCoords[1][trace_idx - 1])
                endCoord = (gprCoords[0][trace_idx], gprCoords[1][trace_idx])
                #prev_x = (trace_idx - 1) * trace_spacing
                #prev_y = [point[2] for point in prev_trace]
                distance1 = np.sqrt((gprCoords[0][trace_idx - 1] - gprCoords[0][trace_idx])**2 
                                    + (gprCoords[1][trace_idx - 1] - gprCoords[1][trace_idx])**2)
                #distance_vector.append(distance)
                distanceTemp = distance(startCoord, endCoord).meters
                dx_vector.append(distanceTemp)
                distance_vector.append(distance_vector[trace_idx - 1] + distanceTemp)

        return distance_vector, dx_vector

    
    #interpolate image onto equal trace spacing 
    def _image_interp_2_constant_spacing(self,dataInput, distInput,latInput, lonInput, min_movement=1.02, spacing=1.0):
        '''
        interpolate image onto equal trace spacing
        
        @args:
        - dataInput  : numpy-array, dims(ntimeSamples,nTraces)
        - dist       : vector for distance in meter for each trace. dim(nTraces)
        - latInput   : vector of latitudes. dim(nTraces)
        - lonInput   : vector of longitudes. dim(nTraces)
        - min_movment: Minimum trace spacing. If there is not this much separation, toss the next shot.
                    Set high to keep everything. Default 1.0e-2.
        - spacing    : new spacing distance for image

        @returns:
        - data      : interpolated image of data 
        - dists_new : interpolated distances vector (with the assigned/fixed uniform spacing)
        - lat_new   : interpolated latitudes
        - lon_new   : interpolated longitudes  
        '''

        data = dataInput
        dist = np.array(distInput)

        # eliminate an interpolation error by masking out little movement
        good_vals = np.hstack((np.array([True]), np.diff(dist * 1000.) >= min_movement))

        # Correct the distances to reduce noise
        for i in range(len(dist)):
            if not good_vals[i]:
                dist[i:] = dist[i:] - (dist[i] - dist[i - 1])
        temp_dist = dist[good_vals]

        dists_new = np.arange(np.min(temp_dist),
                            np.max(temp_dist),
                            step=spacing)
                            # step=spacing / 1000.0)
                            
        #interpolate lats and lons
        lat_new = interp1d(temp_dist, latInput[good_vals])(dists_new)
        lon_new = interp1d(temp_dist, lonInput[good_vals])(dists_new)   

        # interp1d can only handle real values
        if data.dtype in [np.complex128]:
            data = interp1d(temp_dist, np.real(data[:, good_vals]))(dists_new) + 1.j * interp1d(temp_dist, np.imag(data[:, good_vals]))(dists_new)
        else:
            data = interp1d(temp_dist, data[:, good_vals])(dists_new)
            

        return data, dists_new, lat_new, lon_new


    #interpolate image and position data onto equal trace spacing 
    def _interp_segments_2_constant_spacing(self,dataInput, dfPosInput, distInput, min_movement=1.0e-2, spacing=1.0):
        '''
        interpolate image onto equal trace spacing.
        interpolate associated position data to equal trace spacing (pos-file -> xxx-position.csv)
        
        @args:
        - dataInput  : numpy-array, dims(ntimeSamples,nTraces)
        - dfPosInput : data-frame of .pos-file data for current segment
        - dist       : vector for distance in meter for each trace. dim(nTraces)
        - min_movment: Minimum trace spacing. If there is not this much separation, toss the next shot.
                    Set high to keep everything. Default 1.0e-2.
        - spacing    : new spacing distance for image

        @returns:
        - data      : interpolated image of data 
        - dfPosInput : data-frame of .pos-file data for current segment
        - dists_new : interpolated distances vector (with the assigned/fixed uniform spacing)
        '''

        data = dataInput
        dist = np.array(distInput)

        # eliminate an interpolation error by masking out little movement
        good_vals = np.hstack((np.array([True]), np.diff(dist * 1000.) >= min_movement))

        # Correct the distances to reduce noise
        for i in range(len(dist)):
            if not good_vals[i]:
                dist[i:] = dist[i:] - (dist[i] - dist[i - 1])
        temp_dist = dist[good_vals]

        dists_new = np.arange(np.min(temp_dist),
                            np.max(temp_dist),
                            step=spacing)
                            # step=spacing / 1000.0)
                            
        #interpolate each column in pos-file (should exclude some...data, time)
        # Index(['Elapsed', 'Date', 'Time', 'Pitch', 'Roll', 'Yaw', 'Latitude',
        #       'Longitude', 'Altitude', 'Velocity', 'RTK Status', 'Latitude RTK',
        #       'Longitude RTK', 'Altitude RTK', 'ALT:Altitude',
        #       'ALT:Filtered Altitude', 'GPR:Trace'],dtype='object')
        columns = dfPosInput.columns
        dfPos_new = pd.DataFrame()
        if len(temp_dist) == len(dists_new) - 1:
            # Repeat the last distance value to match the number of traces
            temp_dist.append(temp_dist[-1])

        for item in columns:
            tmpVals = dfPosInput[item].values
            newVals = interp1d(temp_dist, tmpVals[good_vals])(dists_new)
            dfPos_new[item] = newVals

        # interp1d can only handle real values
        if data.dtype in [np.complex128]:
            data = interp1d(temp_dist, np.real(data[:, good_vals]))(dists_new) + 1.j * interp1d(temp_dist, np.imag(data[:, good_vals]))(dists_new)
        else:
            data = interp1d(temp_dist, data[:, good_vals])(dists_new)
            

        return data, dists_new, dfPos_new


    def create_segments(self, trace_spacing = None, timeCut = None):
        '''
        @args
        - trace_spacing : dx between traces (interpolation). Default: 0.05 m (None)
        - timeCut       : time cut of data profile (in ns). Default: -1, no cutting (None)
        '''

        timeCut = timeCut or -1
        #timeCut = 40e-9 #seconds
        trace_spacing = trace_spacing or 0.05 #m
        segm_spacing = 5.0 #m

        self.trace_spacing = trace_spacing
        self.trace_time_cut = timeCut        
        
        print('======> Creating segments for mission : ' + str(self.caseDate))
        print(' - Segments are created with equal trace spacing of ' + str(trace_spacing) + 'm')
        print('')
        
        print(' --> Finding timestamps for defined waypoints:')
        wayPoints_survey,wayPointsDates_survey, typeGPR = self._read_system_file()
        
        number_of_surveys = len(wayPoints_survey)

        self.number_of_surveys_included = number_of_surveys
        self.typeGPR = typeGPR
        
        for isurvey in range(number_of_surveys):
            
            wayPoints = wayPoints_survey[isurvey]

            # Read position file and find trace-sections (indices) that correspond to current mission (way points)
            missionTimes,missionTraceNr,dfIdxarray = self._read_position_file(wayPoints)
            
            
            #Note-to-self: potentially add segy header info
            segments_gpr_data, segments_gpr_tracenrs, dt = self._read_segy_file(missionTraceNr,timeCut)

            df_posFile = pd.read_csv(self.csv_filename)

            #interpolate each column in pos-file (should exclude some...data, time)
            # Index(['Elapsed', 'Date', 'Time', 'Pitch', 'Roll', 'Yaw', 'Latitude',
            #       'Longitude', 'Altitude', 'Velocity', 'RTK Status', 'Latitude RTK',
            #       'Longitude RTK', 'Altitude RTK', 'ALT:Altitude',
            #       'ALT:Filtered Altitude', 'GPR:Trace'],dtype='object')
            if self.typeGPR == 'RADSYS_ZOND':
                cols_to_drop = ['Elapsed', 'Date', 'Time', 'RTK Status','GPR:Trace']
                df_posFile = df_posFile.drop(columns=[col for col in cols_to_drop if col in df_posFile.columns])
            
                #check for new file format?
                #if 'ALT:ID' in df_posFile.columns: #new version of csv-file

            else: #self.typeGPR = 'RADSYS_ZGPR'
                #df_posFile = df_posFile.drop(columns=['Elapsed', 'Date', 'Time', 'Pitch', 'Roll', 'Yaw','RTK Status','zGPR:Trace'])
                if 'ALT:ID' in df_posFile.columns: #new version of csv-file
                    # Filter rows where 'zGPR:Traces' has values (non-null and non-zero)
                    df_filtered = df_posFile[df_posFile['zGPR:Trace'].notnull() & (df_posFile['zGPR:Trace'] != 0)]

                    df_posFile_Use = df_filtered
                    cols_to_drop = ['Elapsed', 'Date', 'Time', 'RTK Status','ALT:ID','OBS:Sectors1',\
                                                        'OBS:Sectors2','OBS:Sectors3','OBS:ID','zGPR:Trace']
                    df_posFile = df_posFile_Use.drop(columns=[col for col in cols_to_drop if col in df_posFile_Use.columns])
                else:
                    cols_to_drop = ['Elapsed', 'Date', 'Time', 'RTK Status','zGPR:Trace']
                    df_posFile = df_posFile.drop(columns=[col for col in cols_to_drop if col in df_posFile.columns])


            columnsPosFile = df_posFile.columns
            columnsPosFile.to_list()
            columnsPosFile =[columnsPosFile,'Segment-number']
            df_posFile_intp = pd.DataFrame(columns=columnsPosFile)


            segments_pos_df = list()
            segments_pos_df_intp = list()
            
            segments_data = list() #list of segment profiles, len = number of segments
            segments_distanceVec = list()

            for i in range(len(wayPoints)-1):
                rowStart = dfIdxarray[i]
                rowEnd = dfIdxarray[i+1]-1
                dfTemp = df_posFile.loc[rowStart:rowEnd,:]
                segments_pos_df.append(dfTemp)
                
                #use gps coords to calc real distances (should include option to use RTK corrected)
                #latArr = dfTemp['Latitude RTK'].values
                #lonArr = dfTemp['Longitude RTK'].values
                latArr = dfTemp['Latitude'].values
                lonArr = dfTemp['Longitude'].values
                
                #calc distance for segm (starting from zero) for each segment
                distance_vector, dx_vector = self._calculate_distance_vector([latArr,lonArr])

                
                seg_data_intp, seg_dist_vector_intp, dfTemp_intp = self._interp_segments_2_constant_spacing(segments_gpr_data[i],\
                                            dfTemp, distance_vector, 0.01, trace_spacing)
                
                segments_pos_df_intp.append(dfTemp_intp)
                tmpArr = np.ones(int(len(seg_dist_vector_intp))) * (i+1)
                dfTemp_intp['Segment-number'] = tmpArr

                
                if i>0:
                    df_posFile_intp = pd.concat([df_posFile_intp,dfTemp_intp],ignore_index=True)
                else:
                    df_posFile_intp = dfTemp_intp

                segments_data.append(seg_data_intp)

                segments_distanceVec.append(seg_dist_vector_intp)
                

            #self._segments_gpr_data_list = segments_data
            #self._segments_distanceVec_list = segments_distanceVec
            #self._df_Pos = df_posFile_intp

            self.dt = dt

            config = ProcessingConfig(not_processed=True)
            config.dt = self.dt #set time increment for survey

            config_gpr = GPRtypeConfig(self.typeGPR)

            projGPRdata = ProjectGPRdata(config, config_gpr,
                                        df_posFile_intp,  
                                        segments_distanceVec,
                                        segments_data)
            
            self._list_of_ProjectGPRdata.append(projGPRdata)

            print('\n ==> Survey nr ',str(isurvey))
            print('\n --> Waypoints (survey segments):')
            for i in range(len(wayPoints)-1):
                print('Segm nr ' + str(i+1) )
                print('Start - end: ' + wayPoints[i] + ' - ' + wayPoints[i+1])
                print('Number of traces = ' + str(len(segments_distanceVec[i])))
                print('Segment distance = ' + str(segments_distanceVec[i][-1:]))

            print('')


    def create_segments_from_trace_numbers(self, trace_number_list, trace_spacing = None, timeCut = None):
        '''
        create segments based on a list of selected trace-numbers (representing
        the waypoints)

        @args
        - trace_spacing: dx between traces (interpolation). Default: 0.05 m (None)
        - timeCur      : time cut of data profile (in ns). Default: -1, no cutting (None)
        '''

        timeCut = timeCut or -1
        #timeCut = 40e-9 #seconds
        trace_spacing = trace_spacing or 0.05 #m
        segm_spacing = 5.0 #m
        
        
        print('======> Creating segments for mission : ' + str(self.caseDate))
        print(' - Segments are created with equal trace spacing of ' + str(trace_spacing) + 'm')
        print('')
        
        print(' --> Finding timestamps for defined trace numbers (waypoints):')
        
        #find corresponding timestamps for the trace numbers?
        #nb: check if valid trace numbers (len?)

        wayPoints, wayPointsDates, typeGPR = self._read_system_file(getOnlyGPRconfig=True)

        self.typeGPR = typeGPR
        
        # Read position file and find trace-sections (indices) that correspond to current mission (way points)
        #missionTimes,missionTraceNr,dfIdxarray = self._read_position_file(wayPoints)

        df_posFile = pd.read_csv(self.csv_filename)    

        # Find row indices where 'traces' matches traceNumberList
        if self.typeGPR == 'RADSYS_ZOND':
            dfIdxarray = [df_posFile.index[df_posFile['GPR:Trace'] == trace].tolist()[0] for trace in trace_number_list if trace in df_posFile['GPR:Trace'].values]
        else:
            dfIdxarray = [df_posFile.index[df_posFile['zGPR:Trace'] == trace].tolist()[0] for trace in trace_number_list if trace in df_posFile['zGPR:Trace'].values]

        
        #Note-to-self: potentially add segy header info
        missionTraceNr = trace_number_list

        segments_gpr_data, segments_gpr_tracenrs, dt = self._read_segy_file(missionTraceNr,timeCut)


        #interpolate each column in pos-file (should exclude some...data, time)
        # Index(['Elapsed', 'Date', 'Time', 'Pitch', 'Roll', 'Yaw', 'Latitude',
        #       'Longitude', 'Altitude', 'Velocity', 'RTK Status', 'Latitude RTK',
        #       'Longitude RTK', 'Altitude RTK', 'ALT:Altitude',
        #       'ALT:Filtered Altitude', 'GPR:Trace'],dtype='object')
        if self.typeGPR == 'RADSYS_ZOND':
            #df_posFile = df_posFile.drop(columns=['Elapsed', 'Date', 'Time', 'Pitch', 'Roll', 'Yaw','RTK Status','GPR:Trace'])
            df_posFile = df_posFile.drop(columns=['Elapsed', 'Date', 'Time', 'RTK Status','GPR:Trace'])
            
            #check for new file format?
            #if 'ALT:ID' in df_posFile.columns: #new version of csv-file

        else: #self.typeGPR = 'RADSYS_ZGPR'
            #df_posFile = df_posFile.drop(columns=['Elapsed', 'Date', 'Time', 'Pitch', 'Roll', 'Yaw','RTK Status','zGPR:Trace'])
            if 'ALT:ID' in df_posFile.columns: #new version of csv-file
                # Filter rows where 'zGPR:Traces' has values (non-null and non-zero)
                df_filtered = df_posFile[df_posFile['zGPR:Trace'].notnull() & (df_posFile['zGPR:Trace'] != 0)]

                df_posFile_Use = df_filtered
                df_posFile = df_posFile_Use.drop(columns=['Elapsed', 'Date', 'Time', 'RTK Status','ALT:ID','OBS:Sectors1',\
                                                      'OBS:Sectors2','OBS:Sectors3','OBS:ID','zGPR:Trace'])
            else:
                df_posFile = df_posFile.drop(columns=['Elapsed', 'Date', 'Time', 'RTK Status','zGPR:Trace'])


        columnsPosFile = df_posFile.columns
        columnsPosFile.to_list()
        columnsPosFile =[columnsPosFile,'Segment-number']
        df_posFile_intp = pd.DataFrame(columns=columnsPosFile)


        segments_pos_df = list()
        segments_pos_df_intp = list()
        
        segments_data = list() #list of segment profiles, len = number of segments
        segments_distanceVec = list()

        for i in range(len(trace_number_list)-1):
            rowStart = dfIdxarray[i]
            rowEnd = dfIdxarray[i+1]-1
            dfTemp = df_posFile.loc[rowStart:rowEnd,:]
            segments_pos_df.append(dfTemp)
            
            #use gps coords to calc real distances (should include option to use RTK corrected)
            #latArr = dfTemp['Latitude RTK'].values
            #lonArr = dfTemp['Longitude RTK'].values
            latArr = dfTemp['Latitude'].values
            lonArr = dfTemp['Longitude'].values
            
            #calc distance for segm (starting from zero) for each segment
            distance_vector, dx_vector = self._calculate_distance_vector([latArr,lonArr])


            seg_data_intp, seg_dist_vector_intp, dfTemp_intp = self._interp_segments_2_constant_spacing(segments_gpr_data[i],\
                                        dfTemp, distance_vector, 0.01, trace_spacing)
            
            segments_pos_df_intp.append(dfTemp_intp)
            tmpArr = np.ones(int(len(seg_dist_vector_intp))) * (i+1)
            dfTemp_intp['Segment-number'] = tmpArr

            
            if i>0:
                df_posFile_intp = pd.concat([df_posFile_intp,dfTemp_intp],ignore_index=True)
            else:
                df_posFile_intp = dfTemp_intp

            segments_data.append(seg_data_intp)

            segments_distanceVec.append(seg_dist_vector_intp)
            

        #self._segments_gpr_data_list = segments_data
        #self._segments_distanceVec_list = segments_distanceVec
        #self._df_Pos = df_posFile_intp

        self.dt = dt
        
        config = ProcessingConfig(not_processed=True)
        config.dt = self.dt #set time increment for survey

        config_gpr = GPRtypeConfig(self.typeGPR)

        projGPRdata = ProjectGPRdata(config, config_gpr,
                                    df_posFile_intp,  
                                    segments_distanceVec,
                                    segments_data)
        
        self._list_of_ProjectGPRdata.append(projGPRdata)
        
        print('\n --> Waypoints (survey segments):')
        for i in range(len(trace_number_list)-1):
            print('Segm nr ' + str(i+1) )
            print('Start - end (trace number): ' + str(trace_number_list[i]) + ' - ' + str(trace_number_list[i+1]))
            print('Number of traces = ' + str(len(segments_distanceVec[i])))
            print('Segment distance = ' + str(segments_distanceVec[i][-1:]))

        print('')

    def get_number_of_surveys_in_file(self):
        return self.number_of_surveys_included
    
    
    def get_project_data(self, idx=0)->ProjectGPRdata:
        '''
        get raw data represented as ProjectGPRdata object

        list is zero-based (i.e. idx starts from 0)

        Default: only one survey included in raw file
        '''
        
        # config = ProcessingConfig(not_processed=True)
        # config.dt = self.dt #set time increment for survey

        # config_gpr = GPRtypeConfig(self.typeGPR)

        # projGPRdata = ProjectGPRdata(config, config_gpr,
        #                              self._df_Pos,  
        #                              self._segments_distanceVec_list,
        #                              self._segments_gpr_data_list)

        if idx >= self.number_of_surveys_included:
            print("Error: invalid index: idx = ",str(idx))
            print("Number of surveys in file is ",str(self.number_of_surveys_included))
        else:         
            return self._list_of_ProjectGPRdata[idx]

