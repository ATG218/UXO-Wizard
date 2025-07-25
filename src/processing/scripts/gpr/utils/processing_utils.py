import numpy as np
import sys
import scipy.signal as sig
from scipy.fft import fft, ifft

#%% butter bandpass
def butter_bandpass(lowcut, highcut, fs, order):
    
    nyq =  fs/2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    
    return b, a

def filter_butter(profile,n_samples,resample,nbr_tr,a,b):
    
    profile_Butter=np.zeros((n_samples*resample,nbr_tr))

    for k in range(0, nbr_tr):
        profile_Butter[:,k] = sig.filtfilt(b, a, profile[:,k])
    
    return profile_Butter

#%% Reading segy file (Radsys)
def read_segy(sgy_filename):
    
    from obspy.io.segy.segy import iread_segy

    traces = []
    tf = []
    bf = []
    h = []
    for tr in iread_segy(sgy_filename):
        tf.append(tr.stats.segy.textual_file_header)
        bf.append(tr.stats.segy.binary_file_header)
        h.append(tr.stats.segy.trace_header)
        traces.append(tr.data)
    
    full_raw_profile = np.transpose(traces)
    nbr_tr = len(traces)
    n_samples = bf[0].number_of_samples_per_data_trace
    samples_interval = bf[0].sample_interval_in_microseconds
    dt = samples_interval * 0.001
    time_ns = [(i * dt) for i in range(0, n_samples)]
    print('Number of traces:', np.shape(full_raw_profile)[1], ' Time range (ns):', np.max(time_ns))
    
    return full_raw_profile,nbr_tr,n_samples,samples_interval,dt,time_ns,traces,h,bf,tf

#%% Cut profile in time and traces

def cut_profile(time_cut,dt,trace_start,trace_end,traces,profile):

    
    if time_cut < 0.0:
        n_samples, n_traces  = profile.shape
        index_cut = n_samples-1
    else:
        n_samples, n_traces  = profile.shape
        if time_cut > dt*n_samples:
            index_cut = n_samples-1 #last element
        else:
            index_cut=int(time_cut/dt)
            n_samples=index_cut


    profile_cut=profile[0:index_cut,trace_start:trace_end]
    traces_np=np.array(traces)
    traces_cut=traces_np[trace_start:trace_end,0:index_cut]
    
    time_ns_cut = [(i * dt) for i in range(0, index_cut)]
    nbr_tr=trace_end-trace_start

    return index_cut,traces_cut,n_samples,time_ns_cut,nbr_tr,profile_cut

#%% Resample traces

def resample_trace(trace_raw,n_samples,resample,dt):
    

## Require scipy version 1.7 !! resample_poly does not work with 1.8.1 version ==> check new version
    trace_resampled = sig.resample_poly(trace_raw, n_samples*resample, n_samples)
    time_ns_resampled = [(i * dt/resample) for i in range(0, len(trace_resampled))]
    
    return trace_resampled,time_ns_resampled 

def resample_all_traces(n_samples,resample,nbr_tr,traces_cut):
    

    tr_res_all_traces=np.zeros((n_samples*resample,nbr_tr))

    stack_tr = np.zeros(n_samples*resample)
    for k in range(0, nbr_tr):
        tr_k = traces_cut[k]
        tr_res = sig.resample_poly(tr_k, n_samples*resample, n_samples)
        tr_res_all_traces[:,k] =tr_res
        
    return tr_res_all_traces

#%% FFT

def fft_one_trace(trace):
       
    trace_ft = fft(trace)
    amp_trace = np.abs(trace_ft)
    return trace_ft,amp_trace

def fft_all_profile(profile,n_samples,resample,nbr_tr):
    
    profile_ft = np.zeros((n_samples*resample,nbr_tr),dtype=complex)
    amp_profile = np.zeros((n_samples*resample,nbr_tr))

    for k in range(0, nbr_tr):
        profile_ft[:,k] = fft(profile[:,k])
        amp_profile[:,k] = np.abs(profile_ft[:,k])
        
    return profile_ft, amp_profile

#%% Background removal

def background_removal(profile,n_samples,resample,nbr_tr,trace_number):
    
    profile_BR=np.zeros((n_samples*resample,nbr_tr))
    stack_tr = np.zeros(n_samples*resample)
    
    for k in range(0, nbr_tr):
        for m in range(0, n_samples*resample):
            stack_tr[m] = stack_tr[m] + profile[m,k]
    stack_traces = stack_tr/nbr_tr
    trace_BR = profile[:,trace_number] - stack_traces # Background trace

    for k in range(0, nbr_tr):
        profile_BR[:,k] = profile[:,k] - stack_traces # Background trace
    
    return profile_BR,trace_BR,stack_traces


#%% Calculate two-way traveltime in the air depending on relative altitude or drone height
def calculate_TWT(alt_data,c):
    
    import numpy as np

    # TWT_GPR_to_surface = []
    # for i in range(0,len(alt_data)):
    #     TWT_GPR_to_surface_tmp = np.abs(2 * alt_data[i] *100 / c)
    #     TWT_GPR_to_surface.append(TWT_GPR_to_surface_tmp)
        
    TWT_GPR_to_surface = []
    for i in range(0,len(alt_data)):
        TWT_GPR_to_surface_tmp = np.abs(2 * (np.abs(np.max(np.array(alt_data[i])) - np.array(alt_data[i]))) *100 / c)
        TWT_GPR_to_surface.append(TWT_GPR_to_surface_tmp)

    return TWT_GPR_to_surface

#%% Calculate depth in the air depending on relative altitude or drone height
def calculate_relative_depth(altitude_vector):
    
    import numpy as np

    depth_GPR_to_surface = []
    for i in range(0,len(altitude_vector)):
        depth_GPR_to_surface_tmp = np.abs((np.abs(np.max(np.array(altitude_vector[i])) - np.array(altitude_vector[i]))))
        depth_GPR_to_surface.append(depth_GPR_to_surface_tmp)

    return depth_GPR_to_surface

#%% Calculate maximum time shift and create new time vector
def calculate_timeshift(TWT_vector,dt,resample,time_ns):
    
    import numpy as np
    
    ind_trace_max_time_shift = np.argmax(TWT_vector)
    dt = dt/resample

    time_shift_max_vector = np.zeros((round(TWT_vector[ind_trace_max_time_shift]/dt))+1)
    for i in range(0,round(TWT_vector[ind_trace_max_time_shift]/dt)+1):
        time_shift_max_vector[i] = i * dt

    time_vector_shifted = np.concatenate([time_shift_max_vector, np.array(time_ns)+np.max(time_shift_max_vector)+dt])

    return time_vector_shifted,time_shift_max_vector

#%% Calculate maximum depth shift and create new depth vector
def calculate_depthshift(altitude_vector,dz,depth_vector_m):
    
    import numpy as np

    depth_GPR_to_surface_mod = []
    max_altitude = np.max(np.array(altitude_vector))
    for i in range(0,len(altitude_vector)):
        depth_GPR_to_surface_tmp = np.abs(max_altitude - np.array(altitude_vector[i]))
        depth_GPR_to_surface_mod.append(depth_GPR_to_surface_tmp)

    depth_GPR_to_surface = depth_GPR_to_surface_mod
    #depth_GPR_to_surface = altitude_vector

    #ind_trace_depth_shift = np.argmax(altitude_vector)
    ind_trace_depth_shift = np.argmax(depth_GPR_to_surface)
    

    #depth_shift_max_vector = np.zeros((round(altitude_vector[ind_trace_depth_shift]/dz))+1)
    #len_array = (round(altitude_vector[ind_trace_depth_shift]/dz))+1
    len_array = (round(depth_GPR_to_surface[ind_trace_depth_shift]/dz))+1
    depth_shift_max_vector = np.zeros(len_array)
    for i in range(0,len_array):
        depth_shift_max_vector[i] = i * dz

    depth_vector_shifted = np.concatenate([depth_shift_max_vector, np.array(depth_vector_m)+np.max(depth_shift_max_vector)+dz])

    return depth_vector_shifted, depth_shift_max_vector,depth_GPR_to_surface


#%% Topography correction Shift all GPR traces with time shift corresponding to relative drone altitude
def correction_topo(time_vector_shifted,time_shift_max_vector,Profile_after_processing,TWT_vector,dt,resample):

    import numpy as np

    Profile_shifted = np.zeros((time_vector_shifted.shape[0],Profile_after_processing.shape[1]))
    dt = dt/resample
    
    for i in range(0, Profile_after_processing.shape[1]):
        one_trace = Profile_after_processing[:,i]
        time_shift_vector = np.zeros((round(TWT_vector[i]/dt))+1)
    
        for j in range(0,round(TWT_vector[i]/dt)+1):
            time_shift_vector[j] = j * dt
    
        if time_shift_vector.shape[0] == time_shift_max_vector.shape[0]:
            shifted_trace = np.pad(one_trace, (round(TWT_vector[i]/dt)+1,0), \
                               mode='constant',constant_values=0)
        elif time_shift_vector.shape[0] < time_shift_max_vector.shape[0]:
            shifted_trace = np.pad(one_trace, \
                        (round(TWT_vector[i]/dt)+1,time_shift_max_vector.shape[0] - (round(TWT_vector[i]/dt)+1)), \
                               mode='constant',constant_values=0)
    
        Profile_shifted[:,i] = shifted_trace
    
    return Profile_shifted

#%% Topography correction Shift all GPR traces with time shift corresponding to relative drone altitude
def correction_topo_in_depth(depth_vector_shifted,depth_shift_max_vector,Profile_after_processing, rel_altitude_vector,dz):

    import numpy as np

    Profile_shifted = np.zeros((depth_vector_shifted.shape[0],Profile_after_processing.shape[1]))
    
    for i in range(0, Profile_after_processing.shape[1]):
        one_trace = Profile_after_processing[:,i]
        depth_shift_vector = np.zeros((round(rel_altitude_vector[i]/dz))+1)
    
        for j in range(0,round(rel_altitude_vector[i]/dz)+1):
            depth_shift_vector[j] = j * dz
    
        if depth_shift_vector.shape[0] == depth_shift_max_vector.shape[0]:
            shifted_trace = np.pad(one_trace, (round(rel_altitude_vector[i]/dz)+1,0), \
                               mode='constant',constant_values=0)
        elif depth_shift_vector.shape[0] < depth_shift_max_vector.shape[0]:
            shifted_trace = np.pad(one_trace, \
                        (round(rel_altitude_vector[i]/dz)+1,depth_shift_max_vector.shape[0] - (round(rel_altitude_vector[i]/dz)+1)), \
                               mode='constant',constant_values=0)
    
        Profile_shifted[:,i] = shifted_trace
    
    return Profile_shifted    

#%% permittivity loop
def permittivity_loop(E0, R, perm_min, perm_max, increment=0.5, output_condition_threshold=0.5):
    '''
    return E_ref : permittivity of the bottom layer
    
    --- Parameters 
    E0 : permittivity of the top layer (E_air = 1)
    R = reflectivity coefficient computed from spectral amplitudes
    [perm_min, perm_max] : [1,4] for snow or ice, [4,80] for ground
    increment : The default is 0.5.
    output_condition_threshold : The default is 0.5.
    
    '''
    

    
    print('choosed reflectivity coefficient for the loop =', R)
    if R > 1 : 
        print('R should be < 1, but here R =', R,', it is thus not correct')
        sys.exit()

    perm_ref = []
    if perm_max < 10 :
        perm_ref.append(1)
    else :
        perm_ref.append(4)

    error = []

    for i in range(0, 500):
        # print(i)
        
        ## breaking condition for the tested permittivity
        if perm_ref[i] > perm_max or perm_ref[i] < perm_min :
            print('the permittivity should be between', perm_min, 'and', perm_max, 'but here it is =', perm_ref[i])
            break
        
        ## reflectivity coefficient computed from permittivities
        R_perm_ref = abs((np.sqrt(E0) - np.sqrt(perm_ref[i]))/(np.sqrt(E0) + np.sqrt(perm_ref[i])))
        
        ## computation of the misfit
        error.append(abs(1 - (R_perm_ref / R)) * 100)
        min_error = min(error)
        i_min = i
        if error[i] < output_condition_threshold :
            break
        
        ## increment changes when error is greater than min_error
        ## when error is lower than or equal to the min_error we continue with the same increment
        elif increment == 0.5 :
            if error[i] > error[np.argmin(error)] :
               increment = +0.1
               i_min = np.argmin(error)
        elif increment == 0.1 :
            if error[i] > error[np.argmin(error)] :
              increment = -0.1
              i_min = np.argmin(error)
        elif increment == -0.1 :
            if error[i] > error[np.argmin(error)] :
              increment = +0.01
              i_min = np.argmin(error)
        elif increment == +0.01 :
            if error[i] > error[np.argmin(error)] :
              increment = -0.01
              i_min = np.argmin(error)
        perm_ref.append(perm_ref[i_min] + increment)
        
    ## if the output condition is not reached, print the found minimal value
    E_ref = float(format(perm_ref[np.argmin(error)], '.2f'))
    print('the most accurate permittivity for the trace is', E_ref, 'with an error between the two reflectivities ratio of', error[np.argmin(error)], '%')
    
    return E_ref
    
#%%
def gain_r(v, t):
    '''
    return gain
    
    --- Parameters 
    v : constant velocity (cm/ns)
    t : array of time (ns)
    
    '''
    
    gain = []
    for i in range(0, len(t)):
        gain.append(v * t[i]) # cm
        
    return gain
    
#%%
def attenuation_r(v, t):
    '''
    return attenuation
    
    --- Parameters 
    v : constant velocity (cm/ns)
    t : array of time (ns)
    
    '''
    
    attenuation = []
    for i in range(0, len(t)):
        if i == 0 :
            attenuation.append(1/(v * 0.01))
        else :
            attenuation.append(1/(v * t[i]))
    
    return attenuation
    
#%%
def picking_reflection(trace, traceg, sample_reflection):
    '''
    return reflection, reflection_full

    --- Parameters
    trace : array, trace where the reflections have been initially picked
    traceg : array, processed trace (with gain for example) 
    sample_reflection : float or integer, sample picked on trace
    
    '''
    
    import numpy as np

    sref = int(sample_reflection)
    
    index0p = []
    index0n = []
    
    for i in range(0, sref):
        if trace[sref + i] > 0 and trace[sref + i + 1] < 0 or trace[sref + i] < 0 and trace[sref + i + 1] > 0 :
            index0p.append(i)
        if trace[sref - i] > 0 and trace[sref - i - 1] < 0 or trace[sref - i] < 0 and trace[sref - i - 1] > 0 :
            index0n.append(i)
        if len(index0p) >= 2 and len(index0n) >= 2 :
            break

    reflection = traceg[sref - index0n[1] : sref + index0p[1] +1]
    reflection_full = np.zeros(len(trace))
    reflection_full[sref - index0n[1] : sref + index0p[1] + 1] = reflection

    return reflection, reflection_full

#%% TEST


def apply_geometric_spreading_2(data, distance, ref_distance=1.0):
    """
    Apply geometric spreading correction to seismic data.

    Parameters:
    - data: 2D numpy array, seismic profile
    - distance: 1D numpy array, distance values corresponding to traces
    - ref_distance: reference distance for normalization (default is 1.0)

    Returns:
    - Corrected seismic profile
    """
    spreading_factor = (ref_distance / distance) ** 2
    return data * spreading_factor[:, np.newaxis]

def apply_geometric_spreading(data, time, alpha=2, te = 220, tcst = 20):
    """
    Apply intrinsic attenuation correction to seismic data.
    From 0ns to t=tcst(ns) the power gain is set equal to the gain at tcst(ns), 
    i.e., xg(tcst) (constant value, tcst = 20). The gain is only applied up to te = 220ns.

    Parameters:
    - data: 2D numpy array, seismic profile
    - time: 1D numpy array, time values corresponding to samples (in ns)
    - alpha: Exponential order for intrinsic attenuation correction
    - te: Gain applied until time te (in ns)
    - tcst: Time where gain is selected (in ns)

    Returns:
    - Corrected seismic profile
    """
    # Apply constant power gain up to tcst
    gain_constant = time <= tcst
    gain_constant_factor = np.ones_like(time)
    gain_constant_factor[gain_constant] = tcst**alpha

    # Apply exponential gain from tcst to te
    gain_exponential = (time > tcst) & (time <= te)
    gain_exponential_factor = (time[gain_exponential] - tcst)**alpha

    # Combine the constant and exponential gain factors
    gain_factor = np.ones_like(time)
    gain_factor[gain_constant] = gain_constant_factor[gain_constant]
    gain_factor[gain_exponential] = gain_exponential_factor

    #return data * gain_factor[np.newaxis, :]
    return data * gain_factor[:,np.newaxis]


def apply_geometric_spreading3(data, time, alfa=2, te = 220, tcst = 20):
    """
    Apply intrinsic attenuation correction to seismic data.
    From 0ns to t=tcst(ns) the power gain is set equal to the gain at tcst(ns), 
    i.e., xg(tcst) (constant value, tcst = 20). The gain is only applied up to te = 220ns.

    Parameters:
    - data: 2D numpy array, seismic profile
    - time: 1D numpy array, time values corresponding to samples
    - intrinsic_attenuation_coeff: coefficient for intrinsic attenuation correction

    Returns:
    - Corrected seismic profile
    """
    intrinsic_attenuation_factor = time**alfa
    return data * intrinsic_attenuation_factor[np.newaxis, :]


def apply_intrinsic_attenuation2(data, time, intrinsic_attenuation_coeff=0.002, t0 = 0, te = 60):
    """
    Apply intrinsic attenuation correction to seismic data.
    Ideally, the parameter α in the exponential gain should be close to the slope 
    of the log amplitude decrease. This slope could be estimated by fitting a straight 
    line to the amplitude decrease. After some trials, we apply the exponential gain 
    only between t0(ns) and te(ns). (From RGPR)

    Parameters:
    - data: 2D numpy array, seismic profile
    - time: 1D numpy array, time values corresponding to samples
    - intrinsic_attenuation_coeff: coefficient for intrinsic attenuation correction

    Returns:
    - Corrected seismic profile
    """
    intrinsic_attenuation_factor = np.exp(intrinsic_attenuation_coeff * time)
    return data * intrinsic_attenuation_factor[np.newaxis, :]


def apply_intrinsic_attenuation(data, time, intrinsic_attenuation_coeff=0.002, t0=0, te=60):
    """
    Apply intrinsic attenuation correction to seismic data.
    Ideally, the parameter α in the exponential gain should be close to the slope
    of the log amplitude decrease. This slope could be estimated by fitting a straight
    line to the amplitude decrease. After some trials, we apply the exponential gain
    only between t0(ns) and te(ns).

    Parameters:
    - data: 2D numpy array, seismic profile
    - time: 1D numpy array, time values corresponding to samples
    - intrinsic_attenuation_coeff: coefficient for intrinsic attenuation correction
    - t0: Start time for applying exponential gain (in ns)
    - te: End time for applying exponential gain (in ns)

    Returns:
    - Corrected seismic profile
    """
    # Apply exponential gain only between t0 and te
    exponential_gain = (time >= t0) & (time <= te)
    exponential_gain_factor = np.exp(intrinsic_attenuation_coeff * (time[exponential_gain] - t0))

    # Combine the exponential gain factor with a constant factor for times outside t0 and te
    gain_factor = np.ones_like(time)
    gain_factor[exponential_gain] = exponential_gain_factor

    #return data * gain_factor[np.newaxis, :]
    return data #* gain_factor[:,np.newaxis]


