import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr, PackageNotInstalledError
from rpy2.rinterface_lib.embedded import RRuntimeError
from pathlib import Path
import sys, traceback

from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
from scipy.signal import medfilt2d

from scipy.ndimage import median_filter #generic_filter

from src.processing.scripts.gpr.utils.gpr_configs import *

class GPRdataSegmentProcessor(object):

    def __init__(self, process_config=None, gpr_config=None) -> None:

        # ---------- 1. R handle & the NumPy‑friendly converter ----------
        self.r = ro.r
        self._np_converter = default_converter + numpy2ri.converter   # use later

        # ---------- 2. Load RGPR with *no NumPy rules* active ----------
        try:
            with default_converter.context():     # <‑‑ only base R ↔ Python rules
                # convert=False keeps importr from walking every symbol
                self.rgpr = importr("RGPR")
        except (PackageNotInstalledError, Exception) as e:
            # Nice diagnostics — Python part …
            print("Python caught:", type(e).__name__, e, file=sys.stderr)
            traceback.print_exc()

            # …and R part (works only immediately after failure)
            try:
                ro.r("traceback()")
                print(ro.r("geterrmessage()")[0], file=sys.stderr)
            except Exception:
                pass

            raise RuntimeError("Failed to initialize GPR processor") from e

        # ---------- 3. Helper R script path ----------
        self._rgpr_functions_path = Path(__file__).resolve().with_name("RGPR_functions.R")
        if not self._rgpr_functions_path.exists():
            raise FileNotFoundError(f"RGPR_functions.R not found at {self._rgpr_functions_path}")

        # ---------- 4. Read configs ----------
        self.config     = process_config or ProcessingConfig()
        self.config_gpr = gpr_config     or GPRtypeConfig()

        # attribute vs. dict access covers both dataclass & plain‑dict configs
        if isinstance(self.config_gpr, dict):
            self.TxRx = self.config_gpr.get("TxRx")
            self.freq = self.config_gpr.get("freq_cent")
        else:
            self.TxRx = getattr(self.config_gpr, "TxRx", None)
            self.freq = getattr(self.config_gpr, "freq_cent", None)

        self.resample = self.config.resample
        self.dt       = self.config.dt
      
    def _source_rgpr(self):
        """Source RGPR helper functions via absolute path."""
        self.r.source(str(self._rgpr_functions_path))

    def _build_rgpr_data(self, profile, freq, dt, resample, txrx):
        # Source the RGPR_func.R file
        self._source_rgpr()

        # Call the build_RGPR_data function
        with localconverter(self._np_converter):
            return self.r.build_RGPR_data(profile, freq, dt, resample, txrx)

    def _build_rgpr_mixed_phased_deconv(self, profileR,tWin_start, tWin_end, wtr, mu, f1, f2):
        # Source the RGPR_func.R file
        self._source_rgpr()

        # Call the build_RGPR_data function
        with localconverter(self._np_converter):
            return self.r.process_mixed_phased_deconv_R(profileR,tWin_start, tWin_end, wtr, mu, f1, f2)

    
    def _process_resample(self, segmentGPRdata, config):
        
        resample = config.resample or self.config.resample

        self.resample = resample

        from processing_utils import resample_all_traces
        nsamps, ntraces = segmentGPRdata.shape

        #resample_all_traces(n_samples,resample,nbr_tr,traces_cut)
        profile_resampled = resample_all_traces(nsamps,resample,ntraces,segmentGPRdata.T)
        
        return profile_resampled
    

    def _process_DC_shift_RGPR(self, segmentGPRdata, config):

        config = config or self.config.dc_shift_correction

        dc_sample_start = config.DCsample_start
        dc_sample_end   = config.DCsample_end

        #convert to R (not sure if necessary)
        profile_R = self._build_rgpr_data(segmentGPRdata, self.freq, self.dt, self.resample, self.TxRx)
        
        if 0:
            #Use the samples between DC_sample_start and DC_sample_end to estimate DC shift
            z_dc_corrected = self.rgpr.dcshift(profile_R, u=ro.IntVector(range(dc_sample_start, dc_sample_end + 1)))    
            profile_dc_corrected = self.r.slot(z_dc_corrected,"data")        
            #profile_dc_corrected = z_dc_corrected.rx2("data")

        self._source_rgpr()
        with localconverter(self._np_converter):
            profile_dc_corrected = self.r.process_DCshift_R(profile_R, dc_sample_start, dc_sample_end)

        return profile_dc_corrected


    def _process_t0_correction_2(self, segmentGPRdata,config=None):
        '''
        first break estimation required?
        '''
        
        #convert to R (not sure if necessary)
        profile_R = self._build_rgpr_data(segmentGPRdata, self.freq, self.dt, self.resample, self.TxRx)

        threshold_t0 = config.threshold_t0 #unit ns? check

        print("    >> First break estimation + t0 correction")
        print("       threshold_t0 = " + str(threshold_t0))

        # Import the R mean function
        mean_function = ro.r['mean']
        #first break
        #profile
        
        tr_res_with_t0_all_traces_R = self.rgpr.estimateTime0(profile_R, method="threshold", thr=threshold_t0, FUN= mean_function)
        #tr_res_with_t0_all_traces_R = self.rgpr.estimateTime0(profile_R, **{'method':'threshold', 'thr':threshold_t0,\
        #                                                                     'FUN': mean_function})
        #estimateTime0(x, w = 20, method = "coppens", thr = 0.05, FUN = mean)

        with localconverter(self._np_converter):
            profile_t0_corrected = self.r.slot(tr_res_with_t0_all_traces_R, "data")
        #profile_t0_corrected = tr_res_with_t0_all_traces_R.rx2("data")

        #Resize the new vectors and matrices to match original size (by adding zeros at the end)
        profile_t0=np.pad(profile_t0_corrected,
                        [(0,np.shape(segmentGPRdata)[0]-np.shape(profile_t0_corrected)[0]),(0,0)],
                        mode='constant',constant_values=0)

        return profile_t0
    

    def _process_t0_correction_rgpr(self, segmentGPRdata,config=None):
        '''
        first break estimation + t0 correction using RGPR
        '''

        config = config or self.config.t0_correction

        #convert to R (not sure if necessary)
        profile_R = self._build_rgpr_data(segmentGPRdata, self.freq, self.dt, self.resample, self.TxRx)

        threshold_t0 = config.threshold_t0 #unit ns? check

        print("    >> First break estimation and t0-correction (RGPR)")
        print("       threshold_t0 = " + str(threshold_t0))

        self._source_rgpr()

        with localconverter(self._np_converter):
            profile_t0_corrected = self.r.process_t0correction_R(profile_R, threshold_t0)

        #Resize the new vectors and matrices to match original size (by adding zeros at the end)
        if 0:
            profile_t0=np.pad(profile_t0_corrected,
                            [(0,np.shape(segmentGPRdata)[0]-np.shape(profile_t0_corrected)[0]),(0,0)],
                            mode='constant',constant_values=0)
        else:
            profile_t0 = profile_t0_corrected

        return profile_t0


    def _process_t0_correction(self, segmentGPRdata,config=None):
        '''
        first break estimation required?
        '''

        config = config or self.config.t0_correction

        #convert to R (not sure if necessary)
        profile_R = self._build_rgpr_data(segmentGPRdata, self.freq, self.dt, self.resample, self.TxRx)

        threshold_t0 = config.threshold_t0 #unit ns? check

        print("    >> a) First break estimation")
        print("       threshold_t0 = " + str(threshold_t0))

        #first break
        #profile
        tfb_tr_res_all_traces = self.rgpr.firstBreak(profile_R, method="threshold", thr=threshold_t0)
        
        # Set the result as the time0 attribute
        #self.rgpr.time0(profile_R, tfb_tr_res_all_traces)
        self.rgpr.time0(profile_R) <- tfb_tr_res_all_traces
        #profile_R = self.rgpr.setTime0(profile_R, tfb_tr_res_all_traces)

        print("    >> b) t0-correction step")

        # t0-correction
        tr_res_with_t0_all_traces_R = self.rgpr.time0Cor(profile_R, method = "pchip")
        
        with localconverter(self._np_converter):
            profile_t0_corrected = self.r.slot(tr_res_with_t0_all_traces_R, "data")
        #profile_t0_corrected = tr_res_with_t0_all_traces_R.rx2("data")

        #Resize the new vectors and matrices to match original size (by adding zeros at the end)
        profile_t0=np.pad(profile_t0_corrected,
                        [(0,np.shape(segmentGPRdata)[0]-np.shape(profile_t0_corrected)[0]),(0,0)],
                        mode='constant',constant_values=0)

        return profile_t0
    

    def _process_background_removal(self, segmentGPRdata,config=None):
        '''mean trace'''

        nsamp, ntraces = segmentGPRdata.shape
        mean_trace = np.zeros(nsamp)
        profile_bm_removal = np.zeros((nsamp,ntraces))

        for itrace in range(ntraces):
            mean_trace += segmentGPRdata[:,itrace]

        mean_trace = mean_trace/(ntraces)

        for itrace in range(ntraces):
            profile_bm_removal[:,itrace] = segmentGPRdata[:,itrace] - mean_trace

        return profile_bm_removal



    def _process_background_removal_rgpr(self, segmentGPRdata,config=None):
        
        config = config or self.config.background_removal
        #return segmentGPRdata_Processed_background_rem

        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object

        if config.method == "rmBackgroundMatrix":
            Profile_background_sub_R = self.rgpr.rmBackgroundMatrix(Profile_R, width = 21, trim = 0.2, s = 1, eps = 1, itmax = 5)
            with localconverter(self._np_converter):
                Profile_background_sub = self.r.slot(Profile_background_sub_R,"data")
            #Profile_background_sub = Profile_background_sub_R.rx2("data")
            profile_background_sub = segmentGPRdata - Profile_background_sub
        else: #mean trace
            #mean_trace_R = self.rgpr.traceStat(Profile_R, FUN = "median")
            #mean_trace = self.r.slot(mean_trace_R,"data")            
            self._source_rgpr()
            with localconverter(self._np_converter):
                mean_trace = self.r.process_background_removal_R(Profile_R)

            profile_background_sub = segmentGPRdata - mean_trace
            

    
        return profile_background_sub


    def _process_lowpass_filter(self, segmentGPRdata, config=None):
        '''
        Low-pass filter using Butterworth filter
        '''
        config = config or self.config.low_pass_filter

        # Design a Butterworth low-pass filter
        low_cutoff_frequency = config.low_cutoff_frequency * 1e9 #Hz
        order = config.poly_order

        fs = 1/(self.dt*1e-9) #sample frequency Hz
        nyq = fs/2
        #high = high_cutoff_frequency / nyq
        low = low_cutoff_frequency
    
        b, a = butter(order, low, btype='low', fs = nyq, analog=False, output='ba')

        # Apply the Butterworth filter using filtfilt
        nsamp, ntraces = segmentGPRdata.shape
        profile_lp = np.zeros((nsamp,ntraces))

        for itrace in range(ntraces):
            profile_lp[:,itrace] = filtfilt(b, a, segmentGPRdata[:,itrace])

        return profile_lp
    

    def _process_highpass_filter(self, segmentGPRdata, config=None):
        '''
        High-pass filter using Butterworth filter
        '''
        config = config or self.config.high_pass_filter

        # Design a Butterworth band-pass filter
        high_cutoff_frequency = config.high_cutoff_frequency* 1e9  # hz
        order = config.poly_order

        fs = 1/(self.dt*1e-9) #sample frequency Hz
        nyq = fs/2
        #high = high_cutoff_frequency / nyq
        high = high_cutoff_frequency
    
        b, a = butter(order, high, btype='high', fs = nyq,analog=False, output='ba')

        # Apply the Butterworth filter using filtfilt
        nsamp, ntraces = segmentGPRdata.shape
        profile_hp = np.zeros((nsamp,ntraces))

        for itrace in range(ntraces):
            profile_hp[:,itrace] = filtfilt(b, a, segmentGPRdata[:,itrace])

        return profile_hp


    def _process_bandpass_filter(self, segmentGPRdata, config=None):
        '''
        Band-pass filter using Butterworth filter
        '''

        config = config or self.config.band_pass_filter

        # Design a Butterworth band-pass filter
        low_cutoff_frequency = config.f1_bandpass * 1e9  # hz
        high_cutoff_frequency = config.f2_bandpass * 1e9 # Hz
        order = config.poly_order_bandpass

        fs = 1/(self.dt*1e-9) #sample frequency Hz
        nyq = fs * 1/2
        nyq2 =  self.freq/2          #fs/2
        #low = low_cutoff_frequency / nyq
        #high = high_cutoff_frequency / nyq
        #
        #b, a = butter(order, [low, high], btype='band', output='ba')

        low = low_cutoff_frequency
        high = high_cutoff_frequency
        b, a = butter(order, [low, high], fs=nyq, btype='band', output='ba')
        #b, a = butter(order, [low, high], btype='band', analog=False, output='ba')

        print("    >> butter-filter:")
        print("       f_center    = " + str(self.freq) + 'MHz')
        print("       f1_bandpass = " + str(low_cutoff_frequency* 1e-9) + 'GHz')
        print("       f2_bandpass = " + str(high_cutoff_frequency* 1e-9) + 'GHz')
        print("       order       = " + str(order))
        print("       low  = " + str(low))
        print("       high = " + str(high))
        print("       nyq  = " + str(nyq*1e-9) + 'GHz')

        # Apply the Butterworth filter using filtfilt
        nsamp, ntraces = segmentGPRdata.shape
        profile_bp = np.zeros((nsamp,ntraces))

        for itrace in range(ntraces):
            profile_bp[:,itrace] = filtfilt(b, a, segmentGPRdata[:,itrace])

        return profile_bp


    def _process_bandpass_filter_rgpr(self, segmentGPRdata, config=None):
        
        config = config or self.config.band_pass_filter_rgpr

        f1 = config.f1_bandpass
        f2 = config.f2_bandpass
        order = config.poly_order_bandpass

        print("    >> butter-filter (RGPR):")
        print("       f_center    = " + str(self.freq) + 'MHz')
        print("       f1_bandpass = " + str(f1) + 'GHz')
        print("       f2_bandpass = " + str(f2) + 'GHz')
        print("       order       = " + str(order))        
        
        # Convert numpy array to R matrix
        #r_matrix = ro.r['matrix'](ro.FloatVector(segmentGPRdata.flatten()), \
        #                                nrow=segmentGPRdata.shape[0], ncol=segmentGPRdata.shape[1])
        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)
        
        
        # Call RGPR function for band-pass filter
        if 0:
            #filtered_data = self.rgpr.bandpass(r_matrix, param1, param2)
            #Profile_R_filtered <- fFilter(Profile_R, f=c(f1*1000,f2*1000),type = "bandpass", plotSpec = TRUE)
            Profile_R_filtered = self.rgpr.fFilter(Profile_R, f=ro.FloatVector([f1*1000,f2*1000]),type = "bandpass", plotSpec = False)
            profile_filtered_data = self.r.slot(Profile_R_filtered,"data")
        else:
            self._source_rgpr()

            with localconverter(self._np_converter):
                profile_filtered_data = self.r.process_bandpass_R(Profile_R, f1,f2)


        return profile_filtered_data


    def _process_horizontal_filter(self, segmentGPRdata, config=None):

        config = config or self.config.horizontal_filter

        # Apply a 3x3 median filter along the axis for traces
        #hfiltered_data = median_filter(segmentGPRdata, size=(3, 3), axis=1)
        #hfiltered_data = np.apply_along_axis(median_filter, 0, segmentGPRdata, size=(3, 3))
        hfiltered_data = medfilt2d(segmentGPRdata, kernel_size=3)

        return hfiltered_data


    def _process_horizontal_filter_rgpr(self, segmentGPRdata, config=None):

        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object
        
        #Profile_R_space_filtered = self.rgpr.filter2D(Profile_R, type = "median3x3")
        #Profile_space_filtered = Profile_R_space_filtered.rx2("data")
        self._source_rgpr()
        #this is slow, apparatnly...
        with localconverter(self._np_converter):
            Profile_space_filtered = self.r.process_horizontal_filter_R(Profile_R)

        return Profile_space_filtered
    

    def _process_amplitude_correction(self, segmentGPRdata, config=None):
        '''
        Amplitude correction
        '''
        from processing_utils import apply_geometric_spreading, apply_intrinsic_attenuation

        config = config or self.config.amplitude_correction

        # Access amplitude correction parameters from the Config instance
        alpha_power_gain = config.alpha_power_gain
        te_power_gain = config.te_power_gain
        tcst_power_gain = config.tcst_power_gain
        
        alpha_exp_gain = config.alpha_exp_gain
        t0_exp_gain = config.t0_exp_gain
        te_exp_gain = config.te_exp_gain

        
        print("    >> a) power gain (correct for geometrical spreading)")
        print("       alpha_power_gain = " + str(alpha_power_gain))
        print("       te_power_gain    = " + str(te_power_gain))
        print("       tcst_power_gain  = " + str(tcst_power_gain))

        #create time-array (dim: nsamples) in ns:
        nsamples, ntraces = segmentGPRdata.shape
        timeArr = np.arange(0,self.dt*nsamples, self.dt) #

        #power gain (correct for geometrical spreading)
        profile_power_gain = apply_geometric_spreading(segmentGPRdata, time=timeArr,\
                                                         alpha = alpha_power_gain, te = te_power_gain, \
                                                            tcst = tcst_power_gain)

        
        print("    >> b) exponential gain (correct for intrinsic attenuation)")
        print("       alpha_exp_gain = " + str(alpha_exp_gain))
        print("       t0_exp_gain    = " + str(t0_exp_gain))
        print("       te_exp_gain    = " + str(te_exp_gain))

        #exponential gain (correct for intrinsic attenuation)
        profile_gain_corrections = apply_intrinsic_attenuation(profile_power_gain, time=timeArr,\
                                                          intrinsic_attenuation_coeff = alpha_exp_gain
                                                          , t0 = t0_exp_gain, te = te_exp_gain)


        return profile_gain_corrections


    def _process_amplitude_correction_rgpr(self, segmentGPRdata, config=None):
        '''
        Amplitude correction (geometrical spreading)
        '''
        config = config or self.config.amplitude_correction

        # Access amplitude correction parameters from the Config instance
        alpha_power_gain = config.alpha_power_gain
        te_power_gain = config.te_power_gain
        tcst_power_gain = config.tcst_power_gain
        
        alpha_exp_gain = config.alpha_exp_gain
        t0_exp_gain = config.t0_exp_gain
        te_exp_gain = config.te_exp_gain

        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object

        
        print("    >> a) power gain (correct for geometrical spreading)")
        print("       alpha_power_gain = " + str(alpha_power_gain))
        print("       te_power_gain    = " + str(te_power_gain))
        print("       tcst_power_gain  = " + str(tcst_power_gain))

        self._source_rgpr()

        #==> power gain (correct for geometrical spreading)       
        #Profile_R_power_gain = self.rgpr.gain(Profile_R, type = "power", alpha = alpha_power_gain, te = te_power_gain, tcst = tcst_power_gain)
        with localconverter(self._np_converter):
            profile_power_gain = self.r.process_gain_power_R(Profile_R, alpha_power_gain, te_power_gain, tcst_power_gain)
        
        print("    >> b) exponential gain (correct for intrinsic attenuation)")
        print("       alpha_exp_gain = " + str(alpha_exp_gain))
        print("       t0_exp_gain    = " + str(t0_exp_gain))
        print("       te_exp_gain    = " + str(te_exp_gain))

        #==> exponential gain (correct for intrinsic attenuation)
        #Profile_R_exp_gain = self.rgpr.gain(Profile_R_power_gain, type = "exp", alpha = alpha_exp_gain, t0 = t0_exp_gain, te = te_exp_gain)
        Profile_R_power_gain = self._build_rgpr_data(profile_power_gain,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object

        with localconverter(self._np_converter):
            profile_gain_corrections = self.r.process_gain_exp_R(Profile_R_power_gain, alpha_exp_gain, t0_exp_gain, te_exp_gain)

        #profile_gain_corrections = Profile_R_exp_gain.rx2("data")
        #profile_gain_corrections = self.r.slot(Profile_R_exp_gain, "data")

        return profile_gain_corrections

    def _process_power_gain_rgpr(self, segmentGPRdata, config=None):
        '''
        Amplitude correction (geometrical spreading)
        '''
        config = config or self.config.amplitude_correction

        # Access amplitude correction parameters from the Config instance
        alpha_power_gain = config.alpha_power_gain
        te_power_gain = config.te_power_gain
        tcst_power_gain = config.tcst_power_gain
        

        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object

        
        print("    >> power gain (correct for geometrical spreading)")
        print("       alpha_power_gain = " + str(alpha_power_gain))
        print("       te_power_gain    = " + str(te_power_gain))
        print("       tcst_power_gain  = " + str(tcst_power_gain))

        self._source_rgpr()

        #==> power gain (correct for geometrical spreading)       
        with localconverter(self._np_converter):
            profile_power_gain = self.r.process_gain_power_R(Profile_R, alpha_power_gain, te_power_gain, tcst_power_gain)

        return profile_power_gain


    def _process_dewow(self,segmentGPRdata, config=None):

        config = config or self.config.dewow
        # Apply Savitzky-Golay filter to remove low-frequency components
        # window_size = 15
        # poly_order = 2
        window_size = config.window_size
        poly_order = config.poly_order
        profile_dewowed = savgol_filter(segmentGPRdata, window_size, poly_order)
        
        return profile_dewowed

    def process_source_wavelet_deconvolution_rpgr(self, segmentGPRdata, config=None):
        '''
        Runs mixed-phase deconvolution is RGPR.

        @args:
        - segmentGPRdata : numpy 2d-array - segment profile
        - config         :  config.mixed_phase_deconvolution

        @Output:
        - profile after deconv
        - deconv_wavelet_mixed_phase_x : time
        - deconv_wavelet_mixed_phase_y : all traces
        - deconv_wavelet_min_phase_x   : time
        - deconv_wavelet_min_phase_y   : all traces
        '''

        config = config or self.config.mixed_phase_deconvolution
        # Implement source wavelet estimation logic here
        
        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object

        print("    >> Performs mixed-phased deconvolution:")
        print("     1) Deconvolution")
        print("         tWin_start : " + str(config.tWin_start))
        print("         tWin_end   : " + str(config.tWin_end))
        print("         wtr        : " + str(config.wtr))
        print("         mu         : " + str(config.mu))
        print("     2) Band-pass filter to remove high-frequency noise:")
        print("         f1 : " + str(config.f1_bandpass) + " GHz")
        print("         f2 : " + str(config.f2_bandpass) + " GHz")
        print("     3) Trace scaling")

        tWin_start = config.tWin_start
        tWin_end   = config.tWin_end
        wtr = config.wtr
        mu  = config.mu

        f1 = config.f1_bandpass
        f2 = config.f2_bandpass

        self.r.source("RGPR_functions.R")

        with localconverter(self._np_converter):
            output_list_R = self.r.process_mixed_phased_deconv_all_R(Profile_R,tWin_start,tWin_end,wtr,mu, f1, f2)

        # Convert the result to a Python list
        result_list = list(output_list_R)

        profile_after_deconv = result_list[0]
        deconv_wavelet_mixed_phase_x = result_list[1]
        deconv_wavelet_mixed_phase_y = result_list[2]
        deconv_wavelet_min_phase_x = result_list[3]
        deconv_wavelet_min_phase_y = result_list[4]

        return profile_after_deconv,deconv_wavelet_mixed_phase_x,deconv_wavelet_mixed_phase_y,deconv_wavelet_min_phase_x, deconv_wavelet_min_phase_y



        

    def _process_mixed_phase_deconvolution_rgpr(self, segmentGPRdata, config=None):
        # Implement mixed-phase deconvolution logic here

        config = config or self.config.mixed_phase_deconvolution

        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object

        print("    >> Performs:")
        print("     1) Deconvolution")
        print("         tWin_start : " + str(config.tWin_start))
        print("         tWin_end   : " + str(config.tWin_end))
        print("         wtr        : " + str(config.wtr))
        print("         mu         : " + str(config.mu))
        print("     2) Band-pass filter to remove high-frequency noise:")
        print("         f1 : " + str(config.f1_bandpass) + " GHz")
        print("         f2 : " + str(config.f2_bandpass) + " GHz")
        print("     3) Trace scaling")

        tWin_start = config.tWin_start
        tWin_end   = config.tWin_end
        wtr = config.wtr
        mu  = config.mu

        f1 = config.f1_bandpass
        f2 = config.f2_bandpass

        # ns - time window on which the wavelet is estimated
        #tWin = ro.FloatVector([tWin_start, tWin_end])

        #winElems <- which(depth(Profile_R) > tWin[1] & depth(Profile_R) < tWin[2])
        #wlen = len(winElems)-1
        #profile_R_deconv = self.rgpr.deconv(Profile_R, method="mixed-phase", W = tWin, wtr = wtr, nf = wlen, mu = mu)
        #profile_deconv = self.r.slot(profile_R_deconv, "data")
        
        profile_deconv = self._build_rgpr_mixed_phased_deconv(Profile_R,tWin_start,tWin_end,wtr,mu, f1, f2)

        return profile_deconv
    

    def _process_reflectivity_spectral_ratio(self, segmentGPRdata, param1, param2):
        # Implement reflectivity spectral ratio logic here
        pass

    def _process_reflectivity_spikes_direct(self, segmentGPRdata, param1, param2):
        # Implement reflectivity spikes direct logic here
        pass    

    def process_kirchoff_migration_rgpr(self,segmentGPRdata, velocity_ns = 0.5, fdo = 0.75e3,\
                                         dz = None, max_depth = None ):
        '''
        Perform Kirchoff depth migration on given profile data <segmentGPRdata>.
        @args:
        - segmentGPRdata : GPR profile data. numpy array - dim:(nsamp,ntraces)
        - velocity       : uniform velocity in nano-seconds per meter
        - fdo            : dominant frequency in MHz
        - dz             : vertical resolution for migrated profile
        - max_depth      : max depth in meter

        @output:
        - profile_migrated : migrated profile. numpy array
        - config           : Config object for settings used in migration process
        '''
        nsamp, ntraces = segmentGPRdata.shape

        config_use = KirchoffMigration()
        
        dt = self.dt
        resample = self.resample
        time_window = nsamp * dt / resample #(nsamp-1)*dt

        # fdo - dominant frequency...perform fft on profile

        if dz is None:
            dz = velocity_ns * dt/resample
            max_depth = velocity_ns * time_window

        config_use.dz = dz
        config_use.fdo_MHz = fdo
        config_use.max_depth = max_depth
        config_use.velocity_ns = velocity_ns
        
        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object
        
        with localconverter(self._np_converter):
            profile_migrated = self.r.process_kirchoff_migration_R(Profile_R,velocity_ns, max_depth, dz, fdo)

        return profile_migrated, config_use


    def process_data(self, inputDataSegment,custom_config=None):
        '''
        
        @args:
        - inputDataSegment: numpy array 2D or 1D (nsamp, ntraces) to be processed
        - custom_config   : ProcessingConfig object for selecting steps
                            included in processing as well as associated
                            process parameters.
        
        @output:
        - segment_data   : final processed segment profile (2d/1d numpy array) after
                           every process step has been performed
        - processed_data : list of processed segment profile (2d numpy array) after
                           each process step
        '''
        custom_config = custom_config or ProcessingConfig()

        # Use the provided or default configuration for each step
        ordered_steps = custom_config.processing_order or self.config.processing_order
        #include_steps = custom_config.include_steps or self.config.include_steps

        print(" ===> Process current data segment with " + str(len(ordered_steps)) + " steps:")
        print("Processing pipeline:")
        i = 1
        for step in ordered_steps:
            print("  step " + str(i) + ": " + step)
            i = i + 1
        print(" --- ")


        segment_data = inputDataSegment

        processed_data = list()

        for step in ordered_steps:
            print(" ---> performing step: " + str(step))
            if step == "resample_profile":
                segment_data = self._process_resample(segment_data, config=custom_config)
                processed_data.append(segment_data)                
            elif step == "dc_shift_correction_rgpr":
                segment_data = self._process_DC_shift_RGPR(segment_data, config=custom_config.dc_shift_correction)
                processed_data.append(segment_data)
            elif step == "t0_correction":
                segment_data = self._process_t0_correction(segment_data, config=custom_config.t0_correction)
                processed_data.append(segment_data)
            elif step == "t0_correction_rgpr":
                segment_data = self._process_t0_correction_rgpr(segment_data, config=custom_config.t0_correction)
                processed_data.append(segment_data)
            elif step == "amplitude_correction_rgpr":
                segment_data = self._process_amplitude_correction_rgpr(segment_data, config=custom_config.amplitude_correction)
                processed_data.append(segment_data)
            elif step == "amplitude_correction":
                segment_data = self._process_amplitude_correction(segment_data, config=custom_config.amplitude_correction)
                processed_data.append(segment_data)     
            elif step == "power_gain_rgpr":
                segment_data = self._process_power_gain_rgpr(segment_data, config=custom_config.amplitude_correction)
                processed_data.append(segment_data)  
            elif step == "background_removal":
                segment_data = self._process_background_removal(segment_data, config=custom_config.background_removal)
                processed_data.append(segment_data)
            elif step == "background_removal_rgpr":
                segment_data = self._process_background_removal_rgpr(segment_data, config=custom_config.background_removal)
                processed_data.append(segment_data)
            elif step == "horizontal_filter":
                segment_data = self._process_horizontal_filter(segment_data, config=custom_config.horizontal_filter)
                processed_data.append(segment_data)
            elif step == "horizontal_filter_rgpr":
                segment_data = self._process_horizontal_filter_rgpr(segment_data, config=custom_config.horizontal_filter)
                processed_data.append(segment_data)
            elif step=="bandpass_filter":
                segment_data = self._process_bandpass_filter(segment_data,config=custom_config.band_pass_filter)
                processed_data.append(segment_data)
            elif step=="bandpass_filter_rgpr":
                segment_data = self._process_bandpass_filter_rgpr(segment_data,config=custom_config.band_pass_filter)
                processed_data.append(segment_data)
            elif step=="mixed_phase_deconvolution_rgpr":
                segment_data = self._process_mixed_phase_deconvolution_rgpr(segment_data,config=custom_config.mixed_phase_deconvolution)
                processed_data.append(segment_data)
            else:
                print("  WARNING: " + step + " is not a valid processing step")
                 

        return segment_data, processed_data
