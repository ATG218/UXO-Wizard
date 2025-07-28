import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr, PackageNotInstalledError
from rpy2.rinterface_lib.embedded import RRuntimeError
from pathlib import Path
import sys, traceback
from loguru import logger

from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
from scipy.signal import medfilt2d

from scipy.ndimage import median_filter #generic_filter

from src.processing.scripts.gpr.utils.gpr_configs import *

def debug_data_stats(data, step_name, stage="input", extra_info=""):
    """Helper function to log consistent debugging information for data"""
    logger.info(f"=== {step_name} - {stage.upper()} ===")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"Data type: {data.dtype}")
    logger.info(f"Min: {np.min(data):.6f}")
    logger.info(f"Max: {np.max(data):.6f}")
    logger.info(f"Mean: {np.mean(data):.6f}")
    logger.info(f"Std: {np.std(data):.6f}")
    
    # Check for problematic values
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    zero_count = np.sum(data == 0)
    
    if nan_count > 0:
        logger.warning(f"Contains {nan_count} NaN values")
    if inf_count > 0:
        logger.warning(f"Contains {inf_count} infinite values")
    if zero_count > 0:
        logger.info(f"Contains {zero_count} zero values")
    
    if extra_info:
        logger.info(f"Extra info: {extra_info}")
    logger.info("=" * 50)

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
        if not getattr(self, "__rgpr_sourced", False):
            self.r.source(str(self._rgpr_functions_path))
            self._rgpr_sourced = True

    def _build_rgpr_data(self, profile, freq, dt, resample, txrx):
        # Source the RGPR_func.R file
        self._source_rgpr()

        # Call the build_RGPR_data function
        with localconverter(self._np_converter):
            return self.r.build_RGPR_data(profile, freq, dt, resample, txrx)

    def _build_rgpr_mixed_phased_deconv(self, profileR,tWin_start, tWin_end, wtr, mu, f1, f2):
        # Source the RGPR_func.R file
        logger.info("Sourcing RGPR functions...")
        self._source_rgpr()
        logger.info("RGPR functions sourced successfully")

        # Call the build_RGPR_data function
        logger.info("Setting up numpy converter for R processing...")
        try:
            with localconverter(self._np_converter):
                logger.info("About to call process_mixed_phased_deconv_R...")
                logger.info(f"Parameters: profileR shape={profileR.rx2('data').shape if hasattr(profileR, 'rx2') else 'unknown'}, tWin_start={tWin_start}, tWin_end={tWin_end}, wtr={wtr}, mu={mu}, f1={f1}, f2={f2}")
                
                result = self.r.process_mixed_phased_deconv_R(profileR,tWin_start, tWin_end, wtr, mu, f1, f2)
                logger.info("process_mixed_phased_deconv_R completed successfully")
                return result
        except Exception as e:
            logger.error(f"Error in R processing: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    
    def _process_resample(self, segmentGPRdata, config):
        debug_data_stats(segmentGPRdata, "RESAMPLE", "input")
        
        resample = config.resample or self.config.resample
        logger.info(f"Resample factor: {resample}")

        self.resample = resample

        from .processing_utils import resample_all_traces
        nsamps, ntraces = segmentGPRdata.shape

        #resample_all_traces(n_samples,resample,nbr_tr,traces_cut)
        profile_resampled = resample_all_traces(nsamps,resample,ntraces,segmentGPRdata.T)
        
        debug_data_stats(profile_resampled, "RESAMPLE", "output", f"Resample factor: {resample}")
        return profile_resampled
    

    def _process_DC_shift_RGPR(self, segmentGPRdata, config):
        debug_data_stats(segmentGPRdata, "DC_SHIFT_RGPR", "input")
        logger.info(f"STEP DC_SHIFT_RGPR: dt = {self.dt}  resample = {self.resample}")

        config = config or self.config.dc_shift_correction

        dc_sample_start = config.DCsample_start
        dc_sample_end   = config.DCsample_end
        
        logger.info(f"DC shift parameters: start={dc_sample_start}, end={dc_sample_end}")

        #convert to R (not sure if necessary)
        profile_R = self._build_rgpr_data(segmentGPRdata, self.freq, self.dt, self.resample, self.TxRx)
        
        if 0:
            #Use the samples between DC_sample_start and DC_sample_end to estimate DC shift
            z_dc_corrected = self.rgpr.dcshift(profile_R, u=ro.IntVector(range(dc_sample_start, dc_sample_end + 1)))    
            profile_dc_corrected = self.r.slot(z_dc_corrected,"data")        
            #profile_dc_corrected = z_dc_corrected.rx2("data")

        self._source_rgpr()
        with localconverter(self._np_converter):
            profile_dc_corrected = self.r.process_DCshift_R(profile_R, int(dc_sample_start), int(dc_sample_end))
        
        print("after DC‑shift:", profile_dc_corrected.min(), profile_dc_corrected.max())
        debug_data_stats(profile_dc_corrected, "DC_SHIFT_RGPR", "output", f"DC samples: {dc_sample_start}-{dc_sample_end}")
        return profile_dc_corrected


    def _process_t0_correction_2(self, segmentGPRdata,config=None):
        '''
        first break estimation required?
        '''
        debug_data_stats(segmentGPRdata, "T0_CORRECTION_2", "input")
        
        #convert to R (not sure if necessary)
        profile_R = self._build_rgpr_data(segmentGPRdata, self.freq, self.dt, self.resample, self.TxRx)

        threshold_t0 = config.threshold_t0 #unit ns? check
        logger.info(f"T0 correction threshold: {threshold_t0}")

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

        debug_data_stats(profile_t0, "T0_CORRECTION_2", "output", f"Threshold: {threshold_t0}, Padding added")
        return profile_t0
    

    def _process_t0_correction_rgpr(self, segmentGPRdata,config=None):
        '''
        first break estimation + t0 correction using RGPR
        '''
        debug_data_stats(segmentGPRdata, "T0_CORRECTION_RGPR", "input")
        logger.info(f"STEP T0_CORRECTION_RGPR: dt = {self.dt}  resample = {self.resample}")

        config = config or self.config.t0_correction

        #convert to R (not sure if necessary)
        profile_R = self._build_rgpr_data(segmentGPRdata, self.freq, self.dt, self.resample, self.TxRx)

        threshold_t0 = config.threshold_t0 #unit ns? check
        logger.info(f"T0 correction (RGPR) threshold: {threshold_t0}")

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

        debug_data_stats(profile_t0, "T0_CORRECTION_RGPR", "output", f"Threshold: {threshold_t0}")
        return profile_t0


    def _process_t0_correction(self, segmentGPRdata,config=None):
        '''
        first break estimation required?
        '''
        debug_data_stats(segmentGPRdata, "T0_CORRECTION", "input")

        config = config or self.config.t0_correction

        #convert to R (not sure if necessary)
        profile_R = self._build_rgpr_data(segmentGPRdata, self.freq, self.dt, self.resample, self.TxRx)

        threshold_t0 = config.threshold_t0 #unit ns? check
        logger.info(f"T0 correction threshold: {threshold_t0}")

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

        debug_data_stats(profile_t0, "T0_CORRECTION", "output", f"Threshold: {threshold_t0}, Padding added")
        return profile_t0
    

    def _process_background_removal(self, segmentGPRdata,config=None):
        '''mean trace'''
        debug_data_stats(segmentGPRdata, "BACKGROUND_REMOVAL", "input")

        nsamp, ntraces = segmentGPRdata.shape
        logger.info(f"Processing {ntraces} traces with {nsamp} samples each")
        
        mean_trace = np.zeros(nsamp)
        profile_bm_removal = np.zeros((nsamp,ntraces))

        for itrace in range(ntraces):
            mean_trace += segmentGPRdata[:,itrace]

        mean_trace = mean_trace/(ntraces)
        logger.info(f"Mean trace stats - Min: {np.min(mean_trace):.6f}, Max: {np.max(mean_trace):.6f}, Mean: {np.mean(mean_trace):.6f}")

        for itrace in range(ntraces):
            profile_bm_removal[:,itrace] = segmentGPRdata[:,itrace] - mean_trace

        debug_data_stats(profile_bm_removal, "BACKGROUND_REMOVAL", "output", f"Mean subtracted from {ntraces} traces")
        return profile_bm_removal



    def _process_background_removal_rgpr(self, segmentGPRdata,config=None):
        debug_data_stats(segmentGPRdata, "BACKGROUND_REMOVAL_RGPR", "input")
        
        config = config or self.config.background_removal
        logger.info(f"STEP BACKGROUND_REMOVAL_RGPR: dt = {self.dt}  resample = {self.resample}")
        logger.info(f"Background removal method: {config.method}")
        #return segmentGPRdata_Processed_background_rem

        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object

        if config.method == "rmBackgroundMatrix":
            logger.info("Using rmBackgroundMatrix method")
            Profile_background_sub_R = self.rgpr.rmBackgroundMatrix(Profile_R, width = 21, trim = 0.2, s = 1, eps = 1, itmax = 5)
            with localconverter(self._np_converter):
                Profile_background_sub = self.r.slot(Profile_background_sub_R,"data").copy(order="F")
            #Profile_background_sub = Profile_background_sub_R.rx2("data")
            profile_background_sub = segmentGPRdata - Profile_background_sub
        else:   # mean‑trace branch
            logger.info("Using mean trace method")
            self._source_rgpr()
            with localconverter(self._np_converter):
                mean_trace = self.r.process_background_removal_R(Profile_R)

            # Convert to NumPy and flatten the single column → (nsamp,)
            mean_trace = np.asarray(mean_trace).ravel(order="F")       # or:  mean_trace = mean_trace[:, 0]

            # Sanity‑check
            if mean_trace.shape[0] != segmentGPRdata.shape[0]:
                raise ValueError(f"mean_trace len {mean_trace.shape[0]} "
                                f"≠ nsamp {segmentGPRdata.shape[0]}")
            logger.info(f"segment {segmentGPRdata.shape} mean {mean_trace.shape} min/max mean {mean_trace.min()} {mean_trace.max()}")
            print("corrcoef original vs mean:",
                  np.corrcoef(segmentGPRdata.ravel(),
                  mean_trace[:,None].repeat(segmentGPRdata.shape[1],1).ravel())[0,1])


            # Explicit broadcast along traces
            profile_background_sub = segmentGPRdata - mean_trace[:, None]


        debug_data_stats(profile_background_sub, "BACKGROUND_REMOVAL_RGPR", "output", f"Method: {config.method}")
        return profile_background_sub


    def _process_lowpass_filter(self, segmentGPRdata, config=None):
        '''
        Low-pass filter using Butterworth filter
        '''
        debug_data_stats(segmentGPRdata, "LOWPASS_FILTER", "input")
        
        config = config or self.config.low_pass_filter

        # Design a Butterworth low-pass filter
        low_cutoff_frequency = config.low_cutoff_frequency * 1e9 #Hz
        order = config.poly_order
        
        logger.info(f"Lowpass filter - Cutoff: {low_cutoff_frequency/1e9:.3f} GHz, Order: {order}")

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

        debug_data_stats(profile_lp, "LOWPASS_FILTER", "output", f"Cutoff: {low_cutoff_frequency/1e9:.3f} GHz, Order: {order}")
        return profile_lp
    

    def _process_highpass_filter(self, segmentGPRdata, config=None):
        '''
        High-pass filter using Butterworth filter
        '''
        debug_data_stats(segmentGPRdata, "HIGHPASS_FILTER", "input")
        
        config = config or self.config.high_pass_filter

        # Design a Butterworth band-pass filter
        high_cutoff_frequency = config.high_cutoff_frequency* 1e9  # hz
        order = config.poly_order
        
        logger.info(f"Highpass filter - Cutoff: {high_cutoff_frequency/1e9:.3f} GHz, Order: {order}")

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

        debug_data_stats(profile_hp, "HIGHPASS_FILTER", "output", f"Cutoff: {high_cutoff_frequency/1e9:.3f} GHz, Order: {order}")
        return profile_hp


    def _process_bandpass_filter(self, segmentGPRdata, config=None):
        '''
        Band-pass filter using Butterworth filter
        '''
        debug_data_stats(segmentGPRdata, "BANDPASS_FILTER", "input")

        config = config or self.config.band_pass_filter

        # Design a Butterworth band-pass filter
        low_cutoff_frequency = config.f1_bandpass * 1e9  # hz
        high_cutoff_frequency = config.f2_bandpass * 1e9 # Hz
        order = config.poly_order_bandpass
        
        logger.info(f"Bandpass filter - Low: {low_cutoff_frequency/1e9:.3f} GHz, High: {high_cutoff_frequency/1e9:.3f} GHz, Order: {order}")

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

        debug_data_stats(profile_bp, "BANDPASS_FILTER", "output", f"Low: {low_cutoff_frequency/1e9:.3f} GHz, High: {high_cutoff_frequency/1e9:.3f} GHz")
        return profile_bp


    def _process_bandpass_filter_rgpr(self, segmentGPRdata, config=None):
        debug_data_stats(segmentGPRdata, "BANDPASS_FILTER_RGPR", "input")
        logger.info(f"STEP BANDPASS_FILTER_RGPR: dt = {self.dt}  resample = {self.resample}")
        
        config = config or self.config.band_pass_filter_rgpr

        f1 = config.f1_bandpass
        f2 = config.f2_bandpass
        order = config.poly_order_bandpass
        
        logger.info(f"Bandpass filter (RGPR) - f1: {f1} GHz, f2: {f2} GHz, Order: {order}")

        print("    >> butter-filter (RGPR):")
        print("       f_center    = " + str(self.freq) + 'MHz')
        print("       f1_bandpass = " + str(f1) + 'GHz')
        print("       f2_bandpass = " + str(f2) + 'GHz')
        print("       order       = " + str(order))        
        
        # Convert numpy array to R matrix
        #r_matrix = ro.r['matrix'](ro.FloatVector(segmentGPRdata.flatten()), \
        #                                nrow=segmentGPRdata.shape[0], ncol=segmentGPRdata.shape[1])
        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)
        
        # right after Profile_R is built
        dz_ns  = float(ro.r['slot'](Profile_R, "dz")[0])      # vertical sample period Δt (ns)
        fc_MHz = float(ro.r['slot'](Profile_R, "freq")[0])    # nominal antenna centre freq (MHz)
        print(f"DEBUG hdr: dz = {dz_ns:.6g} ns ;  freq centre = {fc_MHz:.6g} MHz")
        print(f"DEBUG python: dt = {self.dt}  resample = {self.resample}")

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

        debug_data_stats(profile_filtered_data, "BANDPASS_FILTER_RGPR", "output", f"f1: {f1} GHz, f2: {f2} GHz")
        return profile_filtered_data


    def _process_horizontal_filter(self, segmentGPRdata, config=None):
        debug_data_stats(segmentGPRdata, "HORIZONTAL_FILTER", "input")

        config = config or self.config.horizontal_filter
        logger.info("Applying 3x3 median filter")

        # Apply a 3x3 median filter along the axis for traces
        #hfiltered_data = median_filter(segmentGPRdata, size=(3, 3), axis=1)
        #hfiltered_data = np.apply_along_axis(median_filter, 0, segmentGPRdata, size=(3, 3))
        hfiltered_data = medfilt2d(segmentGPRdata, kernel_size=3)

        debug_data_stats(hfiltered_data, "HORIZONTAL_FILTER", "output", "3x3 median filter applied")
        return hfiltered_data


    def _process_horizontal_filter_rgpr(self, segmentGPRdata, config=None):
        debug_data_stats(segmentGPRdata, "HORIZONTAL_FILTER_RGPR", "input")

        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object
        
        #Profile_R_space_filtered = self.rgpr.filter2D(Profile_R, type = "median3x3")
        #Profile_space_filtered = Profile_R_space_filtered.rx2("data")
        self._source_rgpr()
        #this is slow, apparatnly...
        with localconverter(self._np_converter):
            Profile_space_filtered = self.r.process_horizontal_filter_R(Profile_R)

        debug_data_stats(Profile_space_filtered, "HORIZONTAL_FILTER_RGPR", "output", "RGPR median3x3 filter applied")
        return Profile_space_filtered
    

    def _process_amplitude_correction(self, segmentGPRdata, config=None):
        '''
        Amplitude correction
        '''
        debug_data_stats(segmentGPRdata, "AMPLITUDE_CORRECTION", "input")
        
        from .processing_utils import apply_geometric_spreading, apply_intrinsic_attenuation

        config = config or self.config.amplitude_correction

        # Access amplitude correction parameters from the Config instance
        alpha_power_gain = config.alpha_power_gain
        te_power_gain = config.te_power_gain
        tcst_power_gain = config.tcst_power_gain
        
        alpha_exp_gain = config.alpha_exp_gain
        t0_exp_gain = config.t0_exp_gain
        te_exp_gain = config.te_exp_gain

        logger.info(f"Amplitude correction - Power gain: α={alpha_power_gain}, te={te_power_gain}, tcst={tcst_power_gain}")
        logger.info(f"Amplitude correction - Exp gain: α={alpha_exp_gain}, t0={t0_exp_gain}, te={te_exp_gain}")
        
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

        debug_data_stats(profile_power_gain, "AMPLITUDE_CORRECTION", "intermediate", "After power gain")
        
        print("    >> b) exponential gain (correct for intrinsic attenuation)")
        print("       alpha_exp_gain = " + str(alpha_exp_gain))
        print("       t0_exp_gain    = " + str(t0_exp_gain))
        print("       te_exp_gain    = " + str(te_exp_gain))

        #exponential gain (correct for intrinsic attenuation)
        profile_gain_corrections = apply_intrinsic_attenuation(profile_power_gain, time=timeArr,\
                                                          intrinsic_attenuation_coeff = alpha_exp_gain
                                                          , t0 = t0_exp_gain, te = te_exp_gain)

        debug_data_stats(profile_gain_corrections, "AMPLITUDE_CORRECTION", "output", "After exponential gain")
        return profile_gain_corrections


    def _process_amplitude_correction_rgpr(self, segmentGPRdata, config=None):
        '''
        Amplitude correction (geometrical spreading)
        '''
        debug_data_stats(segmentGPRdata, "AMPLITUDE_CORRECTION_RGPR", "input")
        
        config = config or self.config.amplitude_correction

        # Access amplitude correction parameters from the Config instance
        alpha_power_gain = config.alpha_power_gain
        te_power_gain = config.te_power_gain
        tcst_power_gain = config.tcst_power_gain
        
        alpha_exp_gain = config.alpha_exp_gain
        t0_exp_gain = config.t0_exp_gain
        te_exp_gain = config.te_exp_gain

        logger.info(f"Amplitude correction (RGPR) - Power gain: α={alpha_power_gain}, te={te_power_gain}, tcst={tcst_power_gain}")
        logger.info(f"Amplitude correction (RGPR) - Exp gain: α={alpha_exp_gain}, t0={t0_exp_gain}, te={te_exp_gain}")

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
        
        debug_data_stats(profile_power_gain, "AMPLITUDE_CORRECTION_RGPR", "intermediate", "After power gain")
        
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

        debug_data_stats(profile_gain_corrections, "AMPLITUDE_CORRECTION_RGPR", "output", "After exponential gain")
        return profile_gain_corrections

    def _process_power_gain_rgpr(self, segmentGPRdata, config=None):
        '''
        Amplitude correction (geometrical spreading)
        '''
        debug_data_stats(segmentGPRdata, "POWER_GAIN_RGPR", "input")
        
        config = config or self.config.amplitude_correction

        # Access amplitude correction parameters from the Config instance
        alpha_power_gain = config.alpha_power_gain
        te_power_gain = config.te_power_gain
        tcst_power_gain = config.tcst_power_gain
        
        logger.info(f"Power gain (RGPR) - α={alpha_power_gain}, te={te_power_gain}, tcst={tcst_power_gain}")

        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object

        
        print("    >> power gain (correct for geometrical spreading)")
        print("       alpha_power_gain = " + str(alpha_power_gain))
        print("       te_power_gain    = " + str(te_power_gain))
        print("       tcst_power_gain  = " + str(tcst_power_gain))

        self._source_rgpr()

        #==> power gain (correct for geometrical spreading)       
        with localconverter(self._np_converter):
            profile_power_gain = self.r.process_gain_power_R(Profile_R, alpha_power_gain, te_power_gain, tcst_power_gain)

        debug_data_stats(profile_power_gain, "POWER_GAIN_RGPR", "output", f"α={alpha_power_gain}, te={te_power_gain}, tcst={tcst_power_gain}")
        return profile_power_gain


    def _process_dewow(self,segmentGPRdata, config=None):
        debug_data_stats(segmentGPRdata, "DEWOW", "input")

        config = config or self.config.dewow
        # Apply Savitzky-Golay filter to remove low-frequency components
        # window_size = 15
        # poly_order = 2
        window_size = config.window_size
        poly_order = config.poly_order
        
        logger.info(f"Dewow filter - Window size: {window_size}, Poly order: {poly_order}")
        
        profile_dewowed = savgol_filter(segmentGPRdata, window_size, poly_order)
        
        debug_data_stats(profile_dewowed, "DEWOW", "output", f"Window: {window_size}, Poly order: {poly_order}")
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
        debug_data_stats(segmentGPRdata, "SOURCE_WAVELET_DECONV", "input")

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

        logger.info(f"Source wavelet deconv - tWin: {tWin_start}-{tWin_end}, wtr: {wtr}, mu: {mu}, f1: {f1}, f2: {f2}")

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

        debug_data_stats(profile_after_deconv, "SOURCE_WAVELET_DECONV", "output", "Mixed-phase deconvolution completed")
        return profile_after_deconv,deconv_wavelet_mixed_phase_x,deconv_wavelet_mixed_phase_y,deconv_wavelet_min_phase_x, deconv_wavelet_min_phase_y



        

    def _process_mixed_phase_deconvolution_rgpr(self, segmentGPRdata, config=None):
        # Implement mixed-phase deconvolution logic here
        debug_data_stats(segmentGPRdata, "MIXED_PHASE_DECONV_RGPR", "input")

        config = config or self.config.mixed_phase_deconvolution

        logger.info("Starting mixed phase deconvolution RGPR processing")
        logger.info(f"Input data shape: {segmentGPRdata.shape}")
        logger.info(f"Input data range: [{np.min(segmentGPRdata):.6f}, {np.max(segmentGPRdata):.6f}]")

        try:
            logger.info("Building RGPR data object...")
            Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object
            logger.info("RGPR data object built successfully")
        except Exception as e:
            logger.error(f"Failed to build RGPR data object: {e}")
            raise

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

        logger.info(f"Mixed phase deconvolution parameters: tWin_start={tWin_start}, tWin_end={tWin_end}, wtr={wtr}, mu={mu}, f1={f1}, f2={f2}")

        # ns - time window on which the wavelet is estimated
        #tWin = ro.FloatVector([tWin_start, tWin_end])

        #winElems <- which(depth(Profile_R) > tWin[1] & depth(Profile_R) < tWin[2])
        #wlen = len(winElems)-1
        #profile_R_deconv = self.rgpr.deconv(Profile_R, method="mixed-phase", W = tWin, wtr = wtr, nf = wlen, mu = mu)
        #profile_deconv = self.r.slot(profile_R_deconv, "data")
        
        try:
            logger.info("Starting mixed phase deconvolution R processing...")
            profile_deconv = self._build_rgpr_mixed_phased_deconv(Profile_R,tWin_start,tWin_end,wtr,mu, f1, f2)
            logger.info("Mixed phase deconvolution R processing completed")
            logger.info(f"Output data shape: {profile_deconv.shape}")
            logger.info(f"Output data range: [{np.min(profile_deconv):.6f}, {np.max(profile_deconv):.6f}]")
        except Exception as e:
            logger.error(f"Failed during mixed phase deconvolution R processing: {e}")
            raise

        debug_data_stats(profile_deconv, "MIXED_PHASE_DECONV_RGPR", "output", f"tWin: {tWin_start}-{tWin_end}, wtr: {wtr}, mu: {mu}")
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
        debug_data_stats(segmentGPRdata, "KIRCHOFF_MIGRATION", "input")
        
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
        
        logger.info(f"Kirchoff migration - velocity: {velocity_ns} ns/m, max_depth: {max_depth} m, dz: {dz} m, fdo: {fdo} MHz")
        
        Profile_R = self._build_rgpr_data(segmentGPRdata,self.freq, self.dt, self.resample, self.TxRx)  # Build RGPR data object
        
        with localconverter(self._np_converter):
            profile_migrated = self.r.process_kirchoff_migration_R(Profile_R,velocity_ns, max_depth, dz, fdo)

        debug_data_stats(profile_migrated, "KIRCHOFF_MIGRATION", "output", f"velocity: {velocity_ns}, max_depth: {max_depth}")
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

        # Debug initial data
        debug_data_stats(inputDataSegment, "INITIAL_INPUT", "input", f"Processing pipeline: {ordered_steps}")

        segment_data = inputDataSegment

        processed_data = list()

        for step_idx, step in enumerate(ordered_steps):
            print(f" ---> performing step {step_idx + 1}/{len(ordered_steps)}: {step}")
            logger.info(f"Starting processing step {step_idx + 1}/{len(ordered_steps)}: {step}")
            
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
                
            logger.info(f"Completed processing step {step_idx + 1}/{len(ordered_steps)}: {step}")
            logger.info("=" * 80)

        # Debug final output
        debug_data_stats(segment_data, "FINAL_OUTPUT", "output", f"Total steps processed: {len(ordered_steps)}")
        return segment_data, processed_data
