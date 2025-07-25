import json


class GPRtypeConfig:
    def __init__(self, typeGPR = 'RADSYS_ZOND', freq_center = None):

        self.typeGPR = typeGPR

        #zond - 1Ghz
        if typeGPR == 'RADSYS_ZOND':
            if freq_center is None:
                self.freq_cent = 1000 # Central frequency of antenna in MHz (used to build RGPR data object)
            else:
                self.freq_cent = freq_center  
            self.TxRx = 0.15 # m - Spacing between Tx and Rx. For Zond Aero 1GHz, TxRx = 0.15m 
        #### CHECK the other input for RGPR, incl dx (after x interpolation)
        elif typeGPR == 'RADSYS_ZGPR':
            if freq_center is None:
                self.freq_cent = 400 # Central frequency of antenna in MHz (used to build RGPR data object)
            else:
                self.freq_cent = freq_center  
            self.TxRx = 0.0 # m - Spacing between Tx and Rx. For Zond Aero 1GHz, TxRx = 0.15m CHECK ZGPR
            #other paramters?  
        else: #default
            self.freq_cent = 1000 # Central frequency of antenna in MHz (used to build RGPR data object)  
            self.TxRx = 0.15 # m - Spacing between Tx and Rx. For Zond Aero 1GHz, TxRx = 0.15m 

        
    def to_dict(self):
        return {
            'typeGPR'  : self.typeGPR,
            'freq_cent': self.freq_cent,
            'TxRx': self.TxRx
        }    

###############################################################################

class KirchoffMigration():
    def __init__(self,velocity_ns = 0.46, fdo_MHz = 0.75e3, max_depth = 10, dz = 0.01):
        # Bandpass Butterworth filter parameters
        self.velocity_ns = velocity_ns # uniform velocity in nano-seconds/meter
        self.fdo_MHz = fdo_MHz # dominant frequency in MHz
        self.max_depth = max_depth # max depth
        self.dz = dz #vertical resolution in meter for migrated model

    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        return {
            'velocity_ns': self.velocity_ns,
            'fdo_MHz': self.fdo_MHz,
            'max_depth': self.max_depth,
            'dz': self.dz
        }  


class T0CorrectionConfig:
    def __init__(self, time1 = 5,time2 = 17, threshold_t0 = 0.02):
        self.time1 = time1           # Start time for zoom on 1st arrival
        self.time2 = time2          # End time for zoom on 1st arrival
        self.threshold_t0 = threshold_t0 # Threshold value to define t0 (to be tested with different values)
    
    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        return {
            'time1': self.time1,
            'time2': self.time2,
            'threshold_t0':self.threshold_t0
        }    
    
class DewowConfig:
    def __init__(self,window_size = 10,poly_order = 2):
        self.window_size = window_size
        self.poly_order = poly_order    
    
    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        return {
            'window_size': self.window_size,
            'poly_order': self.poly_order
        }    


class DCshiftCorrectionConfig:
    def __init__(self, DCsample_start = 250, DCsample_end = 450):
        # Estimate DC shift between samples DCsample_start and DCsample_end
        self.DCsample_start = DCsample_start  
        self.DCsample_end = DCsample_end
        
    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        return {
            'DCsample_start': self.DCsample_start,
            'DCsample_end': self.DCsample_end
        }

class AmplitudeCorrectionConfig:
    '''
    Process-config:  AmplitudeCorrectionConfig
    See RGPR gain functions: https://emanuelhuber.github.io/RGPR/02_RGPR_tutorial_basic-GPR-data-processing/#amplitude-gain

    Power gain (correct for geometrical spreading)
      -  self.alpha_power_gain = 1 # Power order
      -  self.te_power_gain = 37   # Gain applied until te, in ns
      -  self.tcst_power_gain = 3  # Time where gain is selected, in ns

    Exponential gain (correct for intrinsic attenuation)
      -  self.alpha_exp_gain = 1   # Exponential order, should be close to the slope of the log amplitude decrease
      -  self.t0_exp_gain = 0      # Gain applied from time t0, in ns
      -  self.te_exp_gain = 37     # Gain applied until te, in ns
    '''
    def __init__(self, alpha_power_gain = 1, te_power_gain = 37, tcst_power_gain = 3, \
                 alpha_exp_gain = 1, t0_exp_gain = 0, te_exp_gain = 37 ):
        # See RGPR gain functions: https://emanuelhuber.github.io/RGPR/02_RGPR_tutorial_basic-GPR-data-processing/#amplitude-gain
        
        # Power gain (correct for geometrical spreading)
        self.alpha_power_gain = alpha_power_gain # Power order
        self.te_power_gain = te_power_gain   # Gain applied until te, in ns
        self.tcst_power_gain = tcst_power_gain  # Time where gain is selected, in ns

        # Exponential gain (correct for intrinsic attenuation)
        self.alpha_exp_gain = alpha_exp_gain   # Exponential order, should be close to the slope of the log amplitude decrease
        self.t0_exp_gain = t0_exp_gain      # Gain applied from time t0, in ns
        self.te_exp_gain = te_exp_gain     # Gain applied until te, in ns

    def info(self):
        #provide info for current method
        pass     

    def to_dict(self):
        return{
            'alpha_power_gain': self.alpha_power_gain,
            'te_power_gain'   : self.te_power_gain ,
            'tcst_power_gain' : self.tcst_power_gain,
            'alpha_exp_gain'  : self.alpha_exp_gain,
            't0_exp_gain'     : self.t0_exp_gain ,
            'te_exp_gain'    : self.te_exp_gain 
                }

   

class MixedPhaseDeconvolutionConfig:
    '''
    Process-config:  MixedPhaseDeconvolutionConfig
    Includes
    1) Deconv
    2) Band-pass filter to remove high-frequency noise
    3) Trace scaling
    See https://github.com/emanuelhuber/RGPR/blob/master/R/deconv.R
        
    - self.tWin_start = 10  # start time (in ns) for the time window where deconvolution is applied (and wavelet is estimated)
    - self.tWin_end = 37    # end time (in ns) for the time window where deconvolution is applied (and wavelet is estimated)
    - self.wtr = 5          # number of neighorough traces to be combine into a "super trace"
    - self.mu = 0.00001     # White noise percentage
    - #nf                   # Filter length (calculated from time window 
    # Bandpass Butterworth filter parameters
    - self.f1_bandpass = 0.46 # Lowcut frequency in GHz
    - self.f2_bandpass = 1.1 # Highcut frequency in GHz
    - self.poly_order_bandpass = 4 # Polynomial order
    '''
    def __init__(self,tWin_start = 10, tWin_end = 37, wtr = 5, mu = 0.00001,\
                 f1_bandpass = 0.46, f2_bandpass = 1.1, poly_order_bandpass = 4):
        
        self.tWin_start = tWin_start  # start time (in ns) for the time window where deconvolution is applied (and wavelet is estimated)
        self.tWin_end = tWin_end    # end time (in ns) for the time window where deconvolution is applied (and wavelet is estimated)
        self.wtr = wtr          # number of neighorough traces to be combine into a "super trace"
        self.mu = mu     # White noise percentage
        #nf                   # Filter length (calculated from time window 
        # Bandpass Butterworth filter parameters
        self.f1_bandpass = f1_bandpass # Lowcut frequency in GHz
        self.f2_bandpass = f2_bandpass # Highcut frequency in GHz
        self.poly_order_bandpass = poly_order_bandpass # Polynomial order

    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        return {
            'tWin_start': self.tWin_start,
            'tWin_end': self.tWin_end,
            'wtr': self.wtr,
            'mu' : self.mu,
            'f1_bandpass':     self.f1_bandpass,
            'f2_bandpass':self.f2_bandpass,
            'poly_order_bandpass': self.poly_order_bandpass
        }    
    
class HorizontalFilterConfig:
    def __init__(self, method = "median3x3"):
        # Bandpass Butterworth filter parameters
        self.method = method

    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        return {
            'method': self.method
            }

class BandPassFilterConfig:
    def __init__(self, f1_bandpass = 0.46, f2_bandpass = 1.1, poly_order_bandpass = 4):
        # Bandpass Butterworth filter parameters
        self.f1_bandpass = f1_bandpass # Lowcut frequency in GHz
        self.f2_bandpass = f2_bandpass # Highcut frequency in GHz
        self.poly_order_bandpass = poly_order_bandpass # Polynomial order

    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        return {
            'f1_bandpass': self.f1_bandpass,
            'f2_bandpass': self.f2_bandpass,
            'poly_order_bandpass': self.poly_order_bandpass
        }  
    
class LowPassFilterConfig:
    def __init__(self,f1_bandpass = 0.46, f2_bandpass = 1.1, poly_order_bandpass = 4):
        # Bandpass Butterworth filter parameters
        self.f1_bandpass = f1_bandpass # Lowcut frequency in GHz
        self.f2_bandpass = f2_bandpass # Highcut frequency in GHz
        self.poly_order_bandpass = poly_order_bandpass # Polynomial order

    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        return {
            'f1_bandpass': self.f1_bandpass,
            'f2_bandpass': self.f2_bandpass,
            'poly_order_bandpass': self.poly_order_bandpass
        }  
    
class HiPassFilterConfig:
    def __init__(self,f1_bandpass = 0.46, f2_bandpass = 1.1, poly_order_bandpass = 4):
        # Bandpass Butterworth filter parameters
        self.f1_bandpass = f1_bandpass # Lowcut frequency in GHz
        self.f2_bandpass = f2_bandpass # Highcut frequency in GHz
        self.poly_order_bandpass = poly_order_bandpass # Polynomial order

    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        return {
            'f1_bandpass': self.f1_bandpass,
            'f2_bandpass': self.f2_bandpass,
            'poly_order_bandpass': self.poly_order_bandpass
        }  

class BackgroundRemovalConfig:
    def __init__(self, method = 'mean_trace') -> None:
        
        self.method = method # or 'rmBackgroundMatrix'
        if method == 'rmBackgroundMatrix':
            # Trace average removal with RGPR rmBackgroundMatrix routine
            self.method = method
            self.width = 21
            self.trim = 0.2
            self.s = 1
            self.eps = 1
            self.itmax = 5

    def info(self):
        #provide info for current method
        pass

    def to_dict(self):
        if self.method == 'mean_trace':
            return{
                'method': self.method
            }
        else:
            return {
                'method': self.method,
                'width': self.width,
                'trim': self.trim,
                's': self.s,
                'eps': self.eps, 
                'itmax': self.itmax
            }  
        
###########################################################

# Note-to-self:
# should create defaults for GPR types
# create a .json file for each GPR to be read?

class ProcessingConfig:
    def __init__(self, not_processed = False):

        # Default order of processing steps
        if not_processed == False:
            self.processing_order = ["t0_correction",
                                "bandpass_filter",
                                "background_removal",
                                "amplitude_correction_rgpr", 
                                "mixed_phase_deconvolution_rgpr"]
        else:
            self.processing_order = ["not processed data - raw"] 
        

        self.processing_methods_available = {"resample_profile" : "",
                               "dc_shift_correction_rgpr" : "",
                               "t0_correction"    : "",
                               "t0_correction_rgpr"           : "",
                               "lowpass_filter"           : "",
                               "hipass_filter"           : "",
                               "bandpass_filter"           : "",
                               "bandpass_filter_rgpr"      : "",
                               "background_removal"        : "",
                               "background_removal_rgpr"   : "",
                               "horizontal_filter"         : "",
                               "horizontal_filter_rgpr"    : "",
                               "dewow"                     : "",
                               "amplitude_correction"      : "",                               
                               "amplitude_correction_rgpr" : "",
                               "power_gain_rgpr"           : "",
                               "mixed_phase_deconvolution_rgpr" : "",
                               "kirchoff_migration_rgpr": ""} 
        
        self.dt = 100/512 #nano-seconds - default 512 samples
        self.resample = 1 #default

        self.dc_shift_correction = DCshiftCorrectionConfig()
        self.t0_correction = T0CorrectionConfig()
        self.amplitude_correction_rgpr = AmplitudeCorrectionConfig()
        self.amplitude_correction = AmplitudeCorrectionConfig()
        self.mixed_phase_deconvolution = MixedPhaseDeconvolutionConfig()
        self.low_pass_filter = LowPassFilterConfig()
        self.high_pass_filter = HiPassFilterConfig()
        self.band_pass_filter = BandPassFilterConfig()
        self.band_pass_filter_rgpr = BandPassFilterConfig()
        self.background_removal = BackgroundRemovalConfig()
        self.background_removal_rgpr = BackgroundRemovalConfig()
        self.horizontal_filter = HorizontalFilterConfig()
        self.dewow = DewowConfig()
        self.kirchoff_migration = KirchoffMigration()

    def to_json(self,json_file_path):
        if 0:
            config_data = json.dumps(self, default=lambda o: o.__dict__)
        else: #Check best option
            config_data = {
                'processing_order': self.processing_order,
                'processing_methods_available' : self.processing_methods_available,
                'dt': self.dt,
                'resample': self.resample,
                't0_correction'             : self.t0_correction.to_dict(),
                'mixed_phase_deconvolution' : self.mixed_phase_deconvolution.to_dict(),
                'dc_shift_correction'       : self.dc_shift_correction.to_dict(),
                't0_correction'             : self.t0_correction.to_dict(),
                'amplitude_correction_rgpr' : self.amplitude_correction_rgpr.to_dict(),
                'amplitude_correction'      : self.amplitude_correction.to_dict(),
                'mixed_phase_deconvolution' : self.mixed_phase_deconvolution.to_dict(),
                'low_pass_filter'           : self.low_pass_filter.to_dict(),
                'high_pass_filter'          : self.high_pass_filter.to_dict(),
                'band_pass_filter'          : self.band_pass_filter.to_dict(),
                'band_pass_filter_rgpr'     : self.band_pass_filter_rgpr.to_dict(),
                'background_removal'        : self.background_removal.to_dict(),
                'background_removal_rgpr'   : self.background_removal_rgpr.to_dict(),
                'horizontal_filter'         : self.horizontal_filter.to_dict(),
                'dewow'                     : self.dewow.to_dict(),
                'kirchoff_migration'        : self.kirchoff_migration.to_dict()
                # Add more attributes as needed
            }

        with open(json_file_path, 'w') as file:
            json.dump(config_data, file)
            #json.dump(config_data, file, indent=4)

    @classmethod
    def from_json(cls, json_file_path):
        with open(json_file_path, 'r') as file:
            config_data = json.load(file)


        # Extracting processing_order from config_data
        processing_order = config_data.get('processing_order', [])

        # Extracting processing_methods from config_data
        processing_methods_available = config_data.get('processing_methods_available', [])

        # Extracting other parameters from config_data
        dt = config_data.get('dt', 100/512)
        resample = config_data.get('resample', 1)
        
        # Create an instance of ProcessingConfig
        processing_config = cls(not_processed=(len(processing_order) == 1 and processing_order[0] == "not processed data - raw"))
        
        # Set the extracted values
        processing_config.processing_order = processing_order
        processing_config.processing_methods_available = processing_methods_available
        processing_config.dt = dt
        processing_config.resample = resample

        # Set values for DCshiftCorrectionConfig
        dc_shift_config_data = config_data.get('dc_shift_correction', {})
        processing_config.dc_shift_correction = DCshiftCorrectionConfig(**dc_shift_config_data)

        # Set values for T0CorrectionConfig
        t0_correction_config_data = config_data.get('t0_correction', {})
        processing_config.t0_correction = T0CorrectionConfig(**t0_correction_config_data)

        # Set values for AmplitudeCorrectionConfig 
        amp_correction_config_data = config_data.get('amplitude_correction', {})
        processing_config.amplitude_correction      = AmplitudeCorrectionConfig(**amp_correction_config_data)
        processing_config.amplitude_correction_rgpr = AmplitudeCorrectionConfig(**amp_correction_config_data)

        # Set values for LowPassFilterConfig 
        lowPass_config_data = config_data.get('low_pass_filter', {})
        processing_config.low_pass_filter  = LowPassFilterConfig(**lowPass_config_data)

        # Set values for HiPassFilterConfig 
        hiPass_config_data = config_data.get('high_pass_filter', {})
        processing_config.high_pass_filter = HiPassFilterConfig(**hiPass_config_data)

        # Set values for BandPassFilterConfig 
        bandPass_config_data = config_data.get('band_pass_filter', {})
        processing_config.band_pass_filter      = BandPassFilterConfig(**bandPass_config_data)
        bandPass_config_data = config_data.get('band_pass_filter_rgpr', {})
        processing_config.band_pass_filter_rgpr = BandPassFilterConfig(**bandPass_config_data)

        # Set values for BackgroundRemovalConfig
        backgroundRemoval_config_data = config_data.get('background_removal', {})
        processing_config.background_removal = BackgroundRemovalConfig(**backgroundRemoval_config_data)
        backgroundRemoval_config_data = config_data.get('background_removal_rgpr', {})
        processing_config.background_removal_rgpr = BackgroundRemovalConfig(**backgroundRemoval_config_data)

        # Set values for HorizontalFilterConfig
        horFilter_config_data = config_data.get('horizontal_filter', {})
        processing_config.horizontal_filter = HorizontalFilterConfig(**horFilter_config_data)
        horFilter_config_data = config_data.get('horizontal_filter_rgr', {})
        processing_config.horizontal_filter_rgpr = HorizontalFilterConfig(**horFilter_config_data)

        # Set values for DewowConfig
        dewow_config_data = config_data.get('dewow', {})
        processing_config.dewow = DewowConfig(**dewow_config_data)

        # Set values for MixedPhaseDeconvolutionConfig
        mixed_phase_deconvolution_config_data = config_data.get('mixed_phase_deconvolution', {})
        processing_config.mixed_phase_deconvolution = MixedPhaseDeconvolutionConfig(**mixed_phase_deconvolution_config_data)

        # set values for KirchoffMigration
        kirchoff_migration_config_data = config_data.get('kirchoff_migration', {})
        processing_config.kirchoff_migration = KirchoffMigration(**kirchoff_migration_config_data)

        return processing_config

    def info(self):
        #provide info for current method
        pass
    
    def list_available_processing_methods(self) -> dict:
        
        print('Available processing methods:')
        for key, value in self.processing_methods_available.items():
            print(key, value)
    
    def get_available_processing_methods(self):
        return self.processing_methods_available
    
    def list_processing_order(self) -> dict:
        
        i = 1
        if len(self.processing_order) > 0:
            print("Processing steps included:")
            for step in self.processing_order:
                print("  step " + str(i) + ": " + step)
                i = i + 1
            print("  ")
        else:
            print("No Processing included - raw data")



        