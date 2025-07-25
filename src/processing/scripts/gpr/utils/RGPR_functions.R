build_RGPR_data <- function(profile,freq,dt,resample,TxRx) {
  x <- list(data=profile,freq=freq,dx=1,dz=dt/resample,antsep=TxRx)   # dx= spatial sampling, to be updated after x interpolation is implemented
  profile_R <- as(x, "GPR")
  return(profile_R)
}

process_mixed_phased_deconv_R <- function(Profile_R, tWin_start, tWin_end, wtr, mu, f1, f2) {
  tWin <- c(tWin_start, tWin_end)  # ns - time window on which the wavelet is estimated
  W <- which(depth(Profile_R) > tWin[1] & depth(Profile_R) < tWin[2])
  profile_deconv <- deconv(Profile_R, method="mixed-phase", W = tWin, wtr = wtr, nf = length(W)-1,mu = mu)
  #profile_deconv_x <- profile_deconv@x
  #w_min <- profile_deconv$wmin
  #w_mix <- profile_deconv$wmix
  x7 <- fFilter(profile_deconv$x, f = c(f1*1000, f2*1000), type = "bandpass", plot = FALSE)
  x8 <- traceScaling(x7, type="stat")
  profile_after_deconv <- x8@data
  return(profile_after_deconv)
}

process_mixed_phased_deconv_all_R <- function(Profile_R, tWin_start, tWin_end, wtr, mu, f1, f2) {
  tWin <- c(tWin_start, tWin_end)  # ns - time window on which the wavelet is estimated
  W <- which(depth(Profile_R) > tWin[1] & depth(Profile_R) < tWin[2])
  profile_deconv <- deconv(Profile_R, method="mixed-phase", W = tWin, wtr = wtr, nf = length(W)-1,mu = mu)
  #profile_deconv_x <- profile_deconv@x
  #w_min <- profile_deconv$wmin
  #w_mix <- profile_deconv$wmix
  profile_deconv_wavelet_min_phase <- profile_deconv$wmin
  profile_deconv_wavelet_min_phase_x <- profile_deconv_wavelet_min_phase$x
  profile_deconv_wavelet_min_phase_y <- profile_deconv_wavelet_min_phase$y
  profile_deconv_wavelet_mixed_phase <- profile_deconv$wmix
  profile_deconv_wavelet_mixed_phase_x <- profile_deconv_wavelet_mixed_phase$x
  profile_deconv_wavelet_mixed_phase_y <- profile_deconv_wavelet_mixed_phase$y
  x7 <- fFilter(profile_deconv$x, f = c(f1*1000, f2*1000), type = "bandpass", plot = FALSE)
  x8 <- traceScaling(x7, type="stat")
  profile_after_deconv <- x8@data
  output = list(profile_after_deconv,profile_deconv_wavelet_mixed_phase_x,profile_deconv_wavelet_mixed_phase_y,profile_deconv_wavelet_min_phase_x,profile_deconv_wavelet_min_phase_y)
  return(output)
}

process_bandpass_R <- function(Profile_R, f1, f2) {
  Profile_R_filtered <- fFilter(Profile_R, f=c(f1*1000,f2*1000),type = "bandpass", plotSpec = FALSE)
  Profile_filtered <- Profile_R_filtered@data
  return(Profile_filtered)
}

process_DCshift_R <- function(Profile_R, DCsample_start, DCsample_end) {
  z_DC_corrected <- dcshift(Profile_R,u=DCsample_start:DCsample_end)  #Use the samples between DC_sample_start and DC_sample_end to estimate DC shift
  Profile_DC_corrected <- z_DC_corrected@data
  return(Profile_DC_corrected)
}

process_t0correction_R <- function(Profile_R, threshold_t0) {
  #threshold_t0 <- 0.03 # value to be tested
  tr_res_all_traces_R = Profile_R
  tfb_tr_res_all_traces <- firstBreak(tr_res_all_traces_R, method = "threshold", thr = threshold_t0)
  time0(tr_res_all_traces_R) <- tfb_tr_res_all_traces     # set time0 in GPR object

  tr_res_with_t0_all_traces_R <- time0Cor(tr_res_all_traces_R, method = "pchip")
  Profile_t0_corrected <- tr_res_with_t0_all_traces_R@data

  return(Profile_t0_corrected)
}

process_background_removal_R <- function(Profile_R) {
  Mean_trace_R <- traceStat(Profile_R, FUN = median)
  Mean_trace <- Mean_trace_R@data
  return(Mean_trace)
}

process_horizontal_filter_R <- function(Profile_R) {
  Profile_R_space_filtered <- filter2D(Profile_R, type = "median3x3")
  Profile_space_filtered <- Profile_R_space_filtered@data
  return(Profile_space_filtered)
}

process_gain_power_R <- function(Profile_R,alpha_power_gain,te_power_gain, tcst_power_gain) {
  Profile_after_power_gain <- gain(Profile_R, type = "power", alpha = alpha_power_gain, te = te_power_gain, tcst = tcst_power_gain)
  Profile_space_filtered <- Profile_after_power_gain@data
  return(Profile_space_filtered)
}

process_gain_exp_R <- function(Profile_R,alpha_exp_gain, t0_exp_gain, te_exp_gain) {
  Profile_after_exp_gain <- gain(Profile_R, type = "exp", alpha = alpha_exp_gain, t0 = t0_exp_gain, te = te_exp_gain)
  Profile_space_filtered <- Profile_after_exp_gain@data
  return(Profile_space_filtered)
}

process_amplitude_correction_R <- function(Profile_R,alpha_power_gain,te_power_gain, tcst_power_gain,alpha_exp_gain, t0_exp_gain, te_exp_gain){
  Profile_after_power_gain <- gain(Profile_R, type = "power", alpha = alpha_power_gain, te = te_power_gain, tcst = tcst_power_gain)
  Profile_after_exp_gain <- gain(Profile_after_power_gain, type = "exp", alpha = alpha_exp_gain, t0 = t0_exp_gain, te = te_exp_gain)
  Profile_amp_corr <- Profile_after_exp_gain@data
  return(Profile_amp_corr)
}

process_kirchoff_migration_R <- function(Profile_R, velocity_ns, max_depth_use, dz_use, fdo_use) {
  pre_process = 0
  if (pre_process == 1){
  Profile_after_timeCor <- timeCorOffset(Profile_R)
  Profile_after_upsample <- upsample(Profile_after_timeCor, n = c(3, 1))
  vel(Profile_after_upsample)  <- velocity_ns        # velocity in ns
  
  Profile_migrated <- migrate(Profile_after_upsample, type="kirchhoff", max_depth = 20,
                 dz = 0.01, fdo = 80) #fdo in MHz

  } else {
  vel(Profile_R)  <- velocity_ns        # velocity in ns
  
  Profile_migrated <- migrate(Profile_R, type="kirchhoff", max_depth = max_depth_use,
                 dz = dz_use, fdo = fdo_use)
  }

  post_process = 0

  if (post_process == 1){
  #trace smoothing
  A10 <- filter1D(Profile_migrated, type = "Gaussian", w = 2.5)
  #Automatic gain control
  A11 <- gain(A10, type = "agc", w = 0.55)
  #inverse normal transformations
  #A12 <- traceScaling(A11, type = "invNormal")
  Profile_migrated_ret <- A12@data
  } else {
  Profile_migrated_ret <- Profile_migrated@data
  }


  return(Profile_migrated_ret)
}