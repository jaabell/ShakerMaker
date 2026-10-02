!==============================================================================
! ffsp_wrapper.f90
! Wrapper to integrate FFSP with Python via f2py
! 
! This wrapper encapsulates main FFSP functionality allowing direct calls
! from Python without file I/O overhead
!==============================================================================

subroutine ffsp_run_wrapper( &
    ! Input: Fault parameters
    id_sf_type_in, freq_min_in, freq_max_in, &
    fault_length_in, fault_width_in, &
    x_hypc_in, y_hypc_in, depth_hypc_in, &
    xref_hypc_in, yref_hypc_in, &
    magnitude_in, fc_main_1_in, fc_main_2_in, rv_avg_in, &
    ratio_rise_in, &
    strike_in, dip_in, rake_in, &
    pdip_max_in, prake_max_in, &
    nsubx_in, nsuby_in, &
    nb_taper_TRBL_in, &
    seeds_in, &
    id_ran1_in, id_ran2_in, &
    angle_north_to_x_in, &
    is_moment_in, &
    ! Input: Velocity model
    nlayers_in, &
    vp_in, vs_in, rho_in, thick_in, qa_in, qb_in, &
    ! Output: Dimensions
    n_realizations_out, npts_out, &
    ! Output: Source parameters (all realizations)
    x_out, y_out, z_out, &
    slip_out, rupture_time_out, rise_time_out, peak_time_out, &
    strike_out, dip_out, rake_out, &
    ! Output: Statistics
    ave_tr_out, ave_tp_out, ave_vr_out, err_spectra_out, pdf_out, &
    fc_main_1_out, fc_main_2_out, &
    ! Output: Spectral data dimensions
    ntime_spec_out, nphf_spec_out, lnpt_spec_out, &
    ! Output: STF time domain
    stf_time_out, stf_out, &
    ! Output: Spectrum frequency domain
    freq_spec_out, moment_rate_out, dcf_out, &
    ! Output: Octave-averaged spectrum
    freq_center_out, logmean_synth_out, logmean_dcf_out &
)
    
    use sp_sub_f
    use fdtim_2d
    use time_freq
    implicit NONE
    
    !--------------------------------------------------------------------------
    ! Input parameters
    !--------------------------------------------------------------------------
    integer, intent(in) :: id_sf_type_in
    real, intent(in) :: freq_min_in, freq_max_in
    real, intent(in) :: fault_length_in, fault_width_in
    real, intent(in) :: x_hypc_in, y_hypc_in, depth_hypc_in
    real, intent(in) :: xref_hypc_in, yref_hypc_in
    real, intent(in) :: magnitude_in, fc_main_1_in, fc_main_2_in, rv_avg_in
    real, intent(in) :: ratio_rise_in
    real, intent(in) :: strike_in, dip_in, rake_in
    real, intent(in) :: pdip_max_in, prake_max_in
    integer, intent(in) :: nsubx_in, nsuby_in
    integer, dimension(4), intent(in) :: nb_taper_TRBL_in
    integer, dimension(3), intent(in) :: seeds_in
    integer, intent(in) :: id_ran1_in, id_ran2_in
    real, intent(in) :: angle_north_to_x_in
    integer, intent(in) :: is_moment_in
    
    ! Velocity model
    integer, intent(in) :: nlayers_in
    real, dimension(nlayers_in), intent(in) :: vp_in, vs_in, rho_in, thick_in, qa_in, qb_in
    
    !--------------------------------------------------------------------------
    ! Output parameters - Source data
    !--------------------------------------------------------------------------
    integer, intent(out) :: n_realizations_out, npts_out
    
    ! Output arrays - f2py will calculate dimensions automatically
    real, intent(out) :: x_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: y_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: z_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: slip_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: rupture_time_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: rise_time_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: peak_time_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: strike_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: dip_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: rake_out(nsubx_in*nsuby_in, id_ran2_in-id_ran1_in+1)
    
    ! Statistics per realization
    real, intent(out) :: ave_tr_out(id_ran2_in-id_ran1_in+1)
    real, intent(out) :: ave_tp_out(id_ran2_in-id_ran1_in+1)
    real, intent(out) :: ave_vr_out(id_ran2_in-id_ran1_in+1)
    real, intent(out) :: err_spectra_out(id_ran2_in-id_ran1_in+1)
    real, intent(out) :: pdf_out(id_ran2_in-id_ran1_in+1)
    
    !--------------------------------------------------------------------------
    ! Output parameters - Spectral data (for best realization)
    !--------------------------------------------------------------------------
    integer, intent(out) :: ntime_spec_out, nphf_spec_out, lnpt_spec_out
    
    ! STF time domain - will be allocated based on ntime from time_freq module
    real, intent(out) :: stf_time_out(131072)  ! Max size (will use only ntime_spec_out)
    real, intent(out) :: stf_out(131072, id_ran2_in-id_ran1_in+1)
    
    ! Spectrum frequency domain
    real, intent(out) :: freq_spec_out(65536)  ! Max size (will use only nphf_spec_out)
    real, intent(out) :: moment_rate_out(65536, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: dcf_out(65536)
    
    ! Octave-averaged spectrum
    real, intent(out) :: freq_center_out(17)  ! Max size (will use only lnpt_spec_out)
    real, intent(out) :: logmean_synth_out(17, id_ran2_in-id_ran1_in+1)
    real, intent(out) :: logmean_dcf_out(17)
    
    !--------------------------------------------------------------------------
    ! Local variables
    !--------------------------------------------------------------------------
    integer :: idum1, idum2, idum3
    integer :: idum1_master, idum2_master, idum3_master
    integer :: nsource, i, j, k, i_real
    real :: drp, rdip, rstrike, cosx, sinx, str_ref, dip_ref, area_sub
    real :: xij, yij, xs, ys, xps, yps, zps, factor, rakei, dipi
    real :: ave_tr, ave_tp, ave_vr, err_spectra, pdf
    real, intent(out) :: fc_main_1_out, fc_main_2_out
    
    !--------------------------------------------------------------------------
    ! 0. Clean up previous state (CRITICAL for repeated calls)
    !--------------------------------------------------------------------------
    ! Deallocate arrays from previous runs
    if (allocated(slip)) deallocate(slip, rstm, rptm, pktm, rpvel, beta, amu, &
                        taper, rtx, rtz, rake_prt, dip_prt, amz_prt, depth_source, lrtp)
    if (allocated(freq)) deallocate(freq)

    !--------------------------------------------------------------------------
    ! 1. Assign parameters to sp_sub_f module variables
    !--------------------------------------------------------------------------
    id_sf_type = id_sf_type_in
    flx = fault_length_in
    fwy = fault_width_in
    x_hypc = x_hypc_in
    y_hypc = y_hypc_in
    depth_hypc = depth_hypc_in
    Moment_o = magnitude_in
    fc_main_1 = fc_main_1_in
    fc_main_2 = fc_main_2_in
    nsubx = nsubx_in
    nsuby = nsuby_in
    
    ! Calculate dimensions (for output)
    n_realizations_out = id_ran2_in - id_ran1_in + 1
    npts_out = nsubx * nsuby
    
    !--------------------------------------------------------------------------
    ! 2. Setup velocity model
    !--------------------------------------------------------------------------
    layer = nlayers_in
    
    if (allocated(vvp)) deallocate(vvp, vvs, roh, thk, qp, qs)
    allocate(vvp(layer+2), vvs(layer+2), roh(layer+2))
    allocate(thk(layer+2), qp(layer+2), qs(layer+2))
    
    do i = 1, layer
        vvp(i) = vp_in(i)
        vvs(i) = vs_in(i)
        roh(i) = rho_in(i)
        thk(i) = thick_in(i)
        qp(i) = qa_in(i)
        qs(i) = qb_in(i)
    enddo
    
    !--------------------------------------------------------------------------
    ! 3. Convert Magnitude to Moment (if needed)
    !--------------------------------------------------------------------------
    if (Moment_o < 12.0) Moment_o = 10**(1.5*Moment_o + 9.05)
    Mw = (alog10(Moment_o) - 9.05) / 1.5
    
    !--------------------------------------------------------------------------
    ! 4. Calculate corner frequencies (as in ffsp_dcf_v2.f90)
    !--------------------------------------------------------------------------
    if (Mw < 5.3) then
        fc_main_1 = 10**(1.474 - 0.415*Mw)
    else
        fc_main_1 = 10**(2.375 - 0.585*Mw)
    endif
    fc_main_2 = 10**(3.250 - 0.5*Mw)
    fc_main_1_out = fc_main_1
    fc_main_2_out = fc_main_2
    
    !--------------------------------------------------------------------------
    ! 5. Initialize main fault
    !--------------------------------------------------------------------------
    call mainfault(dip_in, freq_min_in, freq_max_in, rv_avg_in, &
                   ratio_rise_in, nb_taper_TRBL_in)

    ! The axes and DCF target are identical for every realization.
    call compute_spectral_common(fc_main_1, fc_main_2, &
         ntime_spec_out, nphf_spec_out, lnpt_spec_out, &
         stf_time_out, freq_spec_out, dcf_out, &
         freq_center_out, logmean_dcf_out)
    
    !--------------------------------------------------------------------------
    ! 6. Setup geometric parameters
    !--------------------------------------------------------------------------
    drp = 4.0 * atan(1.0) / 180.0
    rstrike = strike_in * drp
    rdip = dip_in * drp
    cosx = cos(angle_north_to_x_in * drp)
    sinx = sin(angle_north_to_x_in * drp)
    str_ref = (ncxp - 0.5) * dx
    dip_ref = (ncyp - 0.5) * dy
    
    area_sub = 1.e-15
    if (is_moment_in == 3) area_sub = area_sub / (dx * dy)
    
    !--------------------------------------------------------------------------
    ! 7. Save master seeds (FOR PARALLELIZATION REPRODUCIBILITY)
    !--------------------------------------------------------------------------
    idum1_master = seeds_in(1)
    idum2_master = seeds_in(2)
    idum3_master = seeds_in(3)
    
    !--------------------------------------------------------------------------
    ! 8. Loop over realizations
    !--------------------------------------------------------------------------
    i_real = 0
    do nsource = id_ran1_in, id_ran2_in
        i_real = i_real + 1
        
        ! Derive the random stream from the master seeds and the GLOBAL realization
        ! id.  This must not depend on the MPI rank, batch limits, or call history.
        !
        ! TRACEABILITY (seed audit, 2026-07): ran3 in ffsp_tool.f is Numerical
        ! Recipes ran1 and keeps iv/iy in SAVE variables.  On a fresh process a
        ! positive idum enters its initialization branch and max(-idum,1) maps
        ! every positive seed to 1.  Consequently, the first realization handled
        ! by every MPI rank used to start from the same slip random field.
        !
        ! A negative idum1 forces an explicit reset before EVERY realization.  The
        ! first realization uses -1 to retain the historical reference FFSP stream;
        ! subsequent realizations use seed1 plus the global-id offset.  idum2 and
        ! idum3 intentionally keep the reference positive-offset convention: they
        ! continue from the known state established by idum1 inside this realization.
        if (nsource == 1) then
            idum1 = -1
        else
            idum1 = -abs(idum1_master + (nsource - 1) * 10000)
        endif
        idum2 = idum2_master + (nsource - 1) * 20000
        idum3 = idum3_master + (nsource - 1) * 30000
   
        ! Generate random field
        call random_field_with_spectra(idum1, idum2, idum3, ave_tr, ave_tp, ave_vr, &
             err_spectra, stf_out(1:ntime_spec_out,i_real), &
             moment_rate_out(1:nphf_spec_out,i_real), &
             logmean_synth_out(1:lnpt_spec_out,i_real))
        
        ! Calculate PDF (quality metric)
        pdf = ((log(ave_tr) - log(rstm_mean + pktm_mean)) / 0.1)**2.0
        pdf = pdf + ((log(ave_tp) - log(pktm_mean)) / 0.1)**2.0
        pdf = pdf + err_spectra
        
        ! Save statistics
        ave_tr_out(i_real) = ave_tr
        ave_tp_out(i_real) = ave_tp
        ave_vr_out(i_real) = ave_vr
        err_spectra_out(i_real) = err_spectra
        pdf_out(i_real) = pdf
        
        ! Save source parameters
        do i = 1, nsubx
            do j = 1, nsuby
                k = (i - 1) * nsuby + j
                
                ! Calculate coordinates
                xij = (i - 0.5) * dx - str_ref
                yij = (j - 0.5) * dy - dip_ref
                xs = xij * cos(rstrike) - yij * sin(rstrike) * cos(rdip)
                ys = xij * sin(rstrike) + yij * cos(rstrike) * cos(rdip)
                xps = xref_hypc_in + xs * cosx + ys * sinx
                yps = yref_hypc_in - xs * sinx + ys * cosx
                zps = depth_hypc + yij * sin(rdip)
                
                ! Conversion factor
                factor = 1.0
                if (is_moment_in > 1) factor = area_sub / amu(k)
                
                ! Perturbations
                dipi = dip_in + dip_prt(k) * pdip_max_in
                rakei = rake_in + rake_prt(k) * prake_max_in
                
                ! Save to output arrays (in meters)
                x_out(k, i_real) = xps * 1000.0
                y_out(k, i_real) = yps * 1000.0
                z_out(k, i_real) = zps * 1000.0
                slip_out(k, i_real) = slip(k) * factor
                rupture_time_out(k, i_real) = rptm(k)
                rise_time_out(k, i_real) = rstm(k)
                peak_time_out(k, i_real) = pktm(k)
                strike_out(k, i_real) = strike_in
                dip_out(k, i_real) = dipi
                rake_out(k, i_real) = rakei
            enddo
        enddo
    enddo
    
end subroutine ffsp_run_wrapper

!==============================================================================
! compute_spectral_common
! Calculates axes and the DCF target shared by every realization.
!==============================================================================
subroutine compute_spectral_common(fc1, fc2, &
     ntime_out, nphf_out, lnpt_out, &
     stf_time, freq_spec, dcf, freq_center, logmean_m)
    
    use time_freq
    implicit NONE
    
    ! Input
    real, intent(in) :: fc1, fc2
    
    ! Output dimensions
    integer, intent(out) :: ntime_out, nphf_out, lnpt_out
    
    ! Output arrays
    real, dimension(*), intent(out) :: stf_time
    real, dimension(*), intent(out) :: freq_spec, dcf
    real, dimension(*), intent(out) :: freq_center, logmean_m
    
    ! Local variables
    integer :: i, j, i1, i2
    real :: tim, sum
    
    ! Set output dimensions from time_freq module
    ntime_out = ntime
    nphf_out = nphf
    lnpt_out = lnpt - 1
    
    !--------------------------------------------------------------------------
    ! 1. Calculate shared time axis
    !--------------------------------------------------------------------------
    do i = 1, ntime
        tim = (i - 1) * dt
        stf_time(i) = tim
    enddo
    
    !--------------------------------------------------------------------------
    ! 3. Calculate DCF target and save frequency spectrum
    !--------------------------------------------------------------------------
    do i = 1, nphf
        freq(i) = (i - 1 + 1.0e-5) * df
        freq_spec(i) = freq(i)
        dcf(i) = 1.0 / ((1.0 + (freq(i)/fc1)**4.0)**0.25) / &
                        ((1.0 + (freq(i)/fc2)**4.0)**0.25)
    enddo
    
    !--------------------------------------------------------------------------
    ! 4. Calculate octave-averaged spectrum
    !--------------------------------------------------------------------------
    do i = 1, lnpt - 1
        i1 = 2**(i-1)
        i2 = 2**i - 1
        
        ! Log-mean DCF target
        sum = 0.0
        do j = i1, i2
            sum = sum + log(dcf(j))
        enddo
        logmean_m(i) = sum / (i2 - i1 + 1)
        
        ! Center frequency of octave
        freq_center(i) = 0.5 * (freq(i1) + freq(i2))
    enddo
    
end subroutine compute_spectral_common
