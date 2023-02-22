
def power_phase_time(start, end):
    """
    Create Fourier-power and phase images for the given time range.
    return: power, phase 
    """
    BP = BrightPoint()
    BP.load(interval=False, force=False)
    BP.periodomap(range=[start, end])
    return BP.power[:,:,:], BP.phase[:,:,:]



if __name__ == '__main__':
    BP = BrightPoint()
    BP.load(interval=False, force=False)

    """
    ・The length of the frequency domain becomes half of the input time domain.

    ・interval --- Number of time bins used for Short-Time Fast Fourier Transform (STFFT).
                   It is better to set the interval to an odd number to obtain the exact median date.
    ・overl    --- Proportion of the number of overlapping time bins to the 'interval' between each iteration of STFFT.
    ・step     --- Number of the time bins between the starting dates of n-th and (n+1)th STFFT.
    ・r0, r1   --- Lists of beginning/ending dates' indices of each STFFT within BP.nt. Hence r0[n+1] - r0[n] == sep.
    ・stfft_date --- Date represents each STFFT. Calculated as the median of the input dates for the FFT.
    ・power_time    --- Fourier-power images data. Shape of [#x, #y, #freq, #length(r0)].
    ・phase_time    --- Fourier-phase images data. Shape of [#x, #y, #freq, #length(r0)].
    """

    # ====== Controls ======
    interval = 401
    overl = 0
    # ======================

    step = int(interval*(1-overl))
    # List of the starting dates of STFFT
    r0 = np.arange(0, BP.nt, step)
    # List of the ending dates of STFFT
    r1 = r0 + interval
    # Make sure r1 does not to exceed the last observation date.
    ii = np.where(r1 <= (BP.nt-1))[0]
    r0 = r0[ii]
    r1 = r1[ii]

    # Define the date represents each STFFT as the median of the interval.
    if interval % 2 == 0: interval += 1
    stfft_date = [BP.jd[int(i)] for i in (r0+interval-1)/2]

    power_time0 = np.zeros((BP.nx, BP.ny, int(interval/2), len(r0)))
    phase_time0 = np.zeros((BP.nx, BP.ny, int(interval/2), len(r0)))
    power_time1 = np.zeros_like(power_time0)
    phase_time1 = np.zeros_like(phase_time0)
    power_time2 = np.zeros_like(power_time0)
    phase_time2 = np.zeros_like(phase_time0)

    print('\n==================================================\n')
    print('Single Threading')

    start0 = time.time()
    for t in range(len(r0)):
        print(f'\n{t+1}/{len(r0)}')
        power, phase = power_phase_time(r0[t], r1[t])
        power_time0[:,:,:,t] = power
        phase_time0[:,:,:,t] = phase
    end0 = time.time()

    print('\n==================================================\n')
    print('Multi Threading') 
    
    start1 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(power_phase_time, r0, r1))
    for t, result in enumerate(results):
        power_time1[:,:,:,t] = result[0]
        phase_time1[:,:,:,t] = result[1]
    end1 = time.time()
    
    print('\n==================================================\n')
    print('Multi Processing')

    start2 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(power_phase_time, r0, r1))
    for t, result in enumerate(results):
            power_time2[:,:,:,t] = result[0]
            phase_time2[:,:,:,t] = result[1]
    end2 = time.time()

    print('\n==================================================\n')

    print('Single Threading (逐次処理):  {:.2f} sec\n'.format(end0-start0))
    print('Multi Threading (並行処理):  {:.2f} sec\n'.format(end1-start1))
    print('Multi Processing (並列処理):  {:.2f} sec\n'.format(end2-start2))