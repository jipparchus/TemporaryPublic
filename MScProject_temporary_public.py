class DataBase:
	"""
	Avoided inheritance of BrightPoint class.
	Attributes:
		path
		path_aia
		path_aia_backup
		bandpass
		saved_dates
		saved_dates_fmt
		jd
		nodata_dates
		df_load
		df_scope

	"""
	def __init__(self, BPnumber):
		self.BPnumber = BPnumber
		self.path = os.path.join(PROJECT_HOME, 'DataBase') + os.path.sep
		self.path_aia = os.path.join(self.path + 'AIA') + os.path.sep
		self.path_load_pkl = os.path.join(PROJECT_HOME, f'{self.BPnumber} Complete', 'df_load.pkl')
		#self.path_load_pkl = os.path.join(self.path_aia, 'df_load.pkl')
		self.path_scope_pkl = os.path.join(PROJECT_HOME, f'{self.BPnumber} Complete', 'df_scope.pkl')
		#self.path_scope_pkl = os.path.join(self.path_aia, 'df_scope.pkl')
		if os.path.isdir(self.path_load_pkl): self.aiaload(force=False)

		self.aiacmap = {'94a': 'blueviolet','131a': 'darkcyan','171a': 'orange','193a': 'sienna','211a': 'purple','304a': 'red','335a': 'royalblue'}
		
	# path: str = os.path.join(PROJECT_HOME, 'DataBase') + os.path.sep
	# path_aia: str = os.path.join(path + 'AIA') + os.path.sep
	#path_aia_backup: str = os.path(path + 'AIA_backup') + os.path.sep

	# def __post_init__(self):
	# 	print(f'PATH for the DataBase: {self.path}')
	# 	if not os.path.isdir(self.path_aia): os.makedirs(self.path_aia)
		#if not os.path.isdir(self.path_aia_backup): os.makedirs(self.path_aia_backup)
	
	def aiadownload(self, startdate, enddate, interval, channels):
		"""
		Download SDO/AIA fits data and save in self.path_aia.
		startdate, enddate --- With format of '2020-01-01T13:59:40'. Do not include decimals in seconds.
		interval --- Unit of seconds. Sampling frequency of data. The minimum interval is 11.9 seconds.
		"""
		print('Downloading SDO/AIA data...')
		format = "%Y-%m-%dT%H:%M:%S"
		dt_s = datetime.strptime(startdate, format)
		dt_e = datetime.strptime(enddate, format)
		dt_delta = (dt_e - dt_s).seconds
		if dt_delta > 11.9:
			for i in range(dt_delta//int(interval)):
				print(f'{i}/{dt_delta//int(interval)}')
				startdate_new = dt_s + timedelta(seconds=int(interval)*i)
				enddate_new = startdate_new + timedelta(seconds=11.9)
				path_fits = os.path.join(self.path_aia, str(startdate_new).replace(':','-'))
				if not os.path.isdir(path_fits): os.makedirs(path_fits)

				startdate_new = str(startdate_new).replace(' ', 'T')
				enddate_new = str(enddate_new).replace(' ', 'T')

				for w in channels:
					# qr = ev_sun.get_data(startdate_new,enddate_new,instrument='aia',wavelnth=w,get=False)
					# filename = ev_sun.get_data(qr,get=True,dir_data=dir_data)
					filename = get_data(startdate_new, enddate_new, wavelnth=w, \
						dir_data=path_fits, instrument='aia', get=True, limit_time=True, first=True)
					print(filename)
		else:
			path_fits = os.path.join(PROJECT_HOME, 'AIA', startdate.replace(':','-')) + os.path.sep
			if not os.path.isdir(path_fits): os.system('mkdir ' + path_fits)
			for w in channels:
				filename = get_data(startdate, enddate, wavelnth=w, \
						dir_data=path_fits, instrument='aia', get=True, limit_time=True, first=True)
			print(filename)
	
	def aiaload(self, **kwargs):
		"""
		Load fits data and create pandas DataFrame to be saved as a pickle file. Use this method everytime after a new dataset is downloaded!
		IF the df_load.pkl is already there, just load the pkl file and add the attributes.
		date_ref: Observation date in 193A referenced to pick up other bandpasses. (i.e.) -> BrightPoint.date.
		index_ref: Indices referenced in the FOV matching. (i.e.) -> BrightPoint.index
		data193: SDO/AIA 193A image data to be compared with other channels. (i.e.) -> BrightPoint.data
		"""
		force = kwargs.pop('force', False)
		
		if force:
			data_ref = kwargs.pop('data_ref', None)
			date_ref = data_ref.date
			index_ref = data_ref.index
			data193 = data_ref.data

			print('Loading SDO/AIA data...')
			self.bandpass = ['94', '131', '171', '211', '335', '304', '193']
			self.saved_dates = [i for i in np.sort(os.listdir(self.path_aia)) if '.' not in i]
			self.nodata_dates = []
			self.df_load = pd.DataFrame()
			self.df_load['DATE-OBS'] = self.saved_dates
			_df = pd.DataFrame(index=self.df_load.index, columns=self.bandpass)
			self.df_load = pd.concat([self.df_load, _df], axis=1)

			for e, dd in enumerate(self.saved_dates):
				print(f'({e+1}/{len(self.saved_dates)}) {dd}')
				files = []
				lis = os.listdir(os.path.join(self.path_aia, dd))
				for ll in lis:
					if ll.endswith('.fits') & ('aia' in ll):
						files.append(os.path.join(self.path_aia, dd, ll))
				
				if len(files) == 0:
					# If no data on the given date, remove the date from list of observation date
					print(f'No data!! {dd}')
					self.nodata_dates.append(dd)
					print(self.nodata_dates)

					# Cancell the following operations and skip to the next dd
					continue

				# Number of bandpasses: 193A + others
				nbandpass = len(files) + 1

				# Load image data from the fits format and header information
				img_raw, index = read_data(files, normalize=True)

				# Match observation date
				date_obs = [datetime.strptime(index[i]['DATE-OBS'],'%Y-%m-%dT%H:%M:%S.%f') for i in range(nbandpass-1)]
				date_obs_ref = [datetime.strptime(date_ref[i],'%Y-%m-%dT%H:%M:%S.%f') for i in range(len(date_ref))]
				date_match_idx = [np.argmin([abs(i-j) for j in date_obs_ref]) for i in date_obs]
				# 'date_match_idx' must be the same over the bandpasses.
				date_match_idx = date_match_idx[0]

				# Match fov (field-of-view) for the first bandpass
				print(f'Matching fov referencing the original BP data on {date_ref[date_match_idx]}...')
				data, hdr = same_fov(img_raw[:,:,0], index[0], index_ref[date_match_idx])
				self.df_load.at[e, str(hdr['WAVELNTH'])] = {'data': data, 'header': hdr}

				# Match fov (other bandpasses if there are any)
				if len(files) >= 2:
					for i in range(1, nbandpass-1):
						data,hdr = same_fov(img_raw[:,:,i], index[i], index_ref[date_match_idx])
						self.df_load.at[e, str(hdr['WAVELNTH'])] = {'data': data, 'header': hdr}
				
				# 193A data from BrightPoint object
				self.df_load.at[e, '193'] = {'data': data193[:,:,date_match_idx], 'header': index_ref[date_match_idx]}
			
			for dd in self.nodata_dates:
				self.saved_dates.remove(dd)
			print(f'nodata_dates len: {len(self.nodata_dates)}')
			print(f'saved_dates len: {len(self.saved_dates)}')

			self.df_load.to_pickle(self.path_load_pkl)
		
		else:
			self.df_load = pd.read_pickle(self.path_load_pkl)
			self.saved_dates = self.df_load['DATE-OBS']
			self.bandpass = self.df_load.columns[1:].values

		self.saved_dates_fmt = [i.replace(' ','T')[:-8]+i.replace('-',':')[-8:] for i in self.saved_dates]
		self.jd = anytim2jd(self.saved_dates_fmt)

		print('Fits Files Loaded:')
		print(self.df_load)
		return self.df_load

	
	def scope(self, *args, **kwargs):
		"""
		Focus on a part of whole disk image data and create a pandas DataFrame for each bandpass in time series.
		'width' MUST be an ODD number!!
		(e.g.)
			DB.scope(x,y,pixel=True)
			DB.scope(xc,yc,width,super_pixel=True)
			DB.scope(xc,yc,width,width_bg,super_pixel_bg=True)
		modes:
			pixel: Purely one pixel.
			super_pixel: Average over super pixel.
			super_pixel_bg: Background subtraction with super pixel.

			xxxxx
			xooox
			xooox
			xooox
			xxxxx
			x: background
		"""
		if len(args) < 2: raise ValueError('At least 2 arguments (x-centre and y-centre) are required!')
		self.xc = args[0]
		self.yc = args[1]
		pixel = kwargs.pop('pixel', False)
		super_pixel = kwargs.pop('super_pixel', False)
		super_pixel_bg = kwargs.pop('super_pixel_bg', False)
		gauss = kwargs.pop('gauss', False)
		
		if pixel:
			super_pixel = False
			super_pixel_bg = False
			self.width = 1
			self.width_bg = 0
			gauss = False

		elif super_pixel:
			super_pixel_bg = False
			self.width = args[2]
			self.width_bg = 0
			gauss = False

		elif super_pixel_bg:
			super_pixel_bg
			self.width = args[2]
			self.width_bg = args[3]
		
		else: raise ValueError('Invalid mode!!')
		if self.width % 2 == 0: raise ValueError("'width' MUST be an ODD number!!")

		if gauss:
			def gaussian(x, y, xc, yc, sigma):
				return 1/(2*np.pi*sigma**2)*np.exp(-0.5*((x-xc)**2+(y-yc)**2)/sigma**2)

			gauss_mask = np.zeros((self.width+self.width_bg*2, self.width+self.width_bg*2))
			for w in range(self.width+self.width_bg*2):
				for h in range(self.width+self.width_bg*2):
					gauss_mask[w, h] = gaussian(w, h, (self.width+self.width_bg*2)//2, (self.width+self.width_bg*2)//2, self.width/2)
		
		config = {'xc':self.xc,'yc':self.yc,'width':self.width,'width_bg':self.width_bg,'pixel':pixel,'super_pixel':super_pixel,'super_pixel_bg':super_pixel_bg,'gauss':gauss}
		print(f'\nscope config:\n{config}')

		# Nan is replaced to True, if not Nan, replaced to False
		#self.df_load = pd.read_pickle(self.path_load_pkl)
		self.df_scope = self.df_load.notnull()
		
		negative_value = []
		for i in self.df_scope.index:
			for b in self.bandpass:
				if self.df_scope.at[i, b]: # if not Nan in df_load: do the following
					# Check the pixel unit and the exposure time for that specific date of observation.
					exptime = self.df_load.at[i, b]['header']['EXPTIME']
					self.df_scope.at[i, f'{b} EXPTIME'] = exptime
					pixlunit = self.df_load.at[i, b]['header']['PIXLUNIT']
					sample_data = self.df_load.at[i, b]['data'][self.xc-self.width//2-self.width_bg:self.xc+self.width//2+self.width_bg+1, self.yc-self.width//2-self.width_bg:self.yc+self.width//2+self.width_bg+1]
					if pixlunit != 'DN/S':
						sample_data = sample_data / exptime
					if gauss: sample_data = sample_data * gauss_mask

					if self.width_bg != 0:
						centre = np.mean(sample_data[self.width_bg:-self.width_bg, self.width_bg:-self.width_bg])
						edge = (np.sum(sample_data) - centre * self.width ** 2) / (np.size(sample_data) - self.width ** 2)
					else:
						centre = np.mean(sample_data)
						edge = 0
					
					signal = centre - edge
					# if the background subtraction end up with negative value, substitute 10000
					if signal < 0:
						signal = 10000
						negative_value.append([i, b])
					self.df_scope.at[i, b] = signal
				else: continue
		
		# replace 10000 with the minimum value in the bandpass
		if len(negative_value) != 0:
			for j in negative_value:
				self.df_scope[j[0], j[1]] = np.min(self.df_scope[j[1]])
		
		self.df_scope['DATE-OBS'] = self.saved_dates_fmt
		self.df_scope = self.df_scope.replace(False, pd.NA)
		self.df_scope.to_pickle(self.path_scope_pkl)

		print('Scope Data:')
		print(self.df_scope)
		return self.df_scope


	def ls_periodogram(self, modes, coord, width, width_bg, **kwargs):
		"""
		Input - input for scope method, x and y coordinates, fixed integers width and width_bg.
		Output - list of mean correlations per coordinates. The values are saved as correlation_map.pkl under self.path_aia.

		Apply Lomb-Scargle periodogram. Evaluate the corelations of frequency domain among bandpasses.

		modes: format for self.scope (e.g.) modes = {'pixel': False, 'super_pixel':False, 'super_pixel_bg': True, 'gauss': True}
		coords: list of [self.xc, self.yc] to apply the scope and ls_periodogram methods
		width: odd int
		width_bg: width for background
		**kwargs: plot: plot LSpower vs frequency plot, phase-folded data, and corelation among the bandpasses.
				  select_freq: 'auto' or frequency in (Hz). set which frequency to pick up to plot phase-folded data.
		"""
		aiacmap = {'94a': 'blueviolet','131a': 'darkcyan','171a': 'orange','193a': 'sienna','211a': 'purple','304a': 'red','335a': 'royalblue'}

		plot = kwargs.pop('plot', False)
		select_freq = kwargs.pop('select_freq', 0.00123)
		self.df_load = pd.read_pickle(self.path_load_pkl)
		self.df_ls_spectrum = pd.DataFrame()

		notnull_idx = self.df_load.index[self.df_load['193'].notnull()]

		if plot: fig_ls = plt.figure('LS Periodograms')
		
		if plot:
			gs_spectrum_phase_corr_coord = fig_ls.add_gridspec(1, 4, wspace=0.1, hspace=0, width_ratios=[2,1,1,1])
			gs_spectrum = gs_spectrum_phase_corr_coord[0].subgridspec(2, 1, wspace=0, hspace=0, height_ratios=[1.5,1])
			ax_spectrum = fig_ls.add_subplot(gs_spectrum[0])
			ax_spectrum_zoom = fig_ls.add_subplot(gs_spectrum[1])
			ax_phase = fig_ls.add_subplot(gs_spectrum_phase_corr_coord[1])
			ax_correlation = fig_ls.add_subplot(gs_spectrum_phase_corr_coord[2])
			ax_coord = fig_ls.add_subplot(gs_spectrum_phase_corr_coord[3])

			ax_spectrum.set_xlabel('Frequency (mHz)')
			ax_spectrum.set_ylabel('Lomb-Scargle Power')
			ax_spectrum.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
			ax_spectrum.set_xlim([0.2, 6.])
			ax_spectrum.set_ylim([0, 1.])
			ax_spectrum.set_aspect('auto')

			ax_spectrum_zoom.set_xlabel('Frequency (mHz)')
			#ax_spectrum_zoom.set_ylabel('Lomb-Scargle Power')
			ax_spectrum_zoom.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
			ax_spectrum_zoom.set_xlim([0.5, 2.])
			ax_spectrum_zoom.set_ylim([0, 1.])
			ax_spectrum_zoom.set_aspect('auto')

			ax_phase.set_xlabel('Folded-Phase')
			ax_phase.set_ylabel('Signal Intensity')
			ax_phase.set_xlim([0., 1.])
			ax_phase.set_ylim([0., 1.])
			ax_phase.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
			ax_phase.set_aspect('equal')

			ax_correlation.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
			ax_correlation.set_aspect('equal')

			ax_coord.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
			ax_coord.set_aspect('equal')

		alpha = 0.2
		freqs = []
		powers = []

		self.xc = coord[0]
		self.yc = coord[1]
		self.width = width
		self.df_scope = self.scope(self.xc,self.yc,self.width,width_bg,pixel=modes['pixel'],super_pixel=modes['super_pixel'],super_pixel_bg=modes['super_pixel_bg'],gauss=modes['gauss'])
		self.df_scope_norm = self.df_scope
		# Save the info of normalise factor as it is used in DEM analysis.
		self.df_data_norm_factor = pd.DataFrame()

		# Plot the box to indicate the area of scope
		if plot:
			ax_coord.imshow((self.df_load.at[0, '193']['data']).T, cmap='jet')
			ax_coord.invert_yaxis()
			patch = patches.Rectangle(xy=(self.xc, self.yc), width=self.width, height=self.width, linewidth=1, edgecolor='white', facecolor='none', angle=0)
			bbox_props = dict(boxstyle="square,pad=0", linewidth=1, facecolor='white', edgecolor='white')
			ax_coord.text(75, 175, f'x={self.xc}, y={self.yc}, width={self.width}', ha="left", va="bottom", rotation=0, size=10, bbox=bbox_props)
			ax_coord.add_patch(patch)

		# 	evp.window()
		# 	evpm = evp.multi(4, 2)
		# _m = 1

		# Normalise the data by maximum value in the bandpass
		for b in self.bandpass:
			norm_factor = np.max(self.df_scope_norm.loc[:, b])
			self.df_data_norm_factor.at[0, f'{b} SIGNAL-MAX'] = norm_factor
			self.df_scope_norm.loc[:, b] = self.df_scope_norm.loc[:, b]/norm_factor

			# # Plot the box to indicate the area of scope
			# if plot:
			# 	evp.multi(evpm, _m)
			# 	evp.plot_image(self.df_load.at[0, b]['data'], cmap=f'sdoaia{b}')
			# 	evp.plot_box([self.xc-self.width//2, self.xc+self.width//2],[self.yc-self.width//2, self.yc+self.width//2], color='white')
			# 	_m += 1

		t_rel = (self.jd - self.jd[0]) * 24 * 3600 * u.second

		# Get the dominant frequency from 193A bandpass
		# notnull_idx = self.df_scope_norm.index[self.df_scope_norm['193'].notnull()]
		freq, power = LombScargle(t_rel[notnull_idx], self.df_scope_norm['193'][notnull_idx]).autopower(minimum_frequency=1/np.ptp(t_rel[notnull_idx]),nyquist_factor=1, method='fastchi2')

		if select_freq == 'auto':
			# Limit the range to find the dominant frequency
			lowest_freq_idx = np.where(freq.value>0.0007)[0][0]
			highest_freq_idx = np.where(freq.value>0.002)[0][0]
			best_freq = freq[np.argmax(power[lowest_freq_idx:highest_freq_idx])+lowest_freq_idx]
		else: best_freq = select_freq / u.second

		# Plot the LS periodogram
		# if plot:
		# 	fig, ax = plt.subplots()
		# 	fig.suptitle(f'Lomb-Scargle Periodogram {self.xc, self.yc}, {self.width}x{self.width}, {[k for k, v in modes.items() if v == True]}')
		# 	# left, bottom, width, height
		# 	inset_zoom = fig.add_axes([0.3, 0.5, 0.3, 0.38])
		# 	inset_phase = fig.add_axes([0.6, 0.5, 0.3, 0.38])

		# DataFrame for phase and phase-folded data
		self.df_ls_phase = pd.DataFrame()

		for b in self.bandpass:
			#if not (b=='193')|(b=='304')|(b=='171')|(b=='211'): continue
			notnull_idx = self.df_scope_norm.index[self.df_scope_norm[b].notnull()]
		
			freq, power = LombScargle(t_rel[notnull_idx], self.df_scope_norm[b][notnull_idx]).autopower(minimum_frequency=1/np.ptp(t_rel[notnull_idx]),nyquist_factor=1, method='fastchi2')
			freqs.append(freq)
			powers.append(power)
			#print(freq[1]-freq[0])

			# Set best frequency manually for each bandpass.
			#best_freq = freq[np.argsort(np.abs(freq.value-0.000689))[0]]
			# Highest LS Power for period showrter than 30 minutes is defined as the signal frequency
			# lowest_freq_idx = np.argsort(np.abs(freq.value-0.0005))[0]
			#lowest_freq_idx = np.argsort(freq.value>0.0005)[0]
			#best_freq = freq[np.argmax(power[lowest_freq_idx:])]

			# (data/period) % 1 -> period-folded data
			phase = t_rel[notnull_idx] * best_freq
			phase_fold = (t_rel[notnull_idx] * best_freq) % 1
			_df_phase = pd.DataFrame({f'{b} PHASE': phase, f'{b} PHASE-FOLDED': phase_fold, f'{b} NORM-DATA': self.df_scope_norm[b][notnull_idx], f'{b} EXPTIME': self.df_scope[f'{b} EXPTIME'][notnull_idx]})
			self.df_ls_phase = pd.concat([self.df_ls_phase, _df_phase], axis=1)
			if b=='193':
				plt.plot(t_rel[notnull_idx]*best_freq, self.df_scope_norm[b][notnull_idx])
				plt.scatter(phase_fold, self.df_scope_norm[b][notnull_idx])
			phase_fit = np.linspace(0, 1)
			y_fit = LombScargle(t_rel[notnull_idx], self.df_scope_norm[b][notnull_idx]).model(t=phase_fit / best_freq, frequency=best_freq)
			#_ = np.argmin(np.abs(1e3*freq.value-2.))

			if plot:
				alpha += 0.1
				# Plot main periodogram
				ax_spectrum.plot(1e3 * freq, power, label=f'{b}A', alpha=alpha, linewidth=3, color=aiacmap[f'{b}a'])
				# Zoomed spectrum plot
				#ax_spectrum_zoom.plot(1e3 * freq[:_], power[:_], alpha=alpha, linewidth=2, color=aiacmap[f'{b}a'])
				ax_spectrum_zoom.plot(1e3 * freq, power, alpha=alpha, linewidth=2, color=aiacmap[f'{b}a'])
				# Phase-folded data plot
				ax_phase.scatter(phase_fold, self.df_scope_norm[b][notnull_idx], label=f'{b}A', alpha=alpha, color=aiacmap[f'{b}a'], s=18)
				ax_phase.plot(phase_fit, y_fit, color=aiacmap[f'{b}a'])
				
				# # Zoomed spectrum plot
				# inset_zoom.plot(1e3 * freq[:_], power[:_], label=f'{b}A', alpha=alpha, linewidth=2)
				# # inset_zoom.plot(1e3 * best_freq * np.ones(10), np.arange(0, 0.5, 0.5/10), linewidth=3, linestyle=':', color='black', label=f'{1e3*best_freq.value:.3g} (mHz)')
				# # Phase-folded data plot
				# inset_phase.scatter(phase_fold, self.df_scope_norm[b][notnull_idx], label=f'{b}A', alpha=alpha)
				# inset_phase.plot(phase_fit, y_fit)

		if plot:
			ax_spectrum_zoom.plot(1e3 * best_freq * np.ones(10), np.arange(0, 0.7, 0.7/10), linewidth=3, linestyle=':', color='black', label=f'{1e3*best_freq.value:.3g} (mHz) = {1/best_freq.value/60:.3g} (minutes)')
			ax_spectrum_zoom.legend(loc='upper right', fancybox=False, edgecolor="black", bbox_to_anchor=(1,1), ncol=4)
			ax_spectrum.legend(loc='upper right', fancybox=False, edgecolor="black", bbox_to_anchor=(1,1), ncol=4)
			# inset_zoom.set_xlabel('Frequency (mHz)')
			# inset_zoom.set_ylabel('Lomb-Scargle Power')
			# inset_zoom.plot(1e3 * best_freq * np.ones(10), np.arange(0, 0.5, 0.5/10), linewidth=3, linestyle=':', color='black', label=f'{1e3*best_freq.value:.3g} (mHz)')
			# inset_zoom.legend(loc='upper center', borderaxespad=0., ncol=4)
			# inset_phase.set_title(f'Phase-folded data at {1e3*best_freq.value:.3g} (mHz) = {1/best_freq.value/60:.3g} (minutes)')
			# inset_phase.set_xlabel('Phase')
			# inset_phase.set_ylabel('Signal Intensity')
			

		# Summerise the frequency and LS power in a table
		_df = pd.DataFrame()
		len_freq = np.min([len(f) for f in freqs])
		for eb, b in enumerate(self.bandpass):
			_df[f'{b} FREQ'] = freqs[eb][:len_freq].value
			_df[f'{b} BEST-FREQ'] = np.ones_like(len_freq) * best_freq
			_df[f'{b} POWER'] = powers[eb][:len_freq]
		self.df_ls_spectrum = pd.concat([self.df_ls_spectrum, _df], axis=1)

		# Correlations
		df_correlation = pd.DataFrame()
		label_lis = []
		for eb, b in enumerate(self.bandpass):
			label_lis.append(f'{b}A')
			df_correlation = pd.concat([df_correlation, self.df_ls_spectrum[[f'{b} POWER']]], axis=1)
			df_correlation.rename(columns={f'{b} POWER': f'{b}A'}, inplace=True)
		self.correlation_matrix = df_correlation.corr()

		if plot:
			ax_correlation.imshow(self.correlation_matrix, cmap='coolwarm', norm=Normalize(vmin=-1, vmax=1))
			#pp = ax_correlation.colorbar(corr_plot, ax=ax1, orientation="vertical")
			#pp.set_clim(-1,1)
			ax_correlation.set_xticks(np.arange(len(label_lis)))
			ax_correlation.set_xticklabels(label_lis)
			ax_correlation.set_yticks(np.arange(len(label_lis)))
			ax_correlation.set_yticklabels(label_lis)

			for (x, y), val in np.ndenumerate(self.correlation_matrix):
				ax_correlation.text(x, y, f'{val:.2g}', ha="center", va="center", color="w")	
		
		print('Lomb-Scalge Periodogram Power Spectrum:')
		print(self.df_ls_spectrum)

		print('Phase Data:')
		print(self.df_ls_phase)
		
		return self.correlation_matrix


	def phase_mean_std(self):
		"""
		Lomb-Scalge periodogram phase folded data fitting. Find standard deviation and mean for small division of phase-folded data.
		"""
		self.df_ls_phase_mean_std = pd.DataFrame()
		division_unit = 10
		for b in self.bandpass:
			head = 0
			phase_mean = []
			phase_std = []
			data_mean = []
			data_std = []
			exptime_mean = []
			df_ls_phasef_asc = self.df_ls_phase[[f'{b} PHASE-FOLDED', f'{b} NORM-DATA', f'{b} EXPTIME']].dropna().sort_values(f'{b} PHASE-FOLDED', ascending=True).reset_index()
			while head + division_unit <= len(df_ls_phasef_asc):
				_ = df_ls_phasef_asc[head:head+division_unit]
				phase_mean.append(np.mean(_[f'{b} PHASE-FOLDED']))
				phase_std.append(np.std(_[f'{b} PHASE-FOLDED']))
				data_mean.append(np.mean(_[f'{b} NORM-DATA']))
				data_std.append(np.std(_[f'{b} NORM-DATA']))
				exptime_mean.append(np.mean(_[f'{b} EXPTIME']))
				head += division_unit//2
			_ = pd.DataFrame({f'{b} PHASE-MEAN': phase_mean, f'{b} PHASE-STD': phase_std, f'{b} DATA-MEAN': data_mean, f'{b} DATA-STD': data_std, f'{b} EXPTIME-MEAN': exptime_mean})
			self.df_ls_phase_mean_std = pd.concat([self.df_ls_phase_mean_std, _], axis=1)
		print('Mean/Std of Phase-Folded Data')
		print(self.df_ls_phase_mean_std)
		return self.df_ls_phase_mean_std
	

	def dem(self, *args, **kwargs):
		"""
		Pass signal intensity data in bandpasses and the indices as lists. If errorbar==True, data_in -> DATA-MEAN + random.uniform(0,1) * DATA-STD. Repeat DEM for a given time.
		REMEMBER: Each bandpass data plotted in phase-folded data is normalised by its own maximum value.
		"""
		self.phase_mean_std()
		DEM_errorbar = kwargs.pop('random_noise', False)
		# If not DEM_errorbar -> do DEM once without random noise.
		if DEM_errorbar:
			nrepeat = int(args[0])
		else: nrepeat = 1

		# Input data for DEM. Array of signals in each bandpass.
		df_phase_match = pd.DataFrame()
		# Each channel has different length of data
		# Get the shortest length
		min_bandpass = self.df_ls_phase_mean_std.notnull().sum(axis=0).idxmin()[:-11]
		# Match phase-mean referencing the min_bandpass
		phase_ref = self.df_ls_phase_mean_std[f'{min_bandpass} PHASE-MEAN'].dropna().values
		df_phase_match['STANDARD PHASE-MEAN'] = phase_ref
		for b in self.bandpass:
			# if b == min_bandpass: continue
			phase2match = self.df_ls_phase_mean_std[f'{b} PHASE-MEAN'].dropna().values
			phase_match_idx = [np.argmin([abs(r-m) for m in phase2match]) for r in phase_ref]
			df_phase_match[f'{b} DATA-MEAN'] = [self.df_ls_phase_mean_std.at[i, f'{b} DATA-MEAN'] * self.df_data_norm_factor[f'{b} SIGNAL-MAX'].iloc[-1] for i in phase_match_idx]
			df_phase_match[f'{b} DATA-STD'] = [self.df_ls_phase_mean_std.at[i, f'{b} DATA-STD'] * self.df_data_norm_factor[f'{b} SIGNAL-MAX'].iloc[-1] for i in phase_match_idx] 
			df_phase_match[f'{b} EXPTIME-MEAN'] = [self.df_ls_phase_mean_std.at[i, f'{b} EXPTIME-MEAN'] for i in phase_match_idx]

		self.df_DEM = pd.DataFrame()
		for r in range(nrepeat):
			out_list = []
			# If not DEM_errorbar, random noise =  0
			if r == 0: rand_noise = 0
			else:
				rand_noise = random.gauss(0, df_phase_match.at[i, f'{b} DATA-STD'])
				while np.min([df_phase_match.at[i, f'{b} DATA-MEAN'] + rand_noise for b in self.bandpass for i in df_phase_match.index]) < 0:
					rand_noise = random.gauss(0, df_phase_match.at[i, f'{b} DATA-STD'])

			for i in df_phase_match.index:
				data_in = []
				index_in = []
				for b in self.bandpass:
					data_in.append(df_phase_match.at[i, f'{b} DATA-MEAN'] + rand_noise)
					# Reconstruct header with minimum info required for DEM. i.e. WAVELNTH, EXPTIME, and PIXLUNIT.
					index_in.append({'WAVELNTH': int(b), 'EXPTIME': df_phase_match.at[i, f'{b} EXPTIME-MEAN'], 'PIXLUNIT': 'DN/S'})
				# DEM
				# print(f'data_in\n{data_in}')
				# print(f'index_in\n{index_in}')
				noise_factor = 1
				los = 1e8 # line of sight in cm
				no304 = True
				out = aiadem(data_in,index_in,
					minT=5.5,
					maxT=7.,
					noerror=False,
					fast=False,
					pos=True,
					diagnostics=True,
					gaussian=True,
					los=los,
					noise_factor=noise_factor,
					no304=no304,
					verbose=True)
				
				if out.LOGT_MAX < 0:
					negative_flag = True
					break
				else:
					out_list.append(out)
					negative_flag = False
			if negative_flag:
				continue
			if r == 0: self.df_DEM['ORIGINAL'] = out_list
			else: self.df_DEM[f'{r} REPEAT'] = out_list

		return self.df_DEM
		


	def ls_summary_plot(self):
		"""
		Plot Lomb-Scalge Periodogram, Phase-Folded Data, and its Mean and Standard Deviation.
		"""
		self.dem(5, random_noise=True)
		
		fig_ls = plt.figure('LS Periodograms')
		gs0 = fig_ls.add_gridspec(1, 2, wspace=0.05, width_ratios=[1, 2])
		gs1 = gs0[1].subgridspec(2, 2, wspace=0.1, hspace=0.1, width_ratios=[1,1], height_ratios=[1,1])
		
		ax_spectrum = fig_ls.add_subplot(gs0[0])
		# left, bottom, width, height
		inset_zoom = fig_ls.add_axes([0.19, 0.5, 0.18, 0.38])
		ax_phase = fig_ls.add_subplot(gs1[0])
		ax_logT_ne = fig_ls.add_subplot(gs1[1])
		ax_correlation = fig_ls.add_subplot(gs1[2])
		ax_coord = fig_ls.add_subplot(gs1[3])

		ax_spectrum.set_xlabel('Frequency (mHz)')
		ax_spectrum.set_ylabel('Lomb-Scargle Power')
		ax_spectrum.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
		ax_spectrum.set_xlim([0.2, 6.])
		ax_spectrum.set_ylim([0, 1.])
		ax_spectrum.set_aspect('auto')

		inset_zoom.set_xlabel('Frequency (mHz)')
		inset_zoom.set_ylabel('Lomb-Scargle Power')
		inset_zoom.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
		inset_zoom.set_xlim([0.5, 1.4])
		inset_zoom.set_ylim([0, 0.7])
		inset_zoom.set_aspect('auto')

		ax_phase.set_xlabel('Folded-Phase')
		ax_phase.set_ylabel('Signal Intensity')
		ax_phase.set_xlim([0., 1.])
		ax_phase.set_ylim([0.05, 1.])
		ax_phase.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
		ax_phase.set_aspect('equal')

		ax_logT_ne.set_xlabel('logT')
		ax_logT_ne.set_ylabel('ne')
		# ax_logT_ne.set_xlim(5.6, 6.3)
		# ax_logT_ne.set_ylim(1.50*1e9, 2.30*1e9)
		ax_logT_ne.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
		ax_logT_ne.set_aspect('auto')

		ax_correlation.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
		ax_correlation.set_aspect('equal')

		ax_coord.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
		ax_coord.set_aspect('equal')

		for b in self.bandpass:
			bandpass_color = self.aiacmap[f'{b}a']
			best_freq = self.df_ls_spectrum[f'{b} BEST-FREQ'][0].value
			# Lomb-Scalge Power Spectrum
			ax_spectrum.plot(1e3*self.df_ls_spectrum[f'{b} FREQ'], self.df_ls_spectrum[f'{b} POWER'], 
		    	color=bandpass_color)
			# Zoomed Spectrum
			inset_zoom.plot(1e3*self.df_ls_spectrum[f'{b} FREQ'], self.df_ls_spectrum[f'{b} POWER'], 
		    	color=bandpass_color, 
				label=f'{b}A')
			# Phase-Folded Data
			ax_phase.scatter(self.df_ls_phase[f'{b} PHASE-FOLDED'], self.df_ls_phase[f'{b} NORM-DATA'],
		    	color=bandpass_color, alpha=0.5,
				s=5)
			# Phase mean & std
			ax_phase.errorbar(self.df_ls_phase_mean_std[f'{b} PHASE-MEAN'], self.df_ls_phase_mean_std[f'{b} DATA-MEAN'], 
		    	yerr=self.df_ls_phase_mean_std[f'{b} DATA-STD'], 
				color=bandpass_color, alpha=0.7, 
				capsize=5, fmt='.', markersize=10, ecolor=bandpass_color, markeredgecolor=bandpass_color)
		inset_zoom.plot(1e3 * best_freq * np.ones(10), np.arange(0, 0.5, 0.5/10),
			color='black', linewidth=3, linestyle=':', 
			label=f'{1e3*best_freq:.4g} (mHz)')
		inset_zoom.legend(loc='upper center', borderaxespad=0., ncol=4)

		ax_phase.text(0.5, 0.95, f'Period = {1/best_freq/60:.3g} (minutes)', ha="center", va="bottom", rotation=0, size=15)
		
		# DEM
		for e, repeat in enumerate(self.df_DEM.columns):
			logt_max = [self.df_DEM[repeat][i].LOGT_MAX for i in self.df_DEM.index]
			ne = [self.df_DEM[repeat][i].NE for i in self.df_DEM.index]
			if repeat == 'ORIGINAL':
				scat_color = 'red'
				scat_size = 70
				scat_marker = 'X'
				original = ax_logT_ne.scatter(logt_max, ne, marker=scat_marker, color=scat_color, s=scat_size, alpha=1, label='Mean Phase-Folded Data')
				for i in range(1, len(logt_max)):
					xyA = (logt_max[i-1], ne[i-1])
					xyB = (logt_max[i], ne[i])
					coordsA = 'data'
					coordsB = 'data'
					con = patches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB, arrowstyle="-|>", shrinkA=4, shrinkB=4, mutation_scale=18, fc="black")
					arrows = ax_logT_ne.add_artist(con)
					dt = self.df_ls_phase_mean_std[f'{b} PHASE-MEAN'][i] - self.df_ls_phase_mean_std[f'{b} PHASE-MEAN'][i-1]
					dt_txt = ax_logT_ne.text((xyA[0]+xyB[0])/2, (xyA[1]+xyB[1])/2, f'{dt/best_freq:.3g}', ha="left", va="bottom", rotation=0, size=10)
					# np.pi*np.arctan2(xyB[0]-xyA[0],xyB[1]-xyB[1])
			else:
				scat_color = ['Blues', 'Greens', 'Oranges', 'Purples', 'Greys'][e % 6]
				scat_size = 12
				scat_marker='o'
				ax_logT_ne.scatter(logt_max, ne, marker=scat_marker, c=np.arange(len(logt_max)), cmap=scat_color, s=scat_size, alpha=0.8)
		original.set_zorder(1000)
		arrows.set_zorder(1000)
		dt_txt.set_zorder(999)
		

		ax_logT_ne.legend()

		# Correlation
		ax_correlation.imshow(self.correlation_matrix, cmap='coolwarm', norm=Normalize(vmin=-1, vmax=1))
		ax_correlation.set_xticks(np.arange(len(self.correlation_matrix.index.values)))
		ax_correlation.set_xticklabels(self.correlation_matrix.index.values)
		ax_correlation.set_yticks(np.arange(len(self.correlation_matrix.index.values)))
		ax_correlation.set_yticklabels(self.correlation_matrix.index.values)
		for (x, y), val in np.ndenumerate(self.correlation_matrix):
			ax_correlation.text(x, y, f'{val:.2g}', ha="center", va="center", color="w")	

		# Coordinates
		ax_coord.imshow((self.df_load.at[0, '193']['data']).T, cmap='jet')
		ax_coord.invert_yaxis()
		patch = patches.Rectangle(xy=(self.xc-self.width/2, self.yc-self.width/2), width=self.width, height=self.width, linewidth=1.8, edgecolor='white', facecolor='none', angle=0)
		#bbox_props = dict(boxstyle="square,pad=0", linewidth=1, facecolor='white', edgecolor='white')
		#ax_coord.text(100, 175, f'x={self.xc}, y={self.yc}, width={self.width}', ha="left", va="bottom", rotation=0, size=10, bbox=bbox_props)
		ax_coord.annotate(f'x={self.xc}, y={self.yc}, width={self.width}', xy=(self.xc, self.yc), xycoords='data',
					xytext=(20, 177), textcoords='data',
					fontsize=15, weight='bold', color='white',
					arrowprops=dict(arrowstyle='->', connectionstyle="arc", mutation_scale=30, color='white'))

		ax_coord.add_patch(patch)

	




	def periodomap(self, map_mode):
		if map_mode == 'correlation_map':
			pass
		elif map_mode == 'lspower_map':
			pass


def main():
	BPnumber = 'BP149'
	BP = BrightPoint(BPnumber)
	BP.load(interval=False, force=False)
	DB = DataBase(BPnumber)
	new_load = False
	modes = {'pixel': False, 'super_pixel':True, 'super_pixel_bg': False, 'gauss': True}
	plot = False
	correlation_map = False
	if correlation_map:	plot = False
	
	if correlation_map:
		dims = np.shape(BP.data)
		# xx = 15 + np.arange(dims[0]-30)
		# yy = 15 + np.arange(dims[1]-30)
		ww = 164
		xx = ww//2 + np.arange(dims[0]-ww)
		yy = ww//2 + np.arange(dims[1]-ww)
		coord = [[i, j] for j in yy for i in xx]
	
	else:
		#coord = [[83, 62], [82, 53], [80, 46]]
		#coord = [81, 55]
		#coord = [88, 60]
		#coord = [80, 59]
		#coord = [87, 89]
		coord = [88, 90]
		#coord = [106, 60]
	width = 3
	width_bg = 1	

	if new_load: DB.aiaload(force=True, data_ref=BP)
	else: DB.aiaload(force=False, data_ref=BP)

	corr_list = DB.ls_periodogram(modes=modes, coord=coord, width=width, width_bg=width_bg, plot=plot, select_freq=float(0.00123))
	print(corr_list)

	if correlation_map:
		corr_map = np.array(corr_list).reshape((len(xx), len(yy)))
		fig2, ax2 = plt.subplots()
		ax2.imshow(corr_map.T, cmap='jet')
		ax2.invert_yaxis()

		f = open(f"{os.path.join(DB.path_aia, 'correlation_map.pkl')}",'wb')
		pickle.dump(corr_map,f)
		f.close

	DB.ls_summary_plot()


def main2():
	BPnumber = 'BP134'
	BP = BrightPoint(BPnumber)
	BP.load(interval=False, force=False)
	evp.plot_image(BP.data[:,:,0], cmap='sdoaia193')
	print(BP.date[0])
	print(BP.date[-1])
	DB = DataBase(BPnumber)
	startdate = BP.date[0][:19]
	enddate = BP.date[-1][:19]
	wavelnth = ['131', '171', '211', '304', '335', '94']
	DB.aiadownload(startdate=startdate, enddate=enddate, interval=12, channels=wavelnth)
	



if __name__ == '__main__':
	main()
	#mode = 'download/DEM'
	#mode = 'AIAOscillation'
	#mode = 'DEM_plot'
	#mode = 'fft_test'
	"""
	・STFFT --- short time FFT (2022)
	・polar --- polar plot to investigate azimuthal and radial wavenumbers (/11/2022)
	・download/DEM --- Download and create dataset for the Bright Point observed in other wavelength/instruments. (12/12/2022)
						Subsequently try DEM analysis applying a gaussian masking
	・DEM_plot --- plot outcome of time series DEM
	・193Periodicity --- plot and investigate the periodicity of intensity
	"""

