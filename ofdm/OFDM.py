
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d


class OFDM():
    """
    Core module of OFDM. 
    Able to model n UEs and m BSs with TDL channel. 
    Conventional channel estimation and data equalizations are added.
    CVAE based channel estimation and channel tracking are included.
    Channel coding scheme needs to be added to reproduce [e.g. convolution].
    """
 
    def __init__(self, Subcarriers, CP, L_Tap, SNRdb, UEs, BSs , T, OFDM_Symbol_total):
        self.Subcarriers = Subcarriers
        self.CP = CP
        self.PilotLen = OFDM_Symbol_total/T
        self.OFDM_Symbol_total = OFDM_Symbol_total
        self.T = T

        self.L_Tap = L_Tap
        self.SNRdb  = SNRdb 
        self.UEs = UEs
        self.BSs = BSs
        self.allCarriers = np.arange(self.Subcarriers)  # indices of all subcarriers ([0, 1, ... K-1])
        self.dataCarriers = self.allCarriers
    
    def PilotGenGoldSeq (self, PilotLen, N_UEs ):
        x1 = np.zeros([1,15000]);
        x2 = np.zeros([N_UEs,15000]);
        c = np.zeros([N_UEs,10000])
        r = np.zeros([N_UEs, PilotLen], dtype = 'complex_')
        Nc= 1600
        for i in range(15000):
            if i == 1:
                x1[0,i] = 1;

            if i >=31:
                x1[0,i] = np.mod(x1[0,i-31] + x1[0,i+N_UEs-32],2);

        for j in range(N_UEs):
            for i in range(15000):
                if i <= 31:
                    if (i-j)==18:
                        x2[j,i]=1
                else:
                    summation = 0
                    for k in range(N_UEs):
                        summation = summation + x2[j,i+k-31]
                    x2[j,i] = np.mod(summation,2);
        
        for j in range(N_UEs):
            for i in range(10000):
                c[j,i] = np.mod(x1[0,i+Nc]+x2[j,i+Nc],2);

        for k in range(N_UEs):
            for i in range(PilotLen):
                r[k,i] = 1/np.sqrt(2)*(1-2*c[k,2*(i+1)])+ 1j*(1/np.sqrt(2)*(1-2*c[k,2*(i+1)+1]))
        return 2*r

    def DataGenAll(self, codingScheme, OFDM_DataLen, UEs, Subcarriers, Repetition): #OFDM_DataLen = OFDM_Symbol_total - PilotLen
        # TODO: Add QAM64 for reproduction.
        
        if  codingScheme== 'QAM16':
            mu = 4 #number of transfered bits per symbo;
            self.mapping_table = {
            (0,0,0,0) : -3-3j,
            (0,0,0,1) : -3-1j,
            (0,0,1,0) : -3+3j,
            (0,0,1,1) : -3+1j,
            (0,1,0,0) : -1-3j,
            (0,1,0,1) : -1-1j,
            (0,1,1,0) : -1+3j,
            (0,1,1,1) : -1+1j,
            (1,0,0,0) :  3-3j,
            (1,0,0,1) :  3-1j,
            (1,0,1,0) :  3+3j,
            (1,0,1,1) :  3+1j,
            (1,1,0,0) :  1-3j,
            (1,1,0,1) :  1-1j,
            (1,1,1,0) :  1+3j,
            (1,1,1,1) :  1+1j
        }
            
        if  codingScheme== 'QAM':
            mu = 2
            self.mapping_table = {
            (0,0) : -3-3j,
            (0,1) : -3+3j,
            (1,0) : 3-3j,
            (1,1) : 3+3j

        }

        
        DataCoded = np.random.binomial(n=1, p=0.5, size=( UEs, int(Subcarriers/Repetition), OFDM_DataLen, mu))
        OFDMMappedDataCoded = np.squeeze(([[[[[self.mapping_table[tuple(Dlen)]] for _ in range(Repetition)] for  Dlen in sub] for sub in tx] for tx in DataCoded]))
        if Repetition > 1 :
            OFDMMappedDataCoded = OFDMMappedDataCoded.transpose(0,2,1,3)
            OFDMMappedDataCoded = OFDMMappedDataCoded.reshape(OFDMMappedDataCoded.shape[0], OFDMMappedDataCoded.shape[1],-1)
            OFDMMappedDataCoded = OFDMMappedDataCoded.transpose(0,2,1)
        
        return OFDMMappedDataCoded

    def PilotDataloc(self, OFDM_Symbol_total, T):
        OFDM_Symbol_total_range = np.arange(OFDM_Symbol_total)
        #pilotLocation_Tdomain = OFDM_Symbol_total_range[::OFDM_Symbol_total//PilotLen] # Pilots is every (K/P)th T slot.
        pilotLocation_Tdomain = OFDM_Symbol_total_range[::T]
        DataLocation_Tdomain = np.delete(OFDM_Symbol_total_range, pilotLocation_Tdomain)
        return pilotLocation_Tdomain, DataLocation_Tdomain

    def PilotInsertionAll(self, UEs, Subcarriers, OFDMMappedData, PilotSeqs, OFDM_Symbol_total, T):
        OFDM_Symbol_total_range = np.arange(OFDM_Symbol_total)
        #pilotLocation_Tdomain = OFDM_Symbol_total_range[::OFDM_Symbol_total//PilotLen] # Pilots is every (K/P)th T slot.
        pilotLocation_Tdomain = OFDM_Symbol_total_range[::T]
        DataLocation_Tdomain = np.delete(OFDM_Symbol_total_range, pilotLocation_Tdomain)
        InsertedPilotDataMatrix = np.zeros([UEs, Subcarriers, OFDM_Symbol_total], dtype=complex)
        for tx in range(UEs):
            InsertedPilotDataMatrix[tx,:,pilotLocation_Tdomain]=PilotSeqs[tx,:]
        InsertedPilotDataMatrix[:,:,DataLocation_Tdomain]=OFDMMappedData
        return InsertedPilotDataMatrix, OFDM_Symbol_total_range
    
    def PilotInsertionOne(self, UEs, Subcarriers, OFDMMappedData, PilotSeqs, OFDM_Symbol_total, pilotLocation_Tdomain, DataLocation_Tdomain):
        InsertedPilotDataMatrix = np.zeros([UEs, Subcarriers, OFDM_Symbol_total], dtype=complex)
        for tx in range(UEs):
            InsertedPilotDataMatrix[tx,:,pilotLocation_Tdomain]=PilotSeqs[tx,:]
        InsertedPilotDataMatrix[:,:,DataLocation_Tdomain]=OFDMMappedData
        return InsertedPilotDataMatrix
    
    def FFTMatrix (self,n):
        return np.fft.fft(np.eye(n))
 
    def IFFTMatrix (self,n):
        return np.fft.ifft(np.eye(n))
   
    def AddingCPIFFT(self, InsertedPilotDataMatrix, CP):
        UEs, Subcarriers, OFDM_Symbol_total = InsertedPilotDataMatrix.shape
        IFFTInsertedPilotDataMatrix = np.squeeze([np.fft.ifft(InsertedPilotDataMatrix[tx], axis=0) for tx in range (UEs)])
        CPIFFTInsertedPilotDataMatrix = np.squeeze([ [ [ np.hstack([IFFTInsertedPilotDataMatrix[tx,-CP:,symbol] ,
           IFFTInsertedPilotDataMatrix[tx,:,symbol] ]) ]
           for symbol in range(OFDM_Symbol_total)] for tx in range(UEs)])
        return CPIFFTInsertedPilotDataMatrix
       
    def ChannelLoading(self, datapath,Subcarriers, L_Tap, OFDM_Symbol_total):
        
        
        try:
            mat_contents = sio.loadmat(datapath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the channel.mat file at: {datapath}. Please check the path and ensure the file exists. To generate channel.mat please visit: https://www.mathworks.com/help/5g/ref/nrtdlchannel-system-object.html ")
        except Exception as e:
            raise RuntimeError(f"Failed to load the channel.mat file at: {datapath}. Error: {str(e)}")
        channel = mat_contents['channel']
        channel_reorder = np.transpose(channel,(3,2,1,0))# [BS, UE, L_TAP, OFDM_Symbol_total]
        _, _, _, channel_len = channel_reorder.shape
        Sampling_Rate = int(np.ceil(channel_len / OFDM_Symbol_total))
        chennel_LTAP = channel_reorder[:,:,:L_Tap,::Sampling_Rate]

        
        FFTMatrix = self.FFTMatrix(Subcarriers)
        FFTMatrix_L_TAP = FFTMatrix[:,:L_Tap]
        channelFFT = FFTMatrix_L_TAP @ chennel_LTAP
        
        
        return channel, chennel_LTAP, channelFFT
  
    def Transmission(self, channel, CPIFFTInsertedPilotDataMatrix, L_TAP, SNRdb, CP):
        #channel index= [OFDM_Symbol_total, L_TAP, UE, BS]
        #CPIFFTInsertedPilotDataMatrix index = [UE, OFDM_Symbol_total, NFFTCP]
        
        channel = np.transpose(channel,(3,2,0,1))# [BS, UE, OFDM_Symbol_total, L_TAP]
        channel = channel[:,:,:,0:L_TAP]
        
        BSs, UEs, OFDM_Symbol_total, L_TAP = channel.shape
        _, _, NFFTCP = CPIFFTInsertedPilotDataMatrix.shape
        
        
        SigConvChannelNoNoise = np.sum(np.squeeze( [ [ [ [np.convolve( CPIFFTInsertedPilotDataMatrix[ue,t,:], channel[bs,ue,t,:])
                        ] for t in range(OFDM_Symbol_total)] for ue in range(UEs)] for bs in range(BSs) ] ), axis=1)
        
        SigConvPower =  np.array([ [ [ np.mean(abs(SigConvChannelNoNoise[bs,t,:]**2))
                        ] for t in range(OFDM_Symbol_total)]  for bs in range(BSs) ])
        NoisePower =  SigConvPower * 10**(-SNRdb/10)
        
        SigConvChannelNoise = np.squeeze( [ [ [SigConvChannelNoNoise[bs,t,:] + 
            np.sqrt(NoisePower[bs,t,:]/2) * (np.random.standard_normal(NFFTCP+L_TAP-1)+1j*np.random.standard_normal(NFFTCP+L_TAP-1))   
            ] for t in range(OFDM_Symbol_total)] for bs in range(BSs) ] )
        
        SigConvChannelNoiseRemoveCP = np.squeeze([[[SigConvChannelNoise[bs,t,CP:NFFTCP]] for t in range(OFDM_Symbol_total)] 
                                       for bs in range(BSs)])
        
        SigConvChannelNoiseRemoveCPFFT = np.squeeze([[ np.fft.fft(SigConvChannelNoiseRemoveCP[bs],axis=1)]
                                                   for bs in range(BSs)])  #[BS, OFDM_Symbol_total , NFFT]
                                       
        return np.transpose(SigConvChannelNoiseRemoveCP,(0,2,1)), np.transpose(SigConvChannelNoiseRemoveCPFFT,(0,2,1)), NoisePower 


    def LmmseEstimation(self, h_ls, PilotMatrix, NoisePower):
        sigma_x = np.mean(np.abs(PilotMatrix)**2)
        
        # h_LS: shape (BS, UE, L_TAP, T)
        BS, UE, L_TAP, pilot_len = h_ls.shape
        _, NFFT, _ = PilotMatrix.shape
        h_LS_flat = h_ls.reshape(BS * UE, L_TAP, pilot_len)

        # Compute sample covariance for each BS-UE pair
        R_hh = np.array([
            np.cov(h_LS_flat[i], bias=True)  # shape (L_TAP, L_TAP)
            for i in range(BS * UE)
        ]).reshape(BS, UE, L_TAP, L_TAP)
        
        sigma_n2_avg = np.mean(NoisePower)
        
        FFTMatrix = self.FFTMatrix(NFFT)
        FFTMatrix_L_TAP = FFTMatrix[:,:L_TAP]
                    
        ###
        h_lmmse = np.zeros((BS, UE, L_TAP, pilot_len), dtype=complex)
        for ue in range(UE):
            for bs in range(BS):
                for t in range(pilot_len):
                    
                    
                    R_h = R_hh[bs, ue]  # shape (L_TAP, L_TAP), or identity * sigma_h if unavailable
                    sigma_h2 = np.mean(np.diag(R_h))  # or just some empirical estimate
                    
                    
                    X = PilotMatrix[ue, :, t]  # (NFFT)
                    XD = np.diag(X) @ FFTMatrix_L_TAP
                    XHX_inv = np.linalg.inv(XD.conj().T @ XD)  # shape (L_TAP, L_TAP)
                    R_h = sigma_h2 * np.eye(XD.shape[1])
                    

                    # LMMSE update (block-wise)
                    # TODO: Also check ( R_h + sigma_n2_avg/sigma_x * eye(L_TAP))
                    G = R_h @ np.linalg.inv(R_h + sigma_n2_avg * XHX_inv)
                    h_lmmse_loc = G @ h_ls[bs, ue, :, t]
                    
                    h_lmmse[bs, ue, :, t] = h_lmmse_loc
        return h_lmmse


    
    def LsEstimation(self, RecSigRemoveCPFFT, InsertedPilotDataMatrix,
      pilotLocation_Tdomain,
      L_TAP,
      NoisePower): 
        
        UE, NFFT, OFDM_Symbol_total = InsertedPilotDataMatrix.shape
        BS, _, _ = RecSigRemoveCPFFT.shape
        
        
        
        RecPilotMatrix = np.squeeze([ [RecSigRemoveCPFFT[bs, :, pilotLocation_Tdomain ] ] 
                                  for bs in range(BS)])
        
        PilotMatrix = np.transpose(np.squeeze([ [InsertedPilotDataMatrix[ue, :, pilotLocation_Tdomain ] ] for ue in range(UE)]), axes=[0,2,1])
        FFTMatrix = self.FFTMatrix(NFFT)
        FFTMatrix_L_TAP = FFTMatrix[:,:L_TAP]
        
        DiagPilot = np.squeeze([[[np.diag(PilotMatrix[ue,:,T]) @ FFTMatrix_L_TAP ]  for T in range(PilotMatrix.shape[-1]) ]for ue in range(UE)])
        #[UE, Pilot_Loc, NFFT, L_TAP]
        Diag_Transmit_Pilot_2D = DiagPilot.transpose(1,2,3,0)# [Pilot_Loc, FFT, L_TAP, UE]
        Diag_Transmit_Pilot_2D = np.reshape(Diag_Transmit_Pilot_2D,
                                            (Diag_Transmit_Pilot_2D.shape[0], Diag_Transmit_Pilot_2D.shape[1],-1), order='F') #[Pilot_Loc, FFT, L_TAP*UE]
        
        h_estimate = np.squeeze([[[np.linalg.inv(Diag_Transmit_Pilot_2D[T].transpose() @
           Diag_Transmit_Pilot_2D[T]) @ Diag_Transmit_Pilot_2D[T].transpose()@
           RecPilotMatrix[bs, T]] for T in range(PilotMatrix.shape[-1]) ]for bs in range(BS)]) #[BS, PILOT LEN, UE*L_TAP ]

        h = np.reshape(h_estimate, (BS,len(pilotLocation_Tdomain), UE, L_TAP))
        h = h.transpose(0,2,3,1)# [BS, UE, L_TAP, PILOT LEN]

        h_lmmse = self.LmmseEstimation(h, PilotMatrix, NoisePower )
        
        clm_range = np.array(pilotLocation_Tdomain)
        clm_rangenew = np.linspace(clm_range.min(), clm_range.max(), OFDM_Symbol_total)
        interpolatef_ls = [[interp1d(clm_range, h[bs,ue], axis=1) for
                        ue in range(UE)] for bs in range(BS)]
        interpolateval_ls = np.squeeze([[interpolatef_ls[bs][ue](clm_rangenew) for
                  ue in range(UE)] for bs in range(BS)]) #[BS, UE, L_TAP, DATA_INTERPOLATED]
        interpolatevalFdomain_ls = np.squeeze([[[FFTMatrix_L_TAP @ interpolateval_ls[bs,
                                 ue]]for ue in range(UE)] for bs in range(BS)])
        
        interpolatef_lmmse = [[interp1d(clm_range, h_lmmse[bs,ue], axis=1) for
                        ue in range(UE)] for bs in range(BS)]
        interpolateval_lmmse = np.squeeze([[interpolatef_lmmse[bs][ue](clm_rangenew) for
                  ue in range(UE)] for bs in range(BS)]) #[BS, UE, L_TAP, DATA_INTERPOLATED]

        interpolatevalFdomain_lmmse = np.squeeze([[[FFTMatrix_L_TAP @ interpolateval_lmmse[bs,
                                 ue]]for ue in range(UE)] for bs in range(BS)])#[BS, UE, NFFT, TOTAL SYMBOLS]
        return interpolateval_ls, interpolatevalFdomain_ls, interpolateval_lmmse, interpolatevalFdomain_lmmse
    
    
        
    def LsEstimationTracking(self, RecSigRemoveCPFFT, InsertedPilotDataMatrix,
      pilotLocation_Tdomain,
      L_TAP): 
        
        #TODO: Add LMMSE as well.
        UE, NFFT, _ = InsertedPilotDataMatrix.shape
        BS, _, _ = RecSigRemoveCPFFT.shape
        
        RecPilotMatrix = np.squeeze([ [RecSigRemoveCPFFT[bs, :, pilotLocation_Tdomain ] ] 
                                  for bs in range(BS)])
        RecPilotMatrix = np.expand_dims(RecPilotMatrix, axis = 1)
        
        PilotMatrix = np.transpose(np.expand_dims( np.squeeze([ [InsertedPilotDataMatrix[ue, :, pilotLocation_Tdomain ] ] for ue in range(UE)]), axis=1), axes=[0,2,1])
        FFTMatrix = self.FFTMatrix(NFFT)
        FFTMatrix_L_TAP = FFTMatrix[:,:L_TAP]
        
        DiagPilot = np.squeeze([[[np.diag(PilotMatrix[ue,:,T]) @ FFTMatrix_L_TAP ]  for T in range(PilotMatrix.shape[-1]) ]for ue in range(UE)])
        DiagPilot = np.expand_dims(DiagPilot, axis=1)
        #[UE, Pilot_Loc, NFFT, L_TAP]
        Diag_Transmit_Pilot_2D = DiagPilot.transpose(1,2,3,0)# [Pilot_Loc, FFT, L_TAP, UE]
        Diag_Transmit_Pilot_2D = np.reshape(Diag_Transmit_Pilot_2D,
                                            (Diag_Transmit_Pilot_2D.shape[0], Diag_Transmit_Pilot_2D.shape[1],-1), order='F') #[Pilot_Loc, FFT, L_TAP*UE]
        
        h_estimate = np.squeeze([[[np.linalg.inv(Diag_Transmit_Pilot_2D[T].transpose() @
           Diag_Transmit_Pilot_2D[T]) @ Diag_Transmit_Pilot_2D[T].transpose()@
           RecPilotMatrix[bs, T]] for T in range(PilotMatrix.shape[-1]) ]for bs in range(BS)]) #[BS, PILOT LEN, UE*L_TAP ]
        h_estimate = np.expand_dims(h_estimate, axis=1)

        h = np.reshape(h_estimate, (BS,len(pilotLocation_Tdomain), UE, L_TAP))
        h = h.transpose(0,2,3,1)# [BS, UE, L_TAP, PILOT LEN]


        interpolatevalFdomain = np.squeeze([[[FFTMatrix_L_TAP @ h[bs,
                                 ue]]for ue in range(UE)] for bs in range(BS)])#[BS, UE, NFFT, PILOT LEN]
        interpolatevalFdomain = np.expand_dims(interpolatevalFdomain, axis=-1)
        return h, interpolatevalFdomain


    
    def Equalizer(self, interpolatevalFdomain, RecSig, DataLocation_Tdomain, SNRdb):
        _, UE, NFFT, _ = interpolatevalFdomain.shape
        CIVFDDL = interpolatevalFdomain[:,:,:,DataLocation_Tdomain] # IFDDL = ChannelinterpolatedvalFdomainDataLocation
        RecSigDL = RecSig[:,:,DataLocation_Tdomain] # RecSigDL = RecSigDataLocation
        CIVFDDL_reorder = CIVFDDL.transpose(3,2,0,1) #[Data len_total, NFFT, BS, UE]
        RecSigDL_reorder =RecSigDL.transpose(2,1,0) # [Dta len_total, NFFT, BS]
        D_len, NFFT, _, UE = CIVFDDL_reorder.shape
        EstimatedData_Reorder = np.squeeze([[[ np.linalg.inv(CIVFDDL_reorder[d][f].transpose() @
            CIVFDDL_reorder[d][f] + 10**(-SNRdb/10)*np.eye(UE)  ) @ CIVFDDL_reorder[d][f].transpose() @
            RecSigDL_reorder[d][f] ] for f in range(NFFT)] for d in range(D_len)])

        EstimatedData = EstimatedData_Reorder.transpose(2,1,0)
        return EstimatedData
    
    def EqualizerTracking(self, interpolatevalFdomain, RecSig, PilotLocation_Tdomain, SNRdb):
        _, UE, NFFT, _ = interpolatevalFdomain.shape
        CIVFDDL = interpolatevalFdomain[:,:,:,[0]] # IFDDL = ChannelinterpolatedvalFdomainDataLocation
        RecSigDL = RecSig[:,:,PilotLocation_Tdomain] # RecSigDL = RecSigDataLocation
        CIVFDDL_reorder = CIVFDDL.transpose(3,2,0,1) #[Data len_total = 1, NFFT, BS, UE]
        RecSigDL_reorder =RecSigDL.transpose(2,1,0) # [Dta len_total = 1, NFFT, BS]
        D_len, NFFT, _, UE = CIVFDDL_reorder.shape
        EstimatedData_Reorder = np.squeeze([[[ np.linalg.inv(CIVFDDL_reorder[d][f].transpose() @
            CIVFDDL_reorder[d][f] + 10**(-SNRdb/10)*np.eye(UE)  ) @ CIVFDDL_reorder[d][f].transpose() @
            RecSigDL_reorder[d][f] ] for f in range(NFFT)] for d in range(D_len)])

        EstimatedData_Reorder = np.expand_dims(EstimatedData_Reorder, axis=0)
        EstimatedData = EstimatedData_Reorder.transpose(2,1,0)
        return EstimatedData

    def ErrorBits(self, EstimatedData, RealData, repetition):
        UE, FFT, Len_Data = EstimatedData.shape
        dictvalues = np.array([val for (key,val) in self.mapping_table.items()]) # symbol's complex values 
        keys = np.array([key for (key,val) in self.mapping_table.items()]) # symbol's complex values 
        len_key = keys.shape[-1]
        
        for ue  in range(EstimatedData.shape[0]):
            for row in range(EstimatedData.shape[1]):
                for clm in range(EstimatedData.shape[2]):
                    EstimatedData[ue,row,clm] = dictvalues[np.argmin (np.abs(EstimatedData[ue,row,clm] - dictvalues)) ]
        #EstimatedData's shape = [UE, FFT, DATA LEN]
        RealData_bits = np.zeros((UE, FFT, Len_Data, len_key))
        EstimatedData_bits = np.zeros((UE, FFT, Len_Data, len_key))
        RealData_categorical = np.zeros((UE, FFT, Len_Data))
        EstimatedData_categorical = np.zeros((UE, FFT, Len_Data))
        for index, value in enumerate(dictvalues):
            RealData_bits[np.where(RealData == value)[0], np.where(RealData == value)[1], np.where(RealData == value)[2],:] = keys[index] 
            EstimatedData_bits[np.where(EstimatedData == value)[0], np.where(EstimatedData == value)[1], np.where(EstimatedData == value)[2],:] = keys[index]
            RealData_categorical[np.where(RealData == value)[0], np.where(RealData == value)[1], np.where(RealData == value)[2]] = index 
            EstimatedData_categorical[np.where(EstimatedData == value)[0], np.where(EstimatedData == value)[1], np.where(EstimatedData == value)[2]] = index
        
        _, _, _, bits = EstimatedData_bits.shape
        
        
        # TODO: Replace with the coding mentioned in the paper [Convolution]
        # TODO: Only support repetition 2. Generalize.
        if repetition > 1:
            EstimatedData_bits_re = EstimatedData_bits.reshape(RealData_bits.shape[0], repetition, int(RealData_bits.shape[1] / repetition), RealData_bits.shape[2], RealData_bits.shape[3], order = 'F')
            RealData_bits_re = RealData_bits.reshape(RealData_bits.shape[0], repetition, int(RealData_bits.shape[1] / repetition), RealData_bits.shape[2], RealData_bits.shape[3], order = 'F')
            EstimatedData_categorical_re = EstimatedData_categorical.reshape(EstimatedData_categorical.shape[0], repetition, int(EstimatedData_categorical.shape[1] / repetition), EstimatedData_categorical.shape[2], order = 'F')
            
            
            a = np.where(EstimatedData_categorical_re[:,0,:,:] == EstimatedData_categorical_re[:,1,:,:])

            # TODO: Add re-sending algo for those wrong packets caught by coding scheme. 
            countCorrectBits = np.count_nonzero(EstimatedData_bits_re[a[0],0,a[1],a[2],:] == RealData_bits_re[a[0],0,a[1],a[2],:])
            TotalBits = len(a[0])* bits
        else:
            countCorrectBits = np.count_nonzero(EstimatedData_bits == RealData_bits)
            TotalBits = EstimatedData_bits.shape[0] * EstimatedData_bits.shape[1] * EstimatedData_bits.shape[2] * EstimatedData_bits.shape[3]
        
        BER = (TotalBits - countCorrectBits)/(TotalBits )
        print(BER)
        return BER        


    def labeling_classifier(self,
                            EstimatedData,
                            RealData,
                            RecSig,
                            interpolatevalFdomain,
                            DataLocation_Tdomain):
        
        Channel_Fdomain_inDataLoc = interpolatevalFdomain[:,:,:,DataLocation_Tdomain]
        RecSig_inDataLoc = RecSig[:,:,DataLocation_Tdomain]
        dictvalues = np.array([val for (key,val) in self.mapping_table.items()]) # symbol's complex values 
        
        EstimatedDataCopy = np.copy(EstimatedData)
        for ue  in range(EstimatedData.shape[0]):
            for row in range(EstimatedData.shape[1]):
                for clm in range(EstimatedData.shape[2]):
                    EstimatedData[ue,row,clm] = dictvalues[np.argmin (np.abs(EstimatedData[ue,row,clm] - dictvalues)) ]
        
        labels_matrix = np.zeros([EstimatedData.shape[1], EstimatedData.shape[2]])
        labels1 = []
        features1 = []
        for row in range(EstimatedData.shape[1]):
            for clm in range(EstimatedData.shape[2]):
                result = np.all(EstimatedData[:,row,clm] == RealData[:,row,clm])
                if result == True:
                    labels_matrix[row, clm] =1
                    labels1.append(1)
                    features1.append(np.concatenate((np.real(EstimatedData[:,row,clm]),
                                                   np.imag(EstimatedData[:,row,clm]),
                                                   np.real(RecSig_inDataLoc[:,row,clm]),
                                                   np.imag(RecSig_inDataLoc[:,row,clm]),
                                                   np.real(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1),
                                                   np.imag(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1))  ))
                    
                else:
                    labels_matrix[row, clm] =0
                    labels1.append(0)
                    features1.append(np.concatenate((    np.real(EstimatedData[:,row,clm]),
                                                        np.imag(EstimatedData[:,row,clm]),
                                                        np.real(RecSig_inDataLoc[:,row,clm]),
                                                        np.imag(RecSig_inDataLoc[:,row,clm]),
                                                        np.real(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1),
                                                        np.imag(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1)     )))
                    
        labels2 = []
        features2 = []
        for bs in range(RecSig_inDataLoc.shape[0]):
            for row in range(EstimatedData.shape[1]):
                for clm in range(EstimatedData.shape[2]):
                    result = np.all(EstimatedData[:,row,clm] == RealData[:,row,clm])
                    if result == True:
                        labels_matrix[row, clm] =1
                        labels2.append(1)
                        features2.append(np.concatenate((np.real(EstimatedData[:,row,clm]),
                                                       np.imag(EstimatedData[:,row,clm]),
                                                       np.expand_dims( np.real(RecSig_inDataLoc[bs,row,clm]), axis=0),
                                                       np.expand_dims( np.imag(RecSig_inDataLoc[bs,row,clm]), axis=0),
                                                       np.real(Channel_Fdomain_inDataLoc[bs,:,row,clm]).reshape(-1),
                                                       np.imag(Channel_Fdomain_inDataLoc[bs,:,row,clm]).reshape(-1))  ))
                        
                    else:
                        labels_matrix[row, clm] =0
                        labels2.append(0)
                        features2.append(np.concatenate((    np.real(EstimatedData[:,row,clm]),
                                                            np.imag(EstimatedData[:,row,clm]),
                                                            np.expand_dims( np.real(RecSig_inDataLoc[bs,row,clm]), axis=0),
                                                            np.expand_dims( np.imag(RecSig_inDataLoc[bs,row,clm]), axis=0),
                                                            np.real(Channel_Fdomain_inDataLoc[bs,:,row,clm]).reshape(-1),
                                                            np.imag(Channel_Fdomain_inDataLoc[bs,:,row,clm]).reshape(-1)     )))
        
        
       
        return np.array(features1), np.array(labels1),   np.array(features2), np.array(labels2)    
  
    def labeling_reconstruction(self,
                                EstimatedData,
                                RealDataOrig,
                                RecSig,
                                interpolatevalFdomain,
                                DataLocation_Tdomain):
        
        Channel_Fdomain_inDataLoc = interpolatevalFdomain[:,:,:,DataLocation_Tdomain]
        RecSig_inDataLoc = RecSig[:,:,DataLocation_Tdomain]
        dictvalues = np.array([val for (key,val) in self.mapping_table.items()]) # symbol's complex values 
        num_cls = len(self.mapping_table)
        
        EstimatedDataLabel = np.copy(EstimatedData)
        EstimatedDataSoft = np.copy(EstimatedData)
        RealDataLabel = np.copy(RealDataOrig)
        for ue  in range(EstimatedDataLabel.shape[0]):
            for row in range(EstimatedDataLabel.shape[1]):
                for clm in range(EstimatedDataLabel.shape[2]):
                   
                    EstimatedDataLabel[ue,row,clm] = np.argmin (np.abs(EstimatedDataLabel[ue,row,clm] - dictvalues)) 
                    RealDataLabel[ue,row,clm] = np.argmin (np.abs(RealDataLabel[ue,row,clm] - dictvalues)) 
                    
        labels_matrix = np.zeros([EstimatedDataLabel.shape[1], EstimatedDataLabel.shape[2]])
        outputs = []
        classes = []
        features1 = []
        
        for row in range(EstimatedDataLabel.shape[1]):
            for clm in range(EstimatedDataLabel.shape[2]):
                result = np.all(EstimatedDataLabel[:,row,clm] == RealDataLabel[:,row,clm])
                if result == True:
                    labels_matrix[row, clm] =1
                    
                    ## labels1.shape =( UE, CLS (modulation length))
                    classes.append(1)
                    outputs.append(np.eye(num_cls)[ np.real(RealDataLabel[:,row,clm]).astype(int)] )
                    
                    ## features1.shape =( UE * CLS (modulation length) + 2*BS + 2*BS*UE)
                    features1.append(np.concatenate((np.real(EstimatedDataSoft[:,row,clm]),
                                                     np.imag(EstimatedDataSoft[:,row,clm]),
                                                   np.real(RecSig_inDataLoc[:,row,clm]),
                                                   np.imag(RecSig_inDataLoc[:,row,clm]),
                                                   np.real(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1),
                                                   np.imag(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1))  ))
                    
                    
                else:
                    labels_matrix[row, clm] =0
                    classes.append(0)
                    outputs.append(np.eye(num_cls)[ np.real(RealDataLabel[:,row,clm]).astype(int)] )
                    features1.append(np.concatenate((np.real(EstimatedDataSoft[:,row,clm]),
                                                     np.imag(EstimatedDataSoft[:,row,clm]),
                                                   np.real(RecSig_inDataLoc[:,row,clm]),
                                                   np.imag(RecSig_inDataLoc[:,row,clm]),
                                                   np.real(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1),
                                                   np.imag(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1))  ))
                
        return np.array(features1), np.array(outputs), np.array(classes), num_cls


    def DataDetectionCVAE(self, labels_matrix, EstimatedData, interpolatevalFdomain,
                          DataLocation_Tdomain, L_TAP):
        UEs, NFFT, _ = EstimatedData.shape

        _, _, _, Total_OFDM_len_Data = interpolatevalFdomain.shape


                
        label_matrix_with_pilot = np.ones((NFFT, Total_OFDM_len_Data))
        for index, i in enumerate(DataLocation_Tdomain):
            label_matrix_with_pilot[:,i] = labels_matrix[:, index]
        
        correct_detection = np.where(labels_matrix ==1)
        
        # TODO: Add the required channel coding scheme [in paper is convolution] to recheck the condition.
        additional_pilot_loc = np.array([DataLocation_Tdomain[i] for i in range(len(DataLocation_Tdomain)) if len(np.where(correct_detection[1] == i)[0]) >= (1+UEs)*L_TAP ])
        New_Data_Loc = np.setdiff1d(DataLocation_Tdomain, additional_pilot_loc)
        New_Pilot_Loc = np.setdiff1d(np.arange(Total_OFDM_len_Data), New_Data_Loc)
        
        return New_Data_Loc, New_Pilot_Loc, additional_pilot_loc, label_matrix_with_pilot
    
    def DataDetectionMLP(self, MLP_model, EstimatedData, RecSig, interpolatevalFdomain, DataLocation_Tdomain, L_TAP, Modulation="QAM16"):
        
        
        UEs, NFFT, _ = EstimatedData.shape
        _, _, _, Total_OFDM_len_Data = interpolatevalFdomain.shape
        Channel_Fdomain_inDataLoc = interpolatevalFdomain[:,:,:,DataLocation_Tdomain]
        RecSig_inDataLoc = RecSig[:,:,DataLocation_Tdomain]
        dictvalues = np.array([val for (key,val) in self.mapping_table.items()]) # symbol's complex values 

        labels_matrix = np.zeros([EstimatedData.shape[1], EstimatedData.shape[2]])

        for ue  in range(EstimatedData.shape[0]):
            for row in range(EstimatedData.shape[1]):
                for clm in range(EstimatedData.shape[2]):
                    EstimatedData[ue,row,clm] = dictvalues[np.argmin (np.abs(EstimatedData[ue,row,clm] - dictvalues)) ]
        
        labels_matrix = np.zeros([EstimatedData.shape[1], EstimatedData.shape[2]])

        features = []
        for row in range(EstimatedData.shape[1]):
            for clm in range(EstimatedData.shape[2]):
                features = np.concatenate((np.real(EstimatedData[:,row,clm]),
                                               np.imag(EstimatedData[:,row,clm]),
                                               np.real(RecSig_inDataLoc[:,row,clm]),
                                               np.imag(RecSig_inDataLoc[:,row,clm]),
                                               np.real(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1),
                                               np.imag(Channel_Fdomain_inDataLoc[:,:,row,clm]).reshape(-1))  )
                labels_matrix[row, clm] = np.argmax(MLP_model.predict(np.expand_dims(features, axis=0)))
                
        label_matrix_with_pilot = np.ones((NFFT, Total_OFDM_len_Data))
        for index, i in enumerate(DataLocation_Tdomain):
            label_matrix_with_pilot[:,i] = labels_matrix[:, index]
        
        correct_detection = np.where(labels_matrix ==1)
        additional_pilot_loc = np.array([DataLocation_Tdomain[i] for i in range(len(DataLocation_Tdomain)) if len(np.where(correct_detection[1] == i)[0]) >= (1+UEs)*L_TAP ])
        New_Data_Loc = np.setdiff1d(DataLocation_Tdomain, additional_pilot_loc)
        New_Pilot_Loc = np.setdiff1d(np.arange(Total_OFDM_len_Data), New_Data_Loc)
    
        return New_Data_Loc, New_Pilot_Loc, additional_pilot_loc, labels_matrix, EstimatedData, label_matrix_with_pilot
    
    def LsEstimation_ForDetectedData(self, RecSigRemoveCPFFT, InsertedPilotDataMatrix,
        pilotLocation_Tdomain,
        L_TAP): 
        
        UE, NFFT, OFDM_Symbol_total = InsertedPilotDataMatrix.shape
        BS, _, _ = RecSigRemoveCPFFT.shape
        
        RecPilotMatrix = np.squeeze([ [RecSigRemoveCPFFT[bs, :, pilotLocation_Tdomain ] ] 
                                  for bs in range(BS)])
        
        PilotMatrix = np.transpose(np.squeeze([ [InsertedPilotDataMatrix[ue, :, pilotLocation_Tdomain ] ] for ue in range(UE)]), axes=[0,2,1])
        FFTMatrix = self.FFTMatrix(NFFT)
        FFTMatrix_L_TAP = FFTMatrix[:,:L_TAP]
        
        DiagPilot = np.squeeze([[[np.diag(PilotMatrix[ue,:,T]) @ FFTMatrix_L_TAP ]  for T in range(PilotMatrix.shape[-1]) ]for ue in range(UE)])
        #[UE, Pilot_Loc, NFFT, L_TAP]
        Diag_Transmit_Pilot_2D = DiagPilot.transpose(1,2,3,0)# [Pilot_Loc, FFT, L_TAP, UE]
        Diag_Transmit_Pilot_2D = np.reshape(Diag_Transmit_Pilot_2D,
                                            (Diag_Transmit_Pilot_2D.shape[0], Diag_Transmit_Pilot_2D.shape[1],-1), order='F') #[Pilot_Loc, FFT, L_TAP*UE]
        
        h_estimate = np.squeeze([[[np.linalg.inv(Diag_Transmit_Pilot_2D[T].transpose() @
           Diag_Transmit_Pilot_2D[T]) @ Diag_Transmit_Pilot_2D[T].transpose()@
           RecPilotMatrix[bs, T]] for T in range(PilotMatrix.shape[-1]) ]for bs in range(BS)]) #[BS, PILOT LEN, UE*L_TAP ]

        h = np.reshape(h_estimate, (BS,len(pilotLocation_Tdomain), UE, L_TAP))
        h = h.transpose(0,2,3,1)# [BS, UE, L_TAP, PILOT LEN]

        clm_range = np.array(pilotLocation_Tdomain)
        clm_rangenew = np.linspace(clm_range.min(), clm_range.max(), OFDM_Symbol_total)
        interpolatef = [[interp1d(clm_range, h[bs,ue], axis=1) for
                        ue in range(UE)] for bs in range(BS)]
        interpolateval = np.squeeze([[interpolatef[bs][ue](clm_rangenew) for
                  ue in range(UE)] for bs in range(BS)]) #[BS, UE, L_TAP, DATA_INTERPOLATED]

        interpolatevalFdomain = np.squeeze([[[FFTMatrix_L_TAP @ interpolateval[bs,
                                 ue]]for ue in range(UE)] for bs in range(BS)])#[BS, UE, NFFT, TOTAL SYMBOLS]
        return interpolateval, interpolatevalFdomain
    
    def LsEstimation_PartiallyDetectedData(self, RecSigRemoveCPFFT, label_matrix_with_pilot,
        InsertedPilotDataMatrix, pilotLocation_Tdomain, L_TAP): 
        
        UE, NFFT, OFDM_Symbol_total = InsertedPilotDataMatrix.shape
        BS, _, _ = RecSigRemoveCPFFT.shape
        
        h_pilot_len = np.zeros((BS, UE, L_TAP, len(pilotLocation_Tdomain)), dtype=np.complex_)
        for index, T in enumerate(pilotLocation_Tdomain):
            
            RecPilotMatrix = np.squeeze([ [RecSigRemoveCPFFT[bs, np.where(label_matrix_with_pilot[:,T] == 1)[0], T] ] 
                                      for bs in range(BS)])
            PilotMatrix = np.squeeze([ [InsertedPilotDataMatrix[ue, np.where(label_matrix_with_pilot[:,T] == 1)[0], T ] ] for ue in range(UE)])
            FFTMatrix = self.FFTMatrix(NFFT)
            FFTMatrix_L_TAP = FFTMatrix[np.where(label_matrix_with_pilot[:,T] == 1)[0],:L_TAP]
            DiagPilot = np.squeeze([[np.diag(PilotMatrix[ue,:]) @ FFTMatrix_L_TAP ]   for ue in range(UE)]) #[UE, NFFT, L_TAP]
            Diag_Transmit_Pilot_2D = DiagPilot.transpose(1,2,0)#[ NFFT, L_TAP, UE]
            Diag_Transmit_Pilot_2D = np.reshape(Diag_Transmit_Pilot_2D,
                                                (Diag_Transmit_Pilot_2D.shape[0], -1), order='F') #[ FFT, L_TAP*UE]
            
            h_estimate = np.squeeze([[np.linalg.inv(Diag_Transmit_Pilot_2D.transpose() @
               Diag_Transmit_Pilot_2D) @ Diag_Transmit_Pilot_2D.transpose()@
               RecPilotMatrix[bs ]] for bs in range(BS)])#[BS, UE*L_TAP ]
            
            h = np.reshape(h_estimate, (BS, UE, L_TAP))# [BS, UE, L_TAP]
            h_pilot_len[:,:,:,index] = h
            
            
        
        clm_range = np.array(pilotLocation_Tdomain)
        clm_rangenew = np.linspace(clm_range.min(), clm_range.max(), OFDM_Symbol_total)
        interpolatef = [[interp1d(clm_range, h_pilot_len[bs,ue], axis=1) for
                        ue in range(UE)] for bs in range(BS)]
        interpolateval = np.squeeze([[interpolatef[bs][ue](clm_rangenew) for
                  ue in range(UE)] for bs in range(BS)]) #[BS, UE, L_TAP, DATA_INTERPOLATED]

        FFTMatrix = self.FFTMatrix(NFFT)
        FFTMatrix_L_TAP = FFTMatrix[:,:L_TAP]
        interpolatevalFdomain = np.squeeze([[[FFTMatrix_L_TAP @ interpolateval[bs,
                                 ue]]for ue in range(UE)] for bs in range(BS)])#[BS, UE, NFFT, TOTAL SYMBOLS]
        return interpolateval, interpolatevalFdomain
    
    