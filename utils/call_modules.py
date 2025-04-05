from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
import torch


def mse_calc(interpolateChannel_LTAP, Realchennel_LTAP):
    # BS, UE, L_TAP, Total_time = Realchennel_LTAP.shape

    LSmse_real = mean_squared_error(np.real(np.reshape(interpolateChannel_LTAP, (interpolateChannel_LTAP.shape[0], -1) )),
                             np.real(np.reshape(Realchennel_LTAP, (Realchennel_LTAP.shape[0], -1) )))
    LSmse_imag = mean_squared_error(np.imag(np.reshape(interpolateChannel_LTAP, (interpolateChannel_LTAP.shape[0], -1) )),
                             np.imag(np.reshape(Realchennel_LTAP, (Realchennel_LTAP.shape[0], -1) )))
    LSmse_total = np.sqrt (LSmse_real**2 + LSmse_imag**2)
    return LSmse_total

def mse_calc2(interpolateChannel_LTAP, Realchennel_LTAP):
    # BS, UE, L_TAP, Total_time = Realchennel_LTAP.shape
    
    interpolateChannel_LTAP = interpolateChannel_LTAP.transpose(2,0,1,3)# [L_tap, BS, UE, Total_Time]
    Realchennel_LTAP = Realchennel_LTAP.transpose(2,0,1,3)
    LSmse_real = mean_squared_error(np.real(np.reshape(interpolateChannel_LTAP, (interpolateChannel_LTAP.shape[0], -1) )),
                             np.real(np.reshape(Realchennel_LTAP, (Realchennel_LTAP.shape[0], -1) )))
    LSmse_imag = mean_squared_error(np.imag(np.reshape(interpolateChannel_LTAP, (interpolateChannel_LTAP.shape[0], -1) )),
                             np.imag(np.reshape(Realchennel_LTAP, (Realchennel_LTAP.shape[0], -1) )))
    LSmse_total = np.sqrt (LSmse_real**2 + LSmse_imag**2)
    return LSmse_total

def CONVENTIONAL_CONFIG(ofdm_obj,
              SigConvChannelNoiseRemoveCPFFT,
              InsertedPilotDataMatrix,
              pilotLocation_Tdomain,
              DataLocation_Tdomain,
              L_Tap,
              OFDMMappedData,
              Realchennel_LTAP,
              modscheme,
              T,
              SNRdb,
              repetition,
              NoisePower):

    interpolateChannel_LTAP, interpolatevalFdomain, interpolateChannel_lmmse, interpolatevalFdomain_lmmse = ofdm_obj.LsEstimation(SigConvChannelNoiseRemoveCPFFT,
                    InsertedPilotDataMatrix, pilotLocation_Tdomain, L_Tap, NoisePower) #[DATA_INTERPOLATED, L_TAP, UE, BS]
                
    LSmse_total = mse_calc(interpolateChannel_LTAP, Realchennel_LTAP)
    LMMSEmse_total = mse_calc(interpolateChannel_lmmse, Realchennel_LTAP)
    # LSmselist.append(LSmse_total)
    
    if T != 1 :
        LSEstimatedData = ofdm_obj.Equalizer( interpolatevalFdomain, SigConvChannelNoiseRemoveCPFFT, DataLocation_Tdomain, SNRdb)
        LSEstimatedDataCopy = np.copy(LSEstimatedData)
        LSBER = ofdm_obj.ErrorBits(LSEstimatedDataCopy, OFDMMappedData, repetition)
        
        LmmseEstimatedData = ofdm_obj.Equalizer( interpolatevalFdomain_lmmse, SigConvChannelNoiseRemoveCPFFT, DataLocation_Tdomain, SNRdb)
        LmmseEstimatedDataCopy = np.copy(LmmseEstimatedData)
        LmmseBER = ofdm_obj.ErrorBits(LmmseEstimatedDataCopy, OFDMMappedData, repetition)
    # LSBERlist.append(LSBER)
    return LSEstimatedData, LSmse_total, LmmseEstimatedData, LMMSEmse_total, LSBER, LmmseBER

def CONV_DATA_AIDED_CONFIG(LSEstimatedData,
                           InsertedPilotDataMatrix,
                           ofdm_obj,
                           SigConvChannelNoiseRemoveCPFFT,
                            OFDM_Symbol_total_range,
                            Realchennel_LTAP,
                            OFDMMappedData,
                            SNRdb,
                            T,
                            DataLocation_Tdomain ,
                            l_tap,
                            repetition,
                            NoisePower):
    
    InsertedPilotEstimatedData = np.copy(InsertedPilotDataMatrix)
    for index, i in enumerate(DataLocation_Tdomain):
        InsertedPilotEstimatedData[:,:,i] = LSEstimatedData[:,:, index]
        
    interpolateChannel_LTAP, interpolatevalFdomain, _, _ = ofdm_obj.LsEstimation(
                                                            SigConvChannelNoiseRemoveCPFFT,
                                                            InsertedPilotEstimatedData,
                                                            OFDM_Symbol_total_range,
                                                            l_tap,
                                                            NoisePower) #[DATA_INTERPOLATED, L_TAP, UE, BS]
    
    
    
    LSmse_total = mse_calc(interpolateChannel_LTAP, Realchennel_LTAP)
    
    if T != 1 :
        LSEstimatedData_conv_data_aided = ofdm_obj.Equalizer( interpolatevalFdomain, SigConvChannelNoiseRemoveCPFFT, DataLocation_Tdomain, SNRdb)
        LSBER = ofdm_obj.ErrorBits(LSEstimatedData_conv_data_aided, OFDMMappedData, repetition)
    return LSmse_total, LSBER

def CHANNEL_TRACKING(OFDMMappedTotal,
                     ofdm_obj,
                     Realchennel_LTAP_reorder,
                     OFDM_Symbol_total_range,
                     UEs,
                     Subcarriers,
                     r,
                     CP,
                     SNRdb,
                     L_Tap):
    
    OFDM_Symbol_total = len(OFDM_Symbol_total_range)
    pilotLocation_Tdomain = [OFDM_Symbol_total_range[0]]
    DataLocation_Tdomain = np.delete(OFDM_Symbol_total_range, pilotLocation_Tdomain)
    OFDMMappedData = OFDMMappedTotal[:,:,DataLocation_Tdomain]
    
    InsertedPilotDataMatrix = ofdm_obj.PilotInsertionOne(UEs, Subcarriers, OFDMMappedData, r, OFDM_Symbol_total,
          pilotLocation_Tdomain, DataLocation_Tdomain)  
    
    CPIFFTInsertedPilotDataMatrix = ofdm_obj.AddingCPIFFT(InsertedPilotDataMatrix, CP) 
    
    _, SigConvChannelNoiseRemoveCPFFT, _ = ofdm_obj.Transmission(Realchennel_LTAP_reorder,
                                    CPIFFTInsertedPilotDataMatrix, L_Tap, SNRdb, CP)# [BS, NFFT, OFDM_Symbol_total]
    
    
    interpolateChannel_LTAP, interpolatevalFdomain = ofdm_obj.LsEstimationTracking(
                                                                SigConvChannelNoiseRemoveCPFFT,
                                                                InsertedPilotDataMatrix,
                                                                pilotLocation_Tdomain,
                                                                L_Tap) #[BS, UE, L_TAP, PILOT_LEN = 1]
    interpolateChannel_LTAP_list = []
    interpolatevalFdomain_list = []
    interpolateChannel_LTAP_list.append(interpolateChannel_LTAP)
    interpolatevalFdomain_list.append(interpolatevalFdomain)
    
    InsertedPilotEstimatedData = np.copy(InsertedPilotDataMatrix)
    for symbol_idx in range(1,OFDM_Symbol_total): # first do equalization with the previous estimated channel, then re-estimate channel with the equalized symbol
        pilotLocation_Tdomain = [OFDM_Symbol_total_range[symbol_idx]]
        DataLocation_Tdomain = np.delete(OFDM_Symbol_total_range, pilotLocation_Tdomain)
        
        LSEstimatedData_tracking = ofdm_obj.EqualizerTracking( interpolatevalFdomain, SigConvChannelNoiseRemoveCPFFT, pilotLocation_Tdomain, SNRdb)
        InsertedPilotEstimatedData[:,:,[symbol_idx]] = LSEstimatedData_tracking
        interpolateChannel_LTAP, interpolatevalFdomain = ofdm_obj.LsEstimationTracking(
                                                                        SigConvChannelNoiseRemoveCPFFT,
                                                                        InsertedPilotEstimatedData,
                                                                        pilotLocation_Tdomain,
                                                                        L_Tap) #[BS, UE, L_TAP, PILOT_LEN]
        interpolateChannel_LTAP_list.append(interpolateChannel_LTAP)
        interpolatevalFdomain_list.append(interpolatevalFdomain)
    return interpolateChannel_LTAP_list, interpolatevalFdomain_list

def TF_DATA_AIDED(LSEstimatedData,
                  InsertedPilotDataMatrix,
                  DataLocation_Tdomain,
                  ofdm_obj,
                  MLP,
                  SigConvChannelNoiseRemoveCPFFT,
                  OFDM_Symbol_total_range,
                 OFDMMappedData,
                 Realchennel_LTAP,
                 l_tap,
                 SNRdb,
                 modscheme,
                 NoisePower):
    
    InsertedPilotEstimatedData = np.copy(InsertedPilotDataMatrix)
    for index, i in enumerate(DataLocation_Tdomain):
        InsertedPilotEstimatedData[:,:,i] = LSEstimatedData[:,:, index]
        
    interpolateChannel_LTAP, interpolatevalFdomain = ofdm_obj.LsEstimation(SigConvChannelNoiseRemoveCPFFT,
            InsertedPilotEstimatedData, OFDM_Symbol_total_range,
            l_tap,
            NoisePower) #[DATA_INTERPOLATED, L_TAP, UE, BS]
    ##
    
    features1, labels1, _, _ =  ofdm_obj.labeling_classifier(LSEstimatedData,
                                                            OFDMMappedData,
                                                            SigConvChannelNoiseRemoveCPFFT,
                                                            interpolatevalFdomain,
                                                            DataLocation_Tdomain)
    simple_mlp = MLP([50, 10, 2])
    simple_mlp.compile(optimizer=tf.keras.optimizers.Adam(), loss= tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    
    simple_mlp.fit(features1[:int(0.8*len(features1))], labels1[:int(0.8*len(labels1))], batch_size=20, epochs=50, validation_split=0.2)
    _, New_Pilot_Loc, _, _, _, label_matrix_with_pilot = ofdm_obj.DataDetectionMLP(
                                                            simple_mlp ,
                                                            LSEstimatedData,
                                                            SigConvChannelNoiseRemoveCPFFT,
                                                            interpolatevalFdomain,
                                                            DataLocation_Tdomain,
                                                            l_tap,
                                                            Modulation = modscheme)
    
    # TODO: Replace LS with LMMSE
    interpolateChannel_LTAP, interpolatevalFdomain = ofdm_obj.LsEstimation_ForDetectedData(
            SigConvChannelNoiseRemoveCPFFT,
            InsertedPilotDataMatrix,
            New_Pilot_Loc,
            l_tap) #[DATA_INTERPOLATED, L_TAP, UE, BS]
    LSFullMLPmse_total = mse_calc(interpolateChannel_LTAP, Realchennel_LTAP)

    
    LSEstimatedData = ofdm_obj.Equalizer( interpolatevalFdomain, SigConvChannelNoiseRemoveCPFFT, DataLocation_Tdomain, SNRdb)
    repetition = 2
    LSFullMLPBER = ofdm_obj.ErrorBits(LSEstimatedData, OFDMMappedData, repetition)

    
    interpolateChannel_LTAP, interpolatevalFdomain = ofdm_obj.LsEstimation_PartiallyDetectedData(
            SigConvChannelNoiseRemoveCPFFT,
            label_matrix_with_pilot,
            InsertedPilotDataMatrix,
            New_Pilot_Loc,
            l_tap) #[DATA_INTERPOLATED, L_TAP, UE, BS]
    
    LSMLPmse_total = mse_calc(interpolateChannel_LTAP, Realchennel_LTAP)
    LSEstimatedData = ofdm_obj.Equalizer( interpolatevalFdomain, SigConvChannelNoiseRemoveCPFFT, DataLocation_Tdomain, SNRdb)
    LSMLPBER = ofdm_obj.ErrorBits(LSEstimatedData, OFDMMappedData, repetition)

    return LSFullMLPmse_total, LSFullMLPBER, LSMLPmse_total, LSMLPBER

def TORCH_DATA_AIDED(LSEstimatedData,
                     InsertedPilotDataMatrix,
                     DataLocation_Tdomain,
                     pilotLocation_Tdomain,
                     ofdm_obj,
                     get_data,
                     train_CVAE,
                     train_BaselineNet,
                     cvae_classifier,
                     SigConvChannelNoiseRemoveCPFFT,
                     OFDM_Symbol_total_range,
                     OFDMMappedData,
                     Realchennel_LTAP,
                     l_tap,
                     SNRdb,
                     modscheme,
                     args,
                     results_file,
                     NoisePower):
    
    LSEstimatedDataCopy = np.copy(LSEstimatedData)
    InsertedPilotEstimatedData = np.copy(InsertedPilotDataMatrix)
    for index, i in enumerate(DataLocation_Tdomain):
        InsertedPilotEstimatedData[:,:,i] = LSEstimatedData[:,:, index]
        
    
    
    _, interpolatevalFdomain, _ , _ = ofdm_obj.LsEstimation(SigConvChannelNoiseRemoveCPFFT,
            InsertedPilotEstimatedData,
            pilotLocation_Tdomain ,
            l_tap,
            NoisePower) #[DATA_INTERPOLATED, L_TAP, UE, BS]
    ## Can use LMMSE as well. Reverse _ for this.
    

    
    features1, labels1, classes, num_cls =  ofdm_obj.labeling_reconstruction(
                                                                            LSEstimatedDataCopy,
                                                                            OFDMMappedData,
                                                                            SigConvChannelNoiseRemoveCPFFT,
                                                                            interpolatevalFdomain,
                                                                            DataLocation_Tdomain)
    
    _, _, Subcarrirers, _ = interpolatevalFdomain.shape
    ## features1.shape =(batch, UE * CLS (modulation length) + UE * CLS (modulation length) + 2*BS + 2*BS*UE)
    _, in_shape = features1.shape
    
    ## labels1.shape =(batch, UE, CLS (modulation length))
    _, out_heads, out_len = labels1.shape
    out_len = num_cls
    # Dataset
    datasets, dataloaders, dataset_sizes = {}, {}, {}
    ['train', 'val']
    
    datasets['train'], dataloaders['train'], dataset_sizes['train'] = get_data(
        features1[:int(0.8*len(features1))], labels1[:int(0.8*len(labels1))], 'train', batch_size=20
    )
    datasets['val'], dataloaders['val'], dataset_sizes['val'] = get_data(
        features1[int(0.8*len(features1)):], labels1[int(0.8*len(labels1)):], 'val', batch_size=20
    )
    
    datasets['cvae_cls_train'], dataloaders['cvae_cls_train'], dataset_sizes['cvae_cls_train'] = get_data(
        features1[:int(0.8*len(features1))], classes[:int(0.8*len(classes))], 'cvae_cls_train', batch_size=20
    )
    
    datasets['cvae_cls_val'], dataloaders['cvae_cls_val'], dataset_sizes['cvae_cls_val'] = get_data(
        features1[int(0.8*len(features1)):], classes[int(0.8*len(classes)):], 'cvae_cls_val', batch_size=20
    )
    
    datasets['cvae_cls_test'], dataloaders['cvae_cls_test'], dataset_sizes['cvae_cls_test'] = get_data(
        features1, classes, 'cvae_cls_test', batch_size=len(classes)
    )


    # Train baseline
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(results_file, 'a') as f:
        f.write(" *************** BaseLine Training \n") 
    baseline_net = train_BaselineNet(
        in_shape,
        out_heads,
        out_len,
        results_file,
        device=device,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stop_patience=args.early_stop_patience,
        model_path='baseline_net.pth')
    
    
    # Train CVAE
    with open(results_file, 'a') as f:
        f.write(" *************** CVAE Training \n") 
    cvae_net = train_CVAE(
        in_shape,
        out_heads,
        out_len,
        args.zdim,
        results_file,
        device=device,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stop_patience=args.early_stop_patience,
        model_path='baseline_net.pth',
        pre_trained_baseline_net=baseline_net
    )
    
    # Train Correctness Classifier
    with open(results_file, 'a') as f:
        f.write(" *************** Classifier Training \n") 
    cvae_cls = cvae_classifier(args.zdim,
                    dataloaders,
                    dataset_sizes,
                    cvae_net,
                    results_file,
                    num_epochs = args.num_epochs,
                    device = device,
                    learning_rate = args.learning_rate, 
                    early_stop_patience = args.early_stop_patience ,
                    model_path ='cvae_classifier.pth')
    
    # Test Correctness Classifier on Real Data
    cvae_cls.eval()
    cvae_net.baseline_net.eval()
    cvae_net.prior_net.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloaders['cvae_cls_test']):
            xs = torch.as_tensor(batch['input'], dtype=torch.float32)
            xs = xs.to(device)
            y_hat = torch.stack(cvae_net.baseline_net(xs), dim = 1)
            loc, scale = cvae_net.prior_net(xs + 1e-8, y_hat)
            # z_samples = dist.Normal(loc, scale).sample()
            z_samples = loc
            preds = cvae_cls(z_samples)
            correctness_cls = preds.view(Subcarrirers, len(DataLocation_Tdomain))
    
    labels_matrix = np.round( correctness_cls.cpu().numpy())
    _, New_Pilot_Loc, _, label_matrix_with_pilot = ofdm_obj.DataDetectionCVAE( 
                        labels_matrix,
                        LSEstimatedData,
                        interpolatevalFdomain,
                        DataLocation_Tdomain,
                        l_tap)
    
    
    # TODO: Replace LS with LMMSE
    interpolateChannel_LTAP_cvae_cls, interpolatevalFdomain_cvae_cls = ofdm_obj.LsEstimation_ForDetectedData(
            SigConvChannelNoiseRemoveCPFFT,
            InsertedPilotDataMatrix,
            OFDM_Symbol_total_range,
            l_tap) #[DATA_INTERPOLATED, L_TAP, UE, BS]
    cvae_cls_mse_total = mse_calc(interpolateChannel_LTAP_cvae_cls, Realchennel_LTAP)
    
    EstimatedData_eq_total = ofdm_obj.Equalizer( interpolatevalFdomain_cvae_cls, SigConvChannelNoiseRemoveCPFFT, DataLocation_Tdomain, SNRdb)
    cvae_cls_ber_total = ofdm_obj.ErrorBits(EstimatedData_eq_total, OFDMMappedData, args.repetition)
    
    

    
    
    interpolateChannel_LTAP_cvaep_cls, interpolatevalFdomain_cvaep_cls = ofdm_obj.LsEstimation_PartiallyDetectedData(
            SigConvChannelNoiseRemoveCPFFT,
            label_matrix_with_pilot,
            InsertedPilotDataMatrix,
            New_Pilot_Loc,
            l_tap) #[DATA_INTERPOLATED, L_TAP, UE, BS]
    
    cvae_cls_mse_partial = mse_calc(interpolateChannel_LTAP_cvaep_cls, Realchennel_LTAP)
    EstimatedData_eq_partial = ofdm_obj.Equalizer( interpolatevalFdomain_cvaep_cls, SigConvChannelNoiseRemoveCPFFT, DataLocation_Tdomain, SNRdb)
    cvae_cls_ber_partial = ofdm_obj.ErrorBits(EstimatedData_eq_partial, OFDMMappedData, args.repetition)

    return cvae_cls_mse_total, cvae_cls_ber_total, cvae_cls_mse_partial, cvae_cls_ber_partial


    
    
    
    

                    
