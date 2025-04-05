# =============================================================================
# Author      : Hamidreza Hashempoor
# Email       : hamidreza.hashemp@snu.ac.kr
# Created     : 2025/04/05
# Description : Implementation of "Deep Learning Based Data-Assisted Channel Estimation and Detection" paper.
# =============================================================================

from pathlib import Path
import argparse
import yaml
from ofdm.OFDM import OFDM   
from utils.call_modules import CONVENTIONAL_CONFIG, CONV_DATA_AIDED_CONFIG, CHANNEL_TRACKING, TF_DATA_AIDED, TORCH_DATA_AIDED
from Dataloader.dataloader import get_data
from cvae.cvae import train as train_CVAE
from cvae.baseline import train_BaselineNet
from cvae.cvae import cvae_classifier
from utils.path_handling import increment_path
from plot.plot import Plotber, Plotmse

              


        
def main(args):
    BER = []
    MSE =[]
    BER_Dict = {}
    MSE_Dict = {}

    args.save_dir = Path(increment_path(Path(args.project) / args.name))  # increment run
    
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    wdir = args.save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = args.save_dir / 'results.txt'

    # Save run settings
    with open(args.save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)

    
    for T in args.timesamplelist:
        
        ofdm_obj = OFDM(args.subcarriers, args.cp, args.ltap, args.snrlist[-1], args.ues, args.bss , T, args.ofdmsymbols) 
    
        datapath = args.datadir
        _, Realchennel_LTAP, _ = ofdm_obj.ChannelLoading(datapath,
                        args.subcarriers, args.ltap, args.ofdmsymbols)
        Realchennel_LTAP_reorder = Realchennel_LTAP.transpose(3,2,1,0)
        
        #SNRlist = [-15, -10, -5, 0, 5, 10, 15]
        # SNRlist = [  -15, -10, -5, 0, 5,10]
        LSmselist = []
        LSBERlist = []
        
        Lmmsemselist = []
        LmmseBERlist = []
        
        ConvAidedmselist = []
        ConvAidedBERlist = []
        
        # MLPFullmselist = []
        # MLPFullBERlist = []
        
        # MLPmselist = []
        # MLPBERlist = []
        
        CVAETotalmselist = []
        CVAETotalBERlist = []
        
        CVAEPartialmselist = []
        CVAEPartialBERlist = []
    
        for SNRdb in args.snrlist:
            print("TimeSpaces in OFDM pilots: {}, SNRdb: {} \n".format(T, SNRdb))

            with open(results_file, 'a') as f:
                f.write("**************************************** TimeSpaces in OFDM pilots: {}, SNRdb: {} ***************************************\n".format(T, SNRdb)) 
            r = ofdm_obj.PilotGenGoldSeq(args.subcarriers, args.ues) 
    
            pilotLocation_Tdomain, DataLocation_Tdomain = ofdm_obj.PilotDataloc(args.ofdmsymbols, T)
             
            OFDMMappedTotal = ofdm_obj.DataGenAll( args.modscheme , args.ofdmsymbols, args.ues, args.subcarriers, args.repetition)
            OFDMMappedData = OFDMMappedTotal[:,:,DataLocation_Tdomain]
            
            InsertedPilotDataMatrix, OFDM_Symbol_total_range = ofdm_obj.PilotInsertionAll(args.ues, args.subcarriers, OFDMMappedData, r,
                  args.ofdmsymbols ,T)  #(self, args.ues, args.subcarriers, OFDMMappedData, PilotSeqs, OFDM_Symbol_total, T): 
                
            CPIFFTInsertedPilotDataMatrix = ofdm_obj.AddingCPIFFT(InsertedPilotDataMatrix, args.cp) 
            
            _, SigConvChannelNoiseRemoveCPFFT, NoisePower = ofdm_obj.Transmission(Realchennel_LTAP_reorder,
                                            CPIFFTInsertedPilotDataMatrix, args.ltap, SNRdb, args.cp)# [BS, NFFT, OFDM_Symbol_total]
            
            ## CONVENTIONAL CONFIG
            LSEstimatedData, LSmse_total, LmmseEstimatedData, LMMSEmse_total, LSBER, LmmseBER = CONVENTIONAL_CONFIG(ofdm_obj,
                                SigConvChannelNoiseRemoveCPFFT,
                                InsertedPilotDataMatrix,
                                pilotLocation_Tdomain,
                                DataLocation_Tdomain,
                                args.ltap,
                                OFDMMappedData,
                                Realchennel_LTAP,
                                args.modscheme,
                                T,
                                SNRdb,
                                args.repetition,
                                NoisePower)
            LSmselist.append(LSmse_total)
            LSBERlist.append(LSBER)
            Lmmsemselist.append(LMMSEmse_total)
            LmmseBERlist.append(LmmseBER)
            
            #conventional data aided config
            CONVmse_total, CONVBER = CONV_DATA_AIDED_CONFIG(LSEstimatedData,
                                                            InsertedPilotDataMatrix,
                                                            ofdm_obj,
                                                            SigConvChannelNoiseRemoveCPFFT,
                                                            OFDM_Symbol_total_range,
                                                            Realchennel_LTAP,
                                                            OFDMMappedData,
                                                            SNRdb,
                                                            T,
                                                            DataLocation_Tdomain ,
                                                            args.ltap,
                                                            args.repetition,
                                                            NoisePower)
            ConvAidedmselist.append(CONVmse_total)
            ConvAidedBERlist.append(CONVBER)

            ##channel tracking
            # interpolateChannel_LTAP_list, interpolatevalFdomain_list = CHANNEL_TRACKING(OFDMMappedTotal, 
            #     ofdm_obj,
            #     Realchennel_LTAP_reorder,
            #     OFDM_Symbol_total_range,
            #     args.ues,
            #     args.subcarriers,
            #     r,
            #     args.cp,
            #     SNRdb,
            #     args.ltap)

            # #data aided config (torch)
            cvae_cls_mse_total, cvae_cls_ber_total, cvae_cls_mse_partial, cvae_cls_ber_partial = TORCH_DATA_AIDED(
                     LSEstimatedData, ## Can be replace with LMMSE estimated data
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
                     args.ltap,
                     SNRdb,
                     args.modscheme,
                     args,
                     results_file,
                     NoisePower)
            CVAETotalmselist.append(cvae_cls_mse_total)
            CVAETotalBERlist.append(cvae_cls_ber_total)
            
            CVAEPartialmselist.append(cvae_cls_mse_partial)
            CVAEPartialBERlist.append(cvae_cls_ber_partial)
    
        
        MSE_Dict[T] = {"LSmselist": LSmselist, "ConvAidedmselist": ConvAidedmselist, "CVAETotalmselist":CVAETotalmselist,
                       "CVAEPartialmselist": CVAEPartialmselist}
        BER_Dict[T] = {"LSBERlist": LSBERlist, "ConvAidedBERlist": ConvAidedBERlist, "CVAETotalBERlist":CVAETotalBERlist,
                       "CVAEPartialBERlist": CVAEPartialBERlist}
    
    return BER, MSE, BER_Dict, MSE_Dict

def load_arguments():
    parser = argparse.ArgumentParser(description='OFDM Arguments')
    #TODO: Add coding scheme for reproducing [e.g. convolution]
    parser.add_argument("--subcarriers", default= 256, help='subcariers should be 128-256')
    parser.add_argument("--cp", default = 30, help='cp length')
    parser.add_argument("--ltap", default = 12, help='num of taps')
    parser.add_argument("--snrlist", default = [  -15, -10, -5, 0, 5,10], help='snr list in experiments')
    parser.add_argument("--ues", default = 2, help='num of ues')
    parser.add_argument("--bss", default = 4, help='num of bss')
    parser.add_argument("--timesamplelist", default = [2,4, 5,10], help='num of bss')
    parser.add_argument("--ofdmsymbols", default = 21, type = int, help='num of ofdm symbols')
    parser.add_argument("--datadir", default = './channel.mat', help='num of ofdm symbols')
    parser.add_argument("--modscheme", default = 'QAM', help='num of ofdm symbols')
    parser.add_argument("--repetition", default = 2, help='repetition num')
    parser.add_argument('-n', '--num-epochs', default=1, type=int,
                        help='number of training epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='latent variable dimension')
    parser.add_argument('-esp', '--early-stop-patience', default=100, type=int,
                        help='early stop patience')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-4, type=float,
                        help='learning rate')
    parser.add_argument('--hyp', default= 'hyp/hyp.scratch.yaml', type=str, help='save dir path')
    parser.add_argument('--save-dir', default= '', type=str, help='save dir path')
    parser.add_argument('--name', default= 'exp', type=str, help='name of project')
    parser.add_argument('--project', default= 'runs', type=str, help='name of project')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = load_arguments()
    BER, MSE, BER_Dict, MSE_Dict = main(args)
    
    BER_ColorDict = {"LSBERlist": "b", "ConvAidedBERlist": "r", "CVAETotalBERlist":"g",
                       "CVAEPartialBERlist": "y"}
    MSE_ColorDict = {"LSmselist": "b", "ConvAidedmselist": "r", "CVAETotalmselist":"g",
                       "CVAEPartialmselist": "y"}
    Plotber(args, BER_Dict, BER_ColorDict)
    Plotmse(args, MSE_Dict, MSE_ColorDict)

    