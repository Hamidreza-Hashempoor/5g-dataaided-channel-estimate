import matplotlib.pyplot as plt

def PlotBER(SNRlist, LSBERlist, CVAETotalBERlist, CVAEPartialBERlist):
    plt.plot(SNRlist, LSBERlist, "-b", label="LS BER")
    plt.plot(SNRlist, LSBERlist, "ob")
    plt.plot(SNRlist, CVAETotalBERlist, "-r", label="CVAE Full BER")
    plt.plot(SNRlist, CVAETotalBERlist, "or")
    plt.plot(SNRlist, CVAEPartialBERlist, "-g", label="CVAE Partial BER")
    plt.plot(SNRlist, CVAEPartialBERlist, "og")
    plt.grid(True, which='both', axis='y', ls="-")
    plt.grid(True, which='major', axis='x', ls="-")
    plt.yscale('log')
    plt.xlabel("SNR")
    plt.ylabel("BER")
    plt.legend(loc="upper right")
    plt.show()
def PlotMSE(SNRlist, LSmselist, CVAETotalmselist, CVAEPartialmselist):
    plt.plot(SNRlist, LSmselist, "-b", label="LS MSE")
    plt.plot(SNRlist, LSmselist, "ob")
    plt.plot(SNRlist, CVAETotalmselist, "-r", label="CVAE Full MSE")
    plt.plot(SNRlist, CVAETotalmselist, "or")
    plt.plot(SNRlist, CVAEPartialmselist, "-g", label="CVAE Partial MSE")
    plt.plot(SNRlist, CVAEPartialmselist, "og")
    plt.grid(True, which='both', axis='y', ls="-")
    plt.grid(True, which='major', axis='x', ls="-")
    plt.yscale('log')
    plt.xlabel("SNR")
    plt.ylabel("MSE")
    plt.legend(loc="upper right")
    plt.show()
    
def PlotBER2(SNRlist, LSBERlist, ConvAidedBERlist):
    plt.plot(SNRlist, LSBERlist, "-b", label="LS BER")
    plt.plot(SNRlist, LSBERlist, "ob")
    
    plt.plot(SNRlist, ConvAidedBERlist, "-r", label="conv data aided BER")
    plt.plot(SNRlist, ConvAidedBERlist, "or")

    plt.xlabel("SNR")
    plt.ylabel("BER")
    plt.legend(loc="upper right")
    plt.show()
def PlotMSE2(SNRlist, LSmselist, ConvAidedmselist):
    plt.plot(SNRlist, LSmselist, "-b", label="LS MSE")
    plt.plot(SNRlist, LSmselist, "ob")
    
    plt.plot(SNRlist, ConvAidedmselist, "-r", label="conv data aided MSE")
    plt.plot(SNRlist, ConvAidedmselist, "or")

    plt.xlabel("SNR")
    plt.ylabel("MSE")
    plt.legend(loc="upper right")
    plt.show()

def Plotber(args, BER_Dict, BER_ColorDict):
    for T, inner_dict in BER_Dict.items():
        fig = plt.figure()
        for module, value in inner_dict.items():
            if len(value) > 0 :
                plt.plot(args.snrlist, value, "-{}".format(BER_ColorDict[module]), label="{} BER".format(module))
                plt.plot(args.snrlist, value, "o{}".format(BER_ColorDict[module]))
        plt.title("T is equal to {}".format(T))
        plt.xlabel("SNR")
        plt.ylabel("BER")
        plt.legend(loc="upper right")
        fig.savefig( args.save_dir / "berT{}.jpg".format(T))
        # plt.show()       

def Plotmse(args, MSE_Dict, MSE_ColorDict):
    for T, inner_dict in MSE_Dict.items():
        fig = plt.figure()
        for module, value in inner_dict.items():
            if len(value) > 0 :
                plt.plot(args.snrlist, value, "-{}".format(MSE_ColorDict[module]), label="{} MSE".format(module))
                plt.plot(args.snrlist, value, "o{}".format(MSE_ColorDict[module]))
        plt.title("T is equal to {}".format(T))
        plt.xlabel("SNR")
        plt.ylabel("MSE")
        plt.legend(loc="upper right")
        fig.savefig(args.save_dir / "mseT{}.jpg".format(T))
        # plt.show() 