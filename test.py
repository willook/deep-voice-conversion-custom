from data_load import get_mfccs_and_phones

import numpy as np
from hparam import hparam as hp
if __name__ =='__main__':
    
    hp.set_hparam_yaml("TINIT2")
    mfccs, phns = get_mfccs_and_phones("/home/cocoonmola/datasets/TIMIT2/TRAIN/DR3/FCMG0/SA1.WAV")
    print(mfccs.shape)
    print(phns.shape)
    np.save("./mfccs",mfccs)
    
