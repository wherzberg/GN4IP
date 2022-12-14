# Create a class for prediction parameters

import GN4IP

class PredictionParameters(object):
    '''
    This object makes storing and accessing prediction parameters easier.
    '''
    
    def __init__(self, device="cpu", scale=1, updateFunction=None, fwd_data_file=""):
        '''
        Initialize the parameters object
        '''
        self.device = device
        self.scale = scale
        self.updateFunction = updateFunction
        self.fwd_data_file = fwd_data_file
    
    def printHeader(self):        
        GN4IP.utils.printLine()
        print("#  Training a model")
        GN4IP.utils.printLine()
        print("# Predicting On:", self.device)
        print("#         Scale:", self.scale)
