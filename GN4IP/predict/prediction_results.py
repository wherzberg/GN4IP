# Create a class for prediction results

import numpy as np
import scipy.io

class PredictionResults(object):
    '''
    This object makes storing and accessing predictions from a model easier.
    '''
    
    def __init__(self, input, input_cnn=None):
        '''
        Results should be initialized with the input to the network in graph
        format and, if it's a CNN, the pixel grid input can also be stored as
        an input. Then the output of the networks are stored as predictions.
        If the networks take two features as input (the second is an update),
        then those inputs can be stored as the updates and updates_cnn 
        attributes.
        '''
        self.input = input
        self.input_cnn = input_cnn
        self.predictions = []
        self.updates = []
        self.predictions_cnn = []
        self.updates_cnn = []
        self.time_pr = []
    
    def appendPredictions(self, results_pr):
        '''
        Append a separate PredictionResults object to this one.
        '''
        self.predictions.extend(results_pr.predictions)
        self.updates.extend(results_pr.updates)
        self.predictions_cnn.extend(results_pr.predictions_cnn)
        self.updates_cnn.extend(results_pr.updates_cnn)
        self.time_pr.extend(results_pr.time_pr)
    
    def save(self, name):
        '''
        Save the prediction results as a .mat file. If self.updates is empty it
        can be skipped. If the self.input_cnn is None, then the CNN stuff can 
        also be skipped.
        '''
        # Build the base dictionary
        save_dict = {
            "input_data"  : self.input,
            "predictions" : np.array(self.predictions),
            "prediction_times" : np.array(self.time_pr),
        }
        
        # Add updates or CNN stuff?
        if len(self.updates) > 0:
            save_dict["updates"] = np.array(self.updates)
        if self.input_cnn is not None:
            save_dict["input_data_cnn"] = self.input_cnn
            save_dict["predictions_cnn"] = np.array(self.predictions_cnn)
            if len(self.updates) > 0:
                save_dict["updates_cnn"] = np.array(self.updates_cnn)
        
        # Save 
        scipy.io.savemat(name, save_dict)
        