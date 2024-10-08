from scipy.signal import TransferFunction
import numpy as np

from numpy_utils import converters as cnvs
from .TF_specific_object import TF_Specific_Object
from .TF_object import TF_object


class Inertial_Object(TF_Specific_Object):
    def __init__(self, time_const, gain, sampling_time=1):
        super().__init__(time_const, gain, sampling_time)

    # Overloading a parent's abstract method
    def _validate_shapes(self):
        return self._time_const.shape == self._gain.shape

    # Overloading a parent's abstract method        
    def _build_siso_tf(self, time_const_val, gain_val):
        # independant input
        if gain_val == 0:
            res = 0
        # implicit proportional object
        elif time_const_val == 0:
            res = gain_val
        # proper object based on: G(s) = K / (Ts + 1)
        else:
            res = TransferFunction([gain_val], [time_const_val, 1])
        return res

    # Extended as intented
    def summary(self, messages=None):
        if messages==None:
            messages = []    
        messages.append(f'This is an inertial object.')
        messages.append(f'Gain matrix: \n{self._gain}')
        messages.append(f'Time constants matrix: \n{self._time_const}')
        super().summary(messages)