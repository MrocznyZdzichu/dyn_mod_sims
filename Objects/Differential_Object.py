from scipy.signal import TransferFunction
import numpy as np

from numpy_utils import converters as cnvs
from .TF_specific_object import TF_Specific_Object
from .TF_object import TF_object


class Differential_Object(TF_Specific_Object):
    def __init__(self, time_const, sampling_time=1):
        gain = self.__build_mock_gain_matrix(time_const)
        super().__init__(time_const, gain, sampling_time)

    def __build_mock_gain_matrix(self, time_const):
        return np.ones(cnvs.to_array_converter().convert_any_type(time_const).shape)
        
    # Overloading a parent's abstract method
    def _validate_shapes(self):
        return self._time_const.shape == self._gain.shape

    # Overloading a parent's abstract method        
    def _build_siso_tf(self, time_const_val, gain_val):
        # Gain is hardcoded 1
        # If time_const == 0 - indenpendant input
        if time_const_val == 0:
            res = 0
        # proper object based on: G(s) = s / (Ts + 1)
        else:
            res = TransferFunction([gain_val, 0], [time_const_val, 1])
        return res

    # Extended as intented
    def summary(self, messages=None):
        if messages==None:
            messages = []    
        messages.append(f'This is a differential object.')
        messages.append(f'Gain-sh time constants matrix: \n{self._gain}')
        messages.append(f'Time constants matrix: \n{self._time_const}')
        super().summary(messages)