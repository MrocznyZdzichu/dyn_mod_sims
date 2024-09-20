import numpy as np
from numpy_utils import converters
from .object import Object


class Proportional_object(Object):
    def __init__(self, gain, sampling_time=0.1):
        self.set_gain(gain)
        
        self.__supported_input_types = [int, float, list, tuple, np.ndarray]
        self.__object_type = "Proportional object"
        self.__describe_parameters = {
            "gain" : self.__gain
        }
        
        super().__init__(sampling_time, self.get_gain().shape[1], self.get_gain().shape[0], self.get_gain().shape[0])
        
    def get_gain(self):
        return self.__gain

    def set_gain(self, gain):
        if not self.__validate_gain(gain):
            try: 
                gain = self.__convert_to_gain_matrix(gain)
            except:
                raise TypeError('Gain has to be a matrix or convertible to a matrix')
                
        self.__gain = gain

    def __validate_gain(self, gain):
        return type(gain) == np.ndarray

    def __convert_to_gain_matrix(self, gain):
        conv = converters.to_array_converter()
        try:
            gain_matrix = conv.convert_any_type(gain)
        except:
            raise TypeError('Gain has to be a matrix or convertible to a matrix')
        return gain_matrix
        
    def summary(self):
        messages = []
        messages.append(self.__object_type)
        messages.append('Object parameters:')
        
        for key in self.__describe_parameters.keys():
            messages.append(f'{key} with value of:\n {self.__describe_parameters[key]}')
        messages.append('')
        
        messages.append(f'Dimensions of the object: \n{self.__gain.shape[0]} outputs and {self.__gain.shape[1]} inputs')
        messages.append('')
        
        super().summary(messages)
        
    def simulate_step(self, u):
        t_hist = self.get_timestamps()
        t_prev = t_hist[-1]
        
        self.__validate_input(u)
        yt = np.dot(self.__gain, u)
        
        super()._update_history(u, yt, yt, t_prev+self._sampling_time)
        
        return t_prev+self._sampling_time, yt, yt

    def __validate_input(self, u):
        if type(u) not in self.__supported_input_types:
            return False

    def show_history(self):
        super().show_history()