from scipy.signal import TransferFunction
import numpy as np

from numpy_utils import converters as cnvs
from .object import Object
from .TF_object import TF_object


class Integral_Object(Object):
    def __init__(self, time_const, gain, sampling_time=1):
        self.set_time_const(time_const)
        self.set_gain(gain)
        if not self.__validate_shapes():
            raise ValueError('The object has to possess same amounts of gains and time constants sets')
        
        self.build_tf_matrix()
        n_states = self.__get_total_states_count()
        super().__init__(sampling_time, self.__tf.shape[1], self.__tf.shape[0], n_states)
        self.__obj = TF_object(sampling_time=sampling_time, tf_matrix=self.__tf)
    
    def __validate_shapes(self):
        return self.__time_const.shape[:2] == self.__gain.shape
        
    def set_time_const(self, time_const):
        self.__time_const = cnvs.to_array_converter().convert_any_type(time_const)

    def set_gain(self, gain):
        self.__gain = cnvs.to_array_converter().convert_any_type(gain)

    def build_tf_matrix(self):
        tf = []
        for i in range(0, len(self.__gain)):
            row = []
            for j in range(0, len(self.__gain[i])):
                row.append(self.__build_siso_tf(self.__time_const[i][j], self.__gain[i][j]))
            tf.append(row)
        self.__tf = cnvs.to_array_converter().convert_any_type(tf)
                
    def __build_siso_tf(self, time_const_list, gain_val):
        # independant input
        if gain_val == 0:
            res = 0
        # implicit proportional object - no time constants and assuming demoniator last term is 1
        elif all(tc == 0 for tc in time_const_list):
            res = gain_val
        # proper object based on: G(s) = K / (Tns^n + ... + Ts + 0) - 0 = astatic bevahiour
        else:
            res = TransferFunction([gain_val], np.append(time_const_list, 0))
        return res

    def __get_total_states_count(self):
        n_output_inputs = self.__tf.shape
        n_states = 0
        
        for output in range(0, n_output_inputs[0]):
            for input in range(0, n_output_inputs[1]):
                current_tf = self.__tf[output, input]
                if type(current_tf) in (int, float, np.int64, np.float64):
                    n_states += 1 
                else:
                    n_states += len(self.__tf[output, input].den) - 1
        return n_states
        
    def summary(self, messages=None):
        if messages==None:
            messages = []    
        messages.append(f'This is an integral object.')
        messages.append(f'Gain matrix: \n{self.__gain}')
        messages.append(f'Time constants matrix: \n{self.__time_const}')
        super().summary(messages)

    def simulate_step(self, u):
        ty, y_vec, x_vec = self.__obj.simulate_step(u)
        self._update_history(u, y_vec, x_vec, ty)
        return ty, y_vec, x_vec