import numpy as np
import scipy
from scipy.signal import TransferFunction

from .object import Object
from .TF_object_SISO import TF_object_SISO
from .proportional_object import Proportional_object
from numpy_utils import converters


class TF_object(Object):
    def __init__(self, sampling_time, tf_matrix):
        if not self.__validate_tf_matrix(tf_matrix):
            raise ValueError('tf_matrix has to be a 2D numpy array of TransferFunctions.')
            
        self.__tf = tf_matrix        
        n_states = self.__get_total_states_count()
        super().__init__(sampling_time, self.__tf.shape[1], self.__tf.shape[0], n_states)
        self.__subobjects = self.__create_SISO_subobjects(sampling_time)
        
    def __validate_tf_matrix(self, tf_matrix):
        if type(tf_matrix) != np.ndarray:
            return False
        elif tf_matrix.ndim != 2:
            return False

        res = False
        for row in tf_matrix:
            for el in row:
                res += type(el) not in (scipy.signal._ltisys.TransferFunctionContinuous, int, float, np.float64, np.int64)
        return res == 0

    def __create_SISO_subobjects(self, sampling_time):
        subobjects = []
        
        for output in range(0, self._n_outputs):
            row = []
            for input in range(0, self._n_inputs):
                current_tf = self.__tf[output, input]
                if type(current_tf) == scipy.signal._ltisys.TransferFunctionContinuous:
                    row.append(TF_object_SISO(
                        sampling_time=sampling_time
                        ,tf=current_tf
                    ))
                else:
                    row.append(Proportional_object(
                        gain=current_tf, 
                        sampling_time=sampling_time
                    ))
            subobjects.append(row)

        return subobjects

    def __get_total_states_count(self):
        n_output_inputs = self.__tf.shape
        n_states = 0
        
        for output in range(0, n_output_inputs[0]):
            for input in range(0, n_output_inputs[1]):
                current_tf = self.__tf[output, input]
                if type(current_tf) in (int, float, np.float64, np.int64):
                    n_states += 1 
                else:
                    n_states += len(self.__tf[output, input].den) - 1
        return n_states
        
    def summary(self, messages=None):
        if messages==None:
            messages = []    
        messages.append(f'Transfer function of the object: {self.__tf}')
        super().summary(messages)
        
    def simulate_step(self, u):
        conv = converters.to_vector_converter()
        t_hist = self.get_timestamps()
        t_prev = t_hist[-1]

        y_vec = np.zeros((self._n_outputs, 1))
        x_vec = []
        
        for output in range(0, self._n_outputs):
            for input in range(0, self._n_inputs):
                subobject = self.__subobjects[output][input]
                ut = conv.convert_any_type(u[input])
                ty, yy, xy = subobject.simulate_step(ut)
                y_vec[output, 0] += yy[-1]
                if hasattr(xy[-1], '__iter__'):
                    for state in xy[-1]:
                        x_vec.append(state)
                else:
                    x_vec.append(xy[-1])

        ty = t_prev + self._sampling_time
        self._update_history(u, y_vec, x_vec, ty)
        
        return ty, y_vec, x_vec

    def reset_history(self):
        super().reset_history()