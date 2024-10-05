import numpy as np

from numpy_utils import converters as cnvs

from .object import Object
from .TF_object import TF_object


class TF_Specific_Object(Object):
    def __init__(self, time_const, gain, sampling_time):
        self.set_time_const(time_const)
        self.set_gain(gain)
        if not self._validate_shapes():
            raise ValueError('The object has to possess same amounts of gains and time constants')

        self._build_tf_matrix()
        n_states = self._get_total_states_count()
        
        self._obj = None
        super().__init__(sampling_time, self._tf.shape[1], self._tf.shape[0], n_states)
        self._obj = TF_object(sampling_time=sampling_time, tf_matrix=self._tf)
        
    def set_time_const(self, time_const):
        self._time_const = cnvs.to_array_converter().convert_any_type(time_const)

    def set_gain(self, gain):
        self._gain = cnvs.to_array_converter().convert_any_type(gain)

    # To be overwritten in children classes
    def _validate_shapes(self):
        pass

    def _build_tf_matrix(self):
        tf = []
        for i in range(0, len(self._gain)):
            row = []
            for j in range(0, len(self._gain[i])):
                row.append(self._build_siso_tf(self._time_const[i][j], self._gain[i][j]))
            tf.append(row)
        self._tf = cnvs.to_array_converter().convert_any_type(tf)

        
    # To be overwritten in children classes
    def _build_siso_tf(self, time_const_val, gain_val):
        pass

    def _get_total_states_count(self):
        n_output_inputs = self._tf.shape
        n_states = 0
        
        for output in range(0, n_output_inputs[0]):
            for input in range(0, n_output_inputs[1]):
                current_tf = self._tf[output, input]
                if type(current_tf) in (int, float, np.int64, np.float64):
                    n_states += 1 
                else:
                    n_states += len(self._tf[output, input].den) - 1
        return n_states
        
    # To be extended in children classes
    def summary(self, messages=None):
        super().summary(messages)

    
    def simulate_step(self, u):
        ty, y_vec, x_vec = self._obj.simulate_step(u)
        self._update_history(u, y_vec, x_vec, ty)
        return ty, y_vec, x_vec

    def reset_history(self):
        if self._obj != None:
            self._obj.reset_history()
        super().reset_history()