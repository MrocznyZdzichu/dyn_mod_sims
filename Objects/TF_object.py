from .object import Object
import numpy as np
from numpy_utils import converters
from scipy.signal import TransferFunction
from scipy.signal import lsim


class TF_object(Object):
    def __init__(self, sampling_time, tf):
        self.__tf = tf
        n_states = len(tf.den) - 1
        super().__init__(sampling_time, 1, 1, n_states)

    def summary(self, messages=None):
        if messages==None:
            messages = []    
        messages.append(f'Transfer function of the object: {self.__tf}')
        super().summary(messages)
        
    def simulate_step(self, u):        
        u_hist = self.get_input_hist()
        t_hist = self.get_timestamps()
        x_hist = self.get_state_hist()

        u_prev = u_hist[-1]
        t_prev = t_hist[-1]
        x_prev = x_hist[-1]

        ut = u
        tt = t_prev + self._sampling_time
        x_prev = x_prev.reshape(self._n_states,)
        
        ty, yy, xy = lsim(self.__tf, [u_prev, ut], [0, self._sampling_time], x_prev)
        self._update_history(ut, yy[-1], xy[-1], tt)
        
        return ty, yy, xy

    def reset_history(self):
        super().reset_history()