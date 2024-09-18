import numpy as np
from numpy_utils import converters


class Object:
    def __init__(self, sampling_time, n_inputs, n_outputs, states_count):
        self._sampling_time = sampling_time
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._n_states = states_count
        self.reset_history()

    def summary(self, messages):
        messages.append(f'Sampling time: {self._sampling_time}')
        for msg in messages:
            print(msg)

    def get_param(self, param_name):
        return eval(f"self._{self.__class__.__name__}__{param_name}")

    def simulate_step(self, u):
        pass

    def reset_history(self):
        self._input_hist = [np.zeros((self._n_inputs, 1))]
        self._output_hist = [np.zeros((self._n_outputs, 1))]
        self._state_hist = [np.zeros((self._n_states, 1))]
        self._timestamps = [0]

    def show_history(self):
        print('History of inputs:')
        print(self.get_input_hist())
        print('')
        print('History of outputs:')
        print(self.get_output_hist())
        print('History of states:')
        print(self.get_state_hist())
        
    def _update_history(self, input_t, output_t, state_t, ts):
        vecconv = converters.to_vector_converter()
        self._log_input(vecconv.convert_any_type(input_t))
        self._log_output(vecconv.convert_any_type(output_t))
        self._log_state(vecconv.convert_any_type(state_t))
        self._log_timestamp(ts)

    def get_input_hist(self):
        return self._input_hist

    def get_output_hist(self):
        return self._output_hist

    def get_state_hist(self):
        return self._state_hist

    def get_timestamps(self):
        return self._timestamps

    def _log_timestamp(self, ts):
        self._timestamps.append(ts)

    def _log_input(self, u):
        self._input_hist.append(u)

    def _log_output(self, y):
        self._output_hist.append(y)

    def _log_state(self, x):
        self._state_hist.append(x)