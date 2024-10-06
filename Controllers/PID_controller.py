import numpy as np
from simple_pid import PID


class PID_controller(PID):
    def __init__(self, Kp, Ki, Kd, setpoint=0, sample_time=1):
        super().__init__(Kp, Ki, Kd, setpoint, sample_time=None)
        self.__sampling_time = sample_time
        self.__initial_setpoint = setpoint
        self.__setpoint_history = [setpoint]
        self.reset_history()

    def __call__(self, u):
        dt = self.get_timestamps()[-1] + self.__sampling_time
        return super().__call__(u, dt)

    def set_setpoint(self, new_SP):
        self.setpoint = new_SP

    def get_setpoint(self):
        return self.setpoint

    def get_setpoint_hist(self):
        return self.__setpoint_history

    def set_CV_limit(self, lower_bound, upper_bound):
        self.output_limits = (lower_bound, upper_bound)

    def get_CV_limit(self):
        return self.output_limits

    def set_proportional_on_measurement(self, set_val):
        self.proportional_on_measurement = set_val

    def set_differential_on_measurement (self, set_val):
        self.differential_on_measurement  = set_val

    def get_proportional_on_measurement(self):
        return self.proportional_on_measurement

    def get_differential_on_measurement (self):
        return self.differential_on_measurement

    def simulate_step(self, u):
        t_prev = self.get_timestamps()[-1]
        tt = t_prev + self.__sampling_time

        cv = self(u)
        self.__update_history(u, cv, tt, self.setpoint)
        return self(u)

    def __update_history(self, input_t, output_t, ts, sp):
        self.__log_input(input_t)
        self.__log_output(output_t)
        self.__log_timestamp(ts)
        self.__log_setpoint(sp)
        
    def reset_history(self):
        self.__input_hist = [0]
        self.__output_hist = [0]
        self.__timestamps = [0]
        self.__setpoint_history = [self.__initial_setpoint]
        super().__init__(self.Kp, self.Ki, self.Kd, setpoint=self.__initial_setpoint, sample_time=None)
        
    def show_history(self):
        print('History of inputs:')
        print(self.get_input_hist())
        print('')
        print('History of outputs:')
        print(self.get_output_hist())

    def get_timestamps(self):
        return self.__timestamps
        
    def get_input_hist(self):
        return self.__input_hist

    def get_output_hist(self):
        return self.__output_hist

    def get_setpoint_hist(self):
        return self.__setpoint_history

    def __log_timestamp(self, ts):
        self.__timestamps.append(ts)
        
    def __log_input(self, u):
        self.__input_hist.append(u)

    def __log_output(self, y):
        self.__output_hist.append(y)

    def __log_setpoint(self, sp):
        self.__setpoint_history.append(sp)