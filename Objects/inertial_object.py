class Inertial_Object(TF_object):
    def __init__(self, time_const, gain):
        self.set_param('time_const', time_const)
        self.set_param('gain', gain)
        numerator = [gain]
        denominator = [time_const, 1]
        super().__init__(numerator, denominator)


    def summary(self):
        messages = ['This is an intertial object.']
        messages.append(f'Gain: {self.__gain}')
        messages.append(f'Time constant: {self.__time_const}')
        super().summary(messages)

    def __validate_matrix_param(self, param_value):
        return type(param_value) == np.ndarray

    def __convert_param_to_matrix(self, param_value):
        conv = converters.to_array_converter()
        try:
            param_matrix = conv.convert_any_type(param_value)
        except:
            raise TypeError('Parameter value has to be a matrix or convertible to a matrix')
        return param_matrix

    def get_param(self, param_name):
        return eval(f'_{self.__class__.__name__}__{param_name}')

    def set_param(self, param_name, param_value):
        if not self.__validate_matrix_param(param_value):
            try: 
                param_matrix = self.__convert_param_to_matrix(param_value)
            except:
                raise TypeError(f'{param_name} has to be a matrix or convertible to a matrix')
                
        setattr(self, f"_{self.__class__.__name__}__{param_name}", param_matrix)

    def simulate_step(self, u, t):
        return super().simulate_step(u, t)