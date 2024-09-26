import numpy as np


class to_array_converter:
    def __init__(self):
        self.__allowed_vec_types = ('vertical', 'horizontal')
        self.__allowed_input_types = (int, float, list, tuple, np.float64, np.int64)
        self.__skip_types = [np.ndarray]
        
    def __convert_scalar(self, input):
        return np.array([[input]])

    def __convert_vector(self, input, vec_type):
        return np.array(input).reshape(1, -1) if vec_type == 'horizontal' else np.array(input).reshape(-1, 1)

    def __convert_2d_list(self, input):
        row_lengths = [len(row) for row in input]
        if len(set(row_lengths)) != 1:
            raise ValueError('All sublists has to be of equal length')
        return np.array(input)

    def __if_skip(self, input):
        if type(input) in self.__skip_types:
            if input.ndim == 2:
                return True
        else:
            return False
            
    def convert_any_type(self, input, vec_type='vertical'):
        if self.__if_skip(input):
            return input
            
        if type(input) not in self.__allowed_input_types:
            raise TypeError('Provided input is not supported. Currently supported types:', self.__allowed_input_types)

        # Single value
        if type(input) in (int, float, np.float64, np.int64):
            array2d = self.__convert_scalar(input)

        # A list or tuple
        if type(input) in (tuple, list):
            # Empty vector
            if len(input) == 0:
                raise ValueError('Cannot convert an empty list')

            # One-element vector
            elif len(input) == 1:
                if type(input[0]) in (float, int):
                    array2d = self.__convert_scalar(input[0])
                elif type(input[0] in (tuple, list)):
                    array2d = self.__convert_vector(input[0], vec_type)
                    

            # Multi-element vector
            elif len(input) > 1:
                # A list of lists
                if all(isinstance(row, (list, tuple)) for row in input):
                    array2d = self.__convert_2d_list(input)
                # Normal vector
                else:
                    if vec_type not in self.__allowed_vec_types:
                        raise TypeError('Input data vectors has to be either horizontal or vertical')
                    else:
                        array2d = self.__convert_vector(input, vec_type)
                                   
        return array2d


class to_vector_converter:
    def __init__(self):
        self.__allowed_vec_types = ('vertical', 'horizontal')
        self.__allowed_input_types = (int, float, list, tuple, np.ndarray)

    def __convert_scalar2array(self, scalar):
        return np.array([scalar])

    def __convert_list2array(self, iterable):
        return np.array(iterable)
        
    def convert_any_type(self, input, vec_type="vertical"):
        if vec_type not in self.__allowed_vec_types:
            raise ValueError("vec_type must be 'vertical' or 'horizontal'")

        if isinstance(input, (int, float)):
            input = self.__convert_scalar2array(input)

        if isinstance(input, (list, tuple)):
            input = self.__convert_list2array(input)

        if type(input) not in self.__allowed_input_types:
            raise TypeError("Input must be of type int, float, list, or numpy.ndarray")

        if input.ndim > 2:
            raise ValueError("Input must be a 1-dimensional array")

        if vec_type == "vertical":
            return input.reshape(-1, 1)  # Konwersja na wektor kolumnowy
        else:
            return input.reshape(1, -1)  # Konwersja na wektor wierszowy