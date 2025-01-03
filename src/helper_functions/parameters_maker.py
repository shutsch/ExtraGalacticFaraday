import libs as Egf 
import utilities as U

class Parameters_maker():
    def __init__(self):
        self.yaml_values = U.parse_all_yaml_params() #TODO: TO BE IMPLEMENTED

    def get_parsed_params(root_param=None):
        return U.parse_yaml_params(root_param)
