import libs as Egf 
import utilities as U

class Parameters_maker():
    def __init__(self):
        all_yaml_params = U.get_all_yaml_params()
        self.yaml_values = U.parse_all_yaml_params(all_yaml_params)
