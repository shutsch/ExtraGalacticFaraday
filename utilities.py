import libs as Egf

def get_galactic_model(sky_domain, params):
    log_amplitude_params = {'fluctuations': {'asperity': params['params_mock_cat.log_amplitude.fluctuations.asperity'], 
                                            'flexibility': params['params_mock_cat.log_amplitude.fluctuations.flexibility'],  
                                            'fluctuations': params['params_mock_cat.log_amplitude.fluctuations.fluctuations'], 
                                            'loglogavgslope': params['params_mock_cat.log_amplitude.fluctuations.loglogavgslope'], },
                            'offset': {'offset_mean': params['params_mock_cat.log_amplitude.offset.offset_mean'], 
                                      'offset_std': params['params_mock_cat.log_amplitude.offset.offset_std']},}

    sign_params = {'fluctuations': {'asperity': params['params_mock_cat.sign.fluctuations.asperity'], 
                                            'flexibility': params['params_mock_cat.sign.fluctuations.flexibility'],  
                                            'fluctuations': params['params_mock_cat.sign.fluctuations.fluctuations'], 
                                            'loglogavgslope': params['params_mock_cat.sign.fluctuations.loglogavgslope'], },
                            'offset': {'offset_mean': params['params_mock_cat.sign.offset.offset_mean'], 
                                      'offset_std': params['params_mock_cat.sign.offset.offset_std']},}

    return Egf.Faraday2020Sky(sky_domain, **{'log_amplitude_parameters': log_amplitude_params,
                                                       'sign_parameters': sign_params})


def get_param(keys, params):
    part_dict = Egf.config
    param_name = ""
    for k in keys:
        part_dict = part_dict[k]
        param_name+=f'{k}.'
    param_name=param_name[:-1]

    params[param_name] = part_dict

    if part_dict == "None": 
        params[param_name] = None
    elif part_dict != "None" and type(part_dict) == str and ',' in part_dict:
        params[param_name] = [float(s) for s in part_dict.split(',')]

yaml_key = []
all_yaml_keys = []

def analyze_node(y):
    global yaml_key

    for k,v in y:
        yaml_key.append(k)
        if type(v) == dict:
            analyze_node(v.items())
        else:
            all_yaml_keys.append(yaml_key.copy())
            yaml_key.pop()

    if len(yaml_key)>0: yaml_key.pop()

def get_all_yaml_params():
    global all_yaml_keys
    y=Egf.config.items()

    analyze_node(y)

    return all_yaml_keys

def parse_all_yaml_params(yaml_params):
    params = { }

    for yaml_param in yaml_params:
        get_param(yaml_param, params)

    return params

def parse_yaml_params(root_param=None):
    pass