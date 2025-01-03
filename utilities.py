import libs as Egf

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

def analyze_child(d, yaml_key, all_yaml_keys):

    for k,v in d:
        if(type(v)==dict):
            yaml_key.append(k)
            analyze_child(v.items(), yaml_key, all_yaml_keys)
        else:
            yaml_key.append(k)
            all_yaml_keys.append(yaml_key)
            yaml_key=yaml_key[:-1]

pass
    

def parse_all_yaml_params():
    all_yaml_keys = []
    yaml_key = []

    all_yaml_keys = analyze_child(Egf.config.items(), yaml_key, all_yaml_keys)


def parse_yaml_params(root_param=None):
    params = { }

    get_param(['params_mock_cat','log_amplitude','fluctuations','asperity'], params)
    get_param(['params_mock_cat','log_amplitude','fluctuations','flexibility'], params)
    get_param(['params_mock_cat','log_amplitude','fluctuations','fluctuations'], params)
    get_param(['params_mock_cat','log_amplitude','fluctuations','loglogavgslope'], params)
    get_param(['params_mock_cat','log_amplitude','offset','offset_mean'], params)
    get_param(['params_mock_cat','log_amplitude','offset','offset_std'], params)
    get_param(['params_mock_cat','sign','fluctuations','asperity'], params)
    get_param(['params_mock_cat','sign','fluctuations','flexibility'], params)
    get_param(['params_mock_cat','sign','fluctuations','fluctuations'], params)
    get_param(['params_mock_cat','sign','fluctuations','loglogavgslope'], params)
    get_param(['params_mock_cat','sign','offset','offset_mean'], params)
    get_param(['params_mock_cat','sign','offset','offset_std'], params)
    get_param(['params_mock_cat','maker_params','seed_inf'], params)
    get_param(['params_mock_cat','maker_params','seed_cat'], params)
    get_param(['params_mock_cat','maker_params','maker_type'], params)
    get_param(['params_mock_cat','maker_params','disk_on'], params)
    get_param(['params','n_eg_params'], params)
    get_param(['params','nside'], params)
    get_param(['params','cat_path'], params)
    get_param(['params','n_los'], params)
    get_param(['params','nglobal'], params)
    get_param(['params','plot_path'], params)
    get_param(['params','results_path'], params)

    return params