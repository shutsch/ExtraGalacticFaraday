import libs as Egf

def get_param(s, params):
    p = Egf.config[s[0]][s[1]][s[2]][s[3]]
    param_name = f'{s[1]}.{s[2]}.{s[3]}'
    params[param_name] = p

    if p == "None": 
        params[param_name] = None
    elif p != "None" and type(p) == str and ',' in p:
        params[param_name] = [float(s) for s in p.split(',')]

def parse_yaml_params():
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

    return params