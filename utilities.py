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

def analyze_node(k, v, yaml_key, all_yaml_keys, father_k):

    if(type(v)==dict):
        if yaml_key is None: 
            yaml_key=[] 
        yaml_key.append(k)
        for _k,_v in v.items():
            yaml_key = analyze_node(_k, _v, yaml_key, all_yaml_keys, father_k)
    else:
        yaml_key.append(k)
        if not father_k in yaml_key : yaml_key.insert(0,father_k)
        all_yaml_keys.append(yaml_key)
        return yaml_key[:-1]

def get_all_yaml_params():
    all_yaml_keys = []
    yaml_key = []
    y=Egf.config.items()

    for k,v in y:
        yaml_key=[]
        father_k = k
        analyze_node(k, v, yaml_key, all_yaml_keys, father_k)

    return all_yaml_keys

def parse_all_yaml_params(yaml_params):
    params = { }

    for yaml_param in yaml_params:
        get_param(yaml_param, params)

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
    get_param(['params_mock_cat','maker_params','eg_on'], params)
    get_param(['params','n_eg_params'], params)
    get_param(['params','nside'], params)
    get_param(['params','cat_path'], params)
    get_param(['params','n_los'], params)
    get_param(['params','nglobal'], params)
    get_param(['params','plot_path'], params)
    get_param(['params','results_path'], params)
    get_param(['params','n_samples_posterior'], params)
    get_param(['params','n_single_fit'], params)
    get_param(['params','resume'], params)
    get_param(['controllers','minimizer','n'], params)
    get_param(['controllers','minimizer','n_final'], params)
    get_param(['controllers','minimizer','increase_step'], params)
    get_param(['controllers','minimizer','increase_rate'], params)
    get_param(['controllers','minimizer','deltaE_threshold'], params)
    get_param(['controllers','minimizer','deltaE_start'], params)
    get_param(['controllers','minimizer','deltaE_end'], params)
    get_param(['controllers','minimizer','convergence_level'], params)
    get_param(['controllers','minimizer','eg_thresh'], params)
    get_param(['controllers','sampler','n'], params)
    get_param(['controllers','sampler','n_final'], params)
    get_param(['controllers','sampler','increase_step'], params)
    get_param(['controllers','sampler','increase_rate'], params)
    get_param(['controllers','sampler','deltaE'], params)
    get_param(['controllers','sampler','convergence_level'], params)
    get_param(['controllers','sampler_eg','n'], params)
    get_param(['controllers','sampler_eg','n_final'], params)
    get_param(['controllers','sampler_eg','increase_step'], params)
    get_param(['controllers','sampler_eg','increase_rate'], params)
    get_param(['controllers','sampler_eg','deltaE'], params)
    get_param(['controllers','sampler_eg','convergence_level'], params)
    get_param(['controllers','minimizer_eg','n'], params)
    get_param(['controllers','minimizer_eg','n_final'], params)
    get_param(['controllers','minimizer_eg','increase_step'], params)
    get_param(['controllers','minimizer_eg','increase_rate'], params)
    get_param(['controllers','minimizer_eg','deltaE'], params)
    get_param(['controllers','minimizer_eg','convergence_level'], params)
    get_param(['controllers','minimizer_samples','n'], params)
    get_param(['controllers','minimizer_samples','n_final'], params)
    get_param(['controllers','minimizer_samples','increase_step'], params)
    get_param(['controllers','minimizer_samples','increase_rate'], params)
    get_param(['controllers','minimizer_samples','deltaE'], params)
    get_param(['controllers','minimizer_samples','convergence_level'], params)
    get_param(['sample_params','n'], params)
    get_param(['sample_params','n_prior'], params)
    get_param(['sample_params','n_final'], params)
    get_param(['sample_params','increase_step'], params)
    get_param(['sample_params','increase_rate'], params)
    


    return params