import libs as Egf

def parse_yaml_params():
    params = {
       'log_amplitude.fluctuations.asperity' :  Egf.config['params_mock_cat']['log_amplitude']['fluctuations']['asperity'],
       'log_amplitude.fluctuations.flexibility' :  Egf.config['params_mock_cat']['log_amplitude']['fluctuations']['flexibility'],
       'log_amplitude.fluctuations.fluctuations' :  Egf.config['params_mock_cat']['log_amplitude']['fluctuations']['fluctuations'],
       'log_amplitude.fluctuations.loglogavgslope' :  Egf.config['params_mock_cat']['log_amplitude']['fluctuations']['loglogavgslope'],
       'log_amplitude.offset.offset_mean' :  Egf.config['params_mock_cat']['log_amplitude']['offset']['offset_mean'],
       'log_amplitude.offset.offset_std' :  Egf.config['params_mock_cat']['log_amplitude']['offset']['offset_std'],
       'sign.fluctuations.asperity' :  Egf.config['params_mock_cat']['sign']['fluctuations']['asperity'],
       'sign.fluctuations.flexibility' :  Egf.config['params_mock_cat']['sign']['fluctuations']['flexibility'],
       'sign.fluctuations.fluctuations' :  Egf.config['params_mock_cat']['sign']['fluctuations']['fluctuations'],
       'sign.fluctuations.loglogavgslope' :  Egf.config['params_mock_cat']['sign']['fluctuations']['loglogavgslope'],
       'sign.offset.offset_mean' :  Egf.config['params_mock_cat']['sign']['offset']['offset_mean'],
       'sign.offset.offset_std' :  Egf.config['params_mock_cat']['sign']['offset']['offset_std'],
    }

    l_asp = Egf.config['params_mock_cat']['log_amplitude']['fluctuations']['asperity']
    if l_asp == "None": 
        params['log_amplitude.fluctuations.asperity'] = None
    elif l_asp != "None" and type(l_asp) == str and ',' in l_asp:
        params['log_amplitude.fluctuations.asperity'] = [float(s) for s in l_asp.split(',')]

    l_flex = Egf.config['params_mock_cat']['log_amplitude']['fluctuations']['flexibility']
    if l_flex == "None": 
        params['log_amplitude.fluctuations.flexibility'] = None
    if l_flex != "None" and type(l_flex) == str and ',' in l_flex:
        params['log_amplitude.fluctuations.flexibility'] = [float(s) for s in l_flex.split(',')]

    l_fluct = Egf.config['params_mock_cat']['log_amplitude']['fluctuations']['fluctuations']
    if l_fluct == "None": 
        params['log_amplitude.fluctuations.fluctuations'] = None
    if l_fluct != "None" and type(l_fluct) == str and ',' in l_fluct:
        params['log_amplitude.fluctuations.fluctuations'] = [float(s) for s in l_fluct.split(',')]

    l_avg = Egf.config['params_mock_cat']['log_amplitude']['fluctuations']['loglogavgslope']
    if l_avg == "None": 
        params['log_amplitude.fluctuations.loglogavgslope'] = None
    if l_avg != "None" and type(l_avg) == str and ',' in l_avg:
        params['log_amplitude.fluctuations.loglogavgslope'] = [float(s) for s in l_avg.split(',')]

    l_offset_mean = Egf.config['params_mock_cat']['log_amplitude']['offset']['offset_mean']
    if l_offset_mean == "None": 
        params['log_amplitude.offset.offset_mean'] = None
    if l_offset_mean != "None" and type(l_offset_mean) == str and ',' in l_offset_mean:
        params['log_amplitude.offset.offset_mean'] = [float(s) for s in l_offset_mean.split(',')]
    
    l_offset_std = Egf.config['params_mock_cat']['log_amplitude']['offset']['offset_std']
    if l_offset_std == "None": 
        params['log_amplitude.offset.offset_std'] = None
    if l_offset_std != "None" and type(l_offset_std) == str and ',' in l_offset_std:
        params['log_amplitude.offset.offset_std'] = [float(s) for s in l_offset_std.split(',')]

    s_asp = Egf.config['params_mock_cat']['sign']['fluctuations']['asperity']
    if s_asp == "None": 
        params['sign.fluctuations.asperity'] = None
    if s_asp != "None" and type(s_asp) == str and ',' in s_asp:
        params['sign.fluctuations.asperity'] = [float(s) for s in s_asp.split(',')]

    s_flex = Egf.config['params_mock_cat']['sign']['fluctuations']['flexibility']
    if s_flex == "None": 
        params['sign.fluctuations.flexibility'] = None
    if s_flex != "None" and type(s_flex) == str and ',' in s_flex:
        params['sign.fluctuations.flexibility'] = [float(s) for s in s_flex.split(',')]

    s_fluct = Egf.config['params_mock_cat']['sign']['fluctuations']['fluctuations']
    if s_fluct == "None": 
        params['sign.fluctuations.fluctuations'] = None
    if s_fluct != "None" and type(s_fluct) == str and ',' in s_fluct:
        params['sign.fluctuations.fluctuations'] = [float(s) for s in s_fluct.split(',')]

    s_avg = Egf.config['params_mock_cat']['sign']['fluctuations']['loglogavgslope']
    if s_avg == "None": 
        params['sign.fluctuations.loglogavgslope'] = None
    if s_avg != "None" and type(s_avg) == str and ',' in s_avg:
        params['sign.fluctuations.loglogavgslope'] = [float(s) for s in s_avg.split(',')]

    s_offset_mean = Egf.config['params_mock_cat']['sign']['offset']['offset_mean']
    if s_offset_mean == "None": 
        params['sign.offset.offset_mean'] = None
    if s_offset_mean != "None" and type(s_offset_mean) == str and ',' in s_offset_mean:
        params['sign.offset.offset_mean'] = [float(s) for s in s_offset_mean.split(',')]
    
    s_offset_std = Egf.config['params_mock_cat']['sign']['offset']['offset_std']
    if s_offset_std == "None": 
        params['sign.offset.offset_std'] = None
    if s_offset_std != "None" and type(s_offset_std) == str and ',' in s_offset_std:
        params['sign.offset.offset_std'] = [float(s) for s in s_offset_std.split(',')]

    return params