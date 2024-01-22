import numpy as np
import nifty8 as ift


def get_controller(m_dict, iteration_number, final, key):
    """
    This routine
    :param m_dict:
    :param iteration_number:
    :param final:
    :param key:
    :return:
    """
    n = _get_adapted_n(m_dict, iteration_number, final)
    if not hasattr(ift, m_dict['type'] + 'Controller'):
        raise 'Unknown Energy controller required'

    ic = getattr(ift, m_dict['type'] + 'Controller')(name=key, iteration_limit=n,
                                                     **m_dict['controller_params'])

    return ic


def get_n_samples(s_dict, iteration_number, final):
    """
    This routine
    :param s_dict:
    :param iteration_number:
    :param final:
    :return:
    """
    n = _get_adapted_n(s_dict, iteration_number, final)
    return n


def _get_adapted_n(m_dict, iteration_number, final):
    if 'change_params' in m_dict and m_dict['change_params'] is not None:
        change_dict = m_dict['change_params']
        if iteration_number == 0 and 'n_prior' in change_dict and change_dict['n_prior'] is not None:
            n = change_dict['n_prior']
        elif final and 'n_final' in change_dict and change_dict['n_final'] is not None:
            n = change_dict['n_final']
        else:
            assert 'n' in m_dict
            i_step = change_dict['increase_step'] if 'increase_step' in change_dict else None
            i_rate = change_dict['increase_rate'] if 'increase_rate' in change_dict else None
            if i_step is None or i_rate is None:
                factor = 1
            else:
                factor = max(1, np.floor(iteration_number / i_step) ** i_rate)
            n_final = change_dict['n_final'] if 'n_final' in change_dict else None
            if n_final is not None:
                n = min(n_final, m_dict['n'] * factor)
            else:
                n = m_dict['n'] * factor
    else:
        n = m_dict['n']
    return n
