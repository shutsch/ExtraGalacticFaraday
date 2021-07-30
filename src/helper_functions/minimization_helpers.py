import numpy as np
import nifty7 as ift


def build_controller_dict(minimization_dict, iteration_number, final, master, logger):
    c_dict = {}
    for key, m_dict in minimization_dict.items():
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
        logger.info('GLOBAL STEP CONFIG (step #{}):: {}:  #{} (iterations/samples)'.format(iteration_number, key, n))
        if m_dict['type'] == 'Samples':
            c_dict.update({'samples': int(n)})
        else:
            assert hasattr(ift, m_dict['type'] + 'Controller'), 'Unknown Energy controller required'

            ic = getattr(ift, m_dict['type'] + 'Controller')\
                (name=key + ' (main task)' if master else key + str(iteration_number),
                 iteration_limit=n, **m_dict['controller_params'])
            c_dict.update({key: ic})
    return c_dict
