class Format:
    end = '\033[0m'
    underline = '\033[4m'


def logging_paragraph(loggerer, string):
    loggerer.info('\n')
    loggerer.info(120 * '+')
    loggerer.info('\n')
    loggerer.info(Format.underline + string + Format.end)


def configuration_logging(pd, log):
    c = 120*'+'
    log.info(c)
    log.info(Format.underline + 'STARTING RUN {}'.format(pd['run']['identifier']) + Format.end)
    log.info(c + '\n')
    log.info(Format.underline + 'ACTIVE LIKELIHOODS:' + Format.end)
    for i, l in enumerate(pd['run']['likelihoods']):
        if pd['run']['likelihoods'][l]:
            log.info('{}: {}'.format(i, l))
    log.info('\n')
    log.info(c + '\n')
    log.info(Format.underline + 'RUN CONFIG:' + Format.end)
    for k, co in pd['run'].items():
        if k not in ['likelihoods', 'components']:
            log.info('{}: {}'.format(k, co))
    log.info('\n\n\n')
    log.info(Format.underline + 'EVALUATION CONFIG:' + Format.end)
    for k, co in pd['evaluation'].items():
        if k not in ['GEO', 'MGVI', 'AV', 'type']:
            log.info('{}: {}'.format(k, co))
    log.info('\nThe posterior is evaluated using: {} \n'.format(pd['evaluation']['type']))
    log.info(c + '\n')


def _logger_init():
    import logging
    res = logging.getLogger('Faraday')
    res.setLevel(logging.DEBUG)
    res.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    res.addHandler(ch)
    return res


logger = _logger_init()
