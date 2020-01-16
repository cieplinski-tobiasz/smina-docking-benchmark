import logging


def setup_and_get_logger(debug=False, name=__name__):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=level)
    return logging.getLogger(name)


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')
