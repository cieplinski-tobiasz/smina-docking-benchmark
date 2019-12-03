import logging


def setup_and_get_logger(debug=False, name=__name__):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=level)
    return logging.getLogger(name)


def set_keras_cores(num_cores):
    import keras
    import tensorflow as tf

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores)
    session = tf.Session(config=config)
    keras.backend.set_session(session)
