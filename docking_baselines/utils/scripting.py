def set_keras_cores(num_cores):
    import keras
    import tensorflow as tf

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, inter_op_parallelism_threads=num_cores)
    session = tf.Session(config=config)
    keras.backend.set_session(session)
