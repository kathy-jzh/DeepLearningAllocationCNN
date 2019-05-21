"""
Hyperparameter storage
"""

###################### General

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TF_OPTIMIZER = 'Adam'
DEFAULT_LOG_ENV = 'training_and_validation'
NB_OF_EPOCHS_FOR_BAYESIAN = 10
FRAC_BATCH_TO_DISPLAY = 0.1

DEFAULT_FILES_NAMES = ['stockData_' + str(i) for i in range(1, 28)]
DEFAULT_START_DATE = 19900101
DEFAULT_END_DATE = 20200102

###################### For AlexNet

AlexNet_hyperparams = {
    # 1st Conv Layer block
    "conv1_conv_kernel": (7, 7, 5, 10),

    # 2nd Conv Layer block
    "conv2_conv_kernel": (3, 3, 10, 10),

    # 3rd Conv Layer block
    "conv3_conv_kernel": (2, 2, 10, 8),

    # 4rd Conv Layer block
    "conv4_conv_kernel": (2, 2, 8, 4),
}

##################### For GoogLeNet

# hyperparameter list for 1st inception module
inception1_hyperparam = {
    # 1x1 pathway
    "1x1_conv_kernel": (1, 1, 8, 6),

    # 1x1 to 3x3 pathway,
    "3x3_conv_kernel1": (1, 1, 8, 6),
    "3x3_conv_kernel2": (3, 3, 6, 8),

    # 1x1 to 5x5 pathway
    "5x5_conv_kernel1": (1, 1, 8, 6),
    "5x5_conv_kernel2": (5, 5, 6, 8),

    # 3x3 to 1x1 pathway
    "pooling1_conv_kernel": (3, 3, 8, 3)
}

# hyperparameter list for 2nd inception module
inception2_hyperparam = {
    # 1x1 pathway
    "1x1_conv_kernel": (1, 1, 25, 4),

    # 1x1 to 3x3 pathway
    "3x3_conv_kernel1": (1, 1, 25, 4),
    "3x3_conv_kernel2": (3, 3, 4, 6),

    # 1x1 to 5x5 pathway
    "5x5_conv_kernel1": (1, 1, 25, 4),
    "5x5_conv_kernel2": (5, 5, 4, 6),

    # 3x3 to 1x1 pathway
    "pooling1_conv_kernel": (3, 3, 25, 3)
}

# hyperparameter list for 3rd inception module
inception3_hyperparam = {
    # 1x1 pathway
    "1x1_conv_kernel": (1, 1, 19, 3),

    # 1x1 to 3x3 pathway
    "3x3_conv_kernel1": (1, 1, 19, 3),
    "3x3_conv_kernel2": (3, 3, 3, 4),

    # 1x1 to 5x5 pathway
    "5x5_conv_kernel1": (1, 1, 19, 3),
    "5x5_conv_kernel2": (5, 5, 3, 4),

    # 3x3 to 1x1 pathway
    "pooling1_conv_kernel": (3, 3, 19, 3),
}
first_block_gglnet_hyperparam = {
    # 1st convolutional layer block
    "conv1_conv_kernel": (4, 4, 5, 16),

    # 2nd convolutional layer block
    "conv2_conv_kernel": (3, 3, 16, 8)
}

GoogleNet_hyperparams = {'first_block': first_block_gglnet_hyperparam,
                         'inception_1': inception1_hyperparam,
                         'inception_2': inception2_hyperparam,
                         'inception_3': inception3_hyperparam,
                         'dropout': 0.15,
                         'learning_rate':0.003,
                         }

ResNet_hyperparams = {'learning_rate':0.003,'tf_optimizer':'rmsprop','dropout':0.1}
