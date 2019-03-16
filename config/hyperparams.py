DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TF_OPTIMIZER = 'Adam' # todo try to avoid tensorflow in here
DEFAULT_LOG_ENV = 'training_and_validation'
NB_OF_EPOCHS_FOR_BAYESIAN = 10
FRAC_BATCH_TO_DISPLAY = 0.1

DEFAULT_FILES_NAMES = ['stockData_'+str(i) for i in range(1,28)]
DEFAULT_START_DATE = 19900101
DEFAULT_END_DATE = 20200102

# Weight parameters as devised in the original research paper
AlexNet_hyperparams = {
    # 1st Conv Layer block
    "conv1_conv_kernel": (2, 2, 4, 6),
    
    # 2nd Conv Layer block
    "conv2_conv_kernel": (2, 2, 6, 10),
    # 3rd Conv Layer block
    "conv3_conv_kernel": (2, 2, 10, 8),
    
    # 3rd Conv Layer block
    "conv4_conv_kernel": (2, 2, 8, 4),
}

inception1_hyperparam={
    # 1x1 pathway
    "1x1_conv_kernel": (1, 1, 8, 6),
    
    # 1x1 to 3x3 pathway,
    "3x3_conv_kernel1":(1,1,8,6),
    "3x3_conv_kernel2": (3,3,6,8),
    
    # 1x1 to 5x5 pathway
    "5x5_conv_kernel1": (1, 1, 8, 6),
    "5x5_conv_kernel2": (5, 5, 6, 8),
    
    # 3x3 to 1x1 pathway
    "pooling1_conv_kernel": (3, 3, 8, 3)
}

# hyperparameter list for 2nd inception module
inception2_hyperparam={
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
inception3_hyperparam={
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
    # "conv1_conv_kernel": (4,4,4,16),
    "conv1_conv_kernel": (4,4,5,16),

    # 2nd convolutional layer block
    "conv2_conv_kernel": (3, 3, 16, 8)
}
# hyperparameter list for other convolutional parts in GoogLeNet
GoogleNet_hyperparams = {'first_block':first_block_gglnet_hyperparam,
                         'inception_1':inception1_hyperparam,
                         'inception_2':inception2_hyperparam,
                         'inception_3':inception3_hyperparam,
                         'dropout':0.15,
}

VAE_hyperparam = {
    
    # encoder 1st convolutional layer
    "encoder_conv1_kernel": (2, 2, 4, 6),
    
    # decoder 2nd convolutional layer
    "encoder_conv2_kernel": (2, 2, 6, 8),
    
    # decoder 1st convolutional layer
    "decoder_conv1_kernel": (2, 2, 4, 8),
    
    # decoder 2nd convolutional layer
    "decoder_conv2_kernel": (2, 2, 8, 4)
        
}


PERMNOS_to_avoid = [10180,
 10253,
 10258,
 10333,
 10443,
 10550,
 10656,
 10696,
 10838,
 10892,
 11018,
 11308,
 11394,
 11481,
 11618,
 11628,
 11644,
 11762,
 11809,
 12266,
 14593,
 16126,
 16505,
 16678,
 17144,
 18148,
 19166,
 20053,
 20482,
 20598,
 22323,
 22509,
 23393,
 23536,
 23887,
 24643,
 25232,
 26614,
 26649,
 27959,
 27983,
 28118,
 29867,
 30737,
 30940,
 31051,
 32803,
 34666,
 37381,
 37568,
 37584,
 38295,
 38659,
 38762,
 39571,
 43553,
 44813,
 45225,
 47677,
 51263,
 51692,
 52337,
 53373,
 55001,
 55212,
 55634,
 57293,
 57568,
 57665,
 57809,
 57904,
 58836,
 58975,
 59483,
 59504,
 61815,
 62033,
 62148,
 62156,
 63263,
 63467,
 63546,
 64450,
 64629,
 64822,
 64961,
 65402,
 66325,
 66835,
 69586,
 70703,
 74500,
 75047,
 75261,
 75470,
 75672,
 75844,
 75905,
 76082,
 76215,
 76221,
 76230,
 76261,
 76360,
 76582,
 76613,
 76804,
 76858,
 76932,
 76963,
 77175,
 77202,
 77236,
 77259,
 77274,
 77437,
 77486,
 77501,
 77699,
 78002,
 78015,
 78044,
 78071,
 78081,
 78156,
 78172,
 78189,
 78569,
 78664,
 78688,
 78705,
 78927,
 78990,
 79006,
 79022,
 79039,
 79065,
 79066,
 79249,
 79303,
 79315,
 79363,
 79382,
 79444,
 79474,
 79571,
 79636,
 79686,
 79702,
 79790,
 79878,
 79903,
 80069,
 80167,
 80193,
 80297,
 80306,
 80307,
 80320,
 80368,
 80563,
 80837,
 80912,
 80962,
 81013,
 81046,
 81282,
 81481,
 81527,
 81677,
 81678,
 81736,
 81774,
 81784,
 82162,
 82276,
 82281,
 82526,
 82552,
 82575,
 82642,
 82649,
 82747,
 82762,
 82779,
 82812,
 82824,
 82833,
 83111,
 83217,
 83221,
 83225,
 83303,
 83382,
 83443,
 83486,
 83651,
 83756,
 83762,
 83835,
 83862,
 83885,
 84007,
 84041,
 84052,
 84184,
 84206,
 84234,
 84302,
 84386,
 84419,
 84511,
 84566,
 84581,
 84601,
 84604,
 84607,
 84734,
 84769,
 84819,
 84820,
 84827,
 85035,
 85187,
 85198,
 85261,
 85293,
 85320,
 85502,
 85567,
 85576,
 85706,
 85863,
 85905,
 85992,
 86083,
 86121,
 86158,
 86165,
 86233,
 86356,
 86382,
 86444,
 86526,
 86563,
 86728,
 86763,
 86810,
 86812,
 86839,
 86949,
 87043,
 87236,
 87251,
 87337,
 87439,
 87444,
 87471,
 87508,
 87510,
 87583,
 87608,
 87657,
 87825,
 88031,
 88159,
 88173,
 88177,
 88208,
 88229,
 88240,
 88264,
 88290,
 88309,
 88332,
 88352,
 88370,
 88391,
 88402,
 88403,
 88417,
 88434,
 88468,
 88511,
 88604,
 88626,
 88678,
 88729,
 88742,
 88779,
 88854,
 88865,
 88873,
 88893,
 88912,
 88994,
 89017,
 89029,
 89043,
 89056,
 89059,
 89063,
 89139,
 89233,
 89244,
 89269,
 89284,
 89323,
 89369,
 89393,
 89403,
 89445,
 89547,
 89548,
 89571,
 89574,
 89626,
 89648,
 89756,
 89790,
 89866,
 89875,
 89889,
 89898,
 89901,
 89925,
 89927,
 89935,
 89942,
 89949,
 89960,
 89988,
 90031,
 90077,
 90098,
 90101,
 90121,
 90125,
 90126,
 90175,
 90207,
 90213,
 90215,
 90279,
 90288,
 90298,
 90307,
 90312,
 90329,
 90338,
 90361,
 90384,
 90432,
 90476,
 90483,
 90496,
 90516,
 90525,
 90539,
 90589,
 90595,
 90603,
 90604,
 90608,
 90618,
 90634,
 90640,
 90703,
 90704,
 90735,
 90791,
 90805,
 90829,
 90857,
 90875,
 90907,
 90926,
 90943,
 90948,
 90952,
 90955,
 90976,
 90986,
 90993,
 91004,
 91010,
 91018,
 91021,
 91040,
 91082,
 91111,
 91119,
 91124,
 91233,
 91269,
 91283,
 91301,
 91307,
 91309,
 91310,
 91311,
 91312,
 91313,
 91314,
 91316,
 91330,
 91348,
 91356,
 91374,
 91379,
 91384,
 91385,
 91386,
 91387,
 91392,
 91395,
 91416,
 91450,
 91457,
 91471,
 91472,
 91488,
 91491,
 91498,
 91556,
 91606,
 91615,
 91627,
 91656,
 91664,
 91677,
 91678,
 91684,
 91717,
 91718,
 91720,
 91721,
 91727,
 91729,
 91734,
 91742,
 91774,
 91776,
 91777,
 91778,
 91779,
 91780,
 91781,
 91782,
 91784,
 91785,
 91786,
 91787,
 91788,
 91789,
 91790,
 91793,
 91795,
 91796,
 91797,
 91801,
 91808,
 91809,
 91810,
 91811,
 91832,
 91836,
 91845,
 91858,
 91947,
 91954,
 91964,
 91973,
 92011,
 92020,
 92025,
 92032,
 92040,
 92061,
 92089,
 92091,
 92130,
 92131,
 92155,
 92213,
 92240,
 92248,
 92251,
 92252,
 92337,
 92345,
 92352,
 92380,
 92385,
 92409,
 92410,
 92411,
 92432,
 92448,
 92453,
 92456,
 92470,
 92486,
 92520,
 92521,
 92523,
 92543,
 92557,
 92571,
 92594,
 92606,
 92611,
 92621,
 92662,
 92665,
 92687,
 92688,
 92700,
 92748,
 92753,
 92777,
 92778,
 92812,
 92813,
 92814,
 92816,
 92817,
 92819,
 92842,
 92843,
 92844,
 92845,
 92855,
 92856,
 92858,
 92860,
 92863,
 92864,
 92865,
 92880,
 92883,
 92904,
 92905,
 92907,
 92917,
 92949,
 92955,
 92956,
 92959,
 92960,
 92961,
 92971,
 92972,
 92986,
 93012,
 93016,
 93031,
 93083,
 93126,
 93128,
 93132,
 93155,
 93157,
 93187]
