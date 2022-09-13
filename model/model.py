from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import backend as K

conv_layer = partial(layers.Conv2D,
                     padding='same',
                     kernel_initializer='he_normal',
                     bias_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     bias_regularizer=keras.regularizers.l2(0.001)
                     )


def decoder_block(input_tensor, concat_tensor=None, nFilters=512, nConvs=2, i=0, rate=0.2,
                  name_prefix='decoder_block', noise=1, activation='relu',  **kwargs):
    deconv = input_tensor
   # for j in range(n_convs):
        #deconv = conv_layer(n_filters, (3, 3), name=f'{name_prefix}{i}_deconv{j + 1}')(deconv)
        #deconv = layers.BatchNormalization(name=f'{name_prefix}{i}_batchnorm{j + 1}')(deconv)
       # deconv = layers.Activation(activation, name=f'{name_prefix}{i}_activation{j + 1}')(deconv)
      #  deconv = layers.GaussianNoise(stddev=noise, name=f'{name_prefix}{i}_noise{j + 1}')(deconv)

    for j in range(nConvs):
        deconv = layers.Conv2D(nFilters, 3, activation='relu',padding='same',name=f"{name_prefix}{i}_deconv{j+1}")(deconv)
        deconv = layers.BatchNormalization(name=f"{name_prefix}{i}_batchnorm{j+1}")(deconv)
        if j == 0:
            if concat_tensor is not None:
                deconv = layers.concatenate([deconv,concat_tensor],name=f"{name_prefix}{i}_concat")
            deconv = layers.Dropout(0.2, seed=0+i,name=f"{name_prefix}{i}_dropout")(deconv)
       # if j == 0 and concat_tensor is not None:
            #deconv = layers.Dropout(rate=rate, name=f'{name_prefix}{i}_dropout')(deconv)
            # if combo == 'add':
            #    deconv = layers.add([deconv, concat_tensor], name=f'{name_prefix}{i}_residual')
            #elif combo == 'concat':
            #deconv = layers.concatenate([deconv, concat_tensor], name=f'{name_prefix}{i}_concat')

    up = layers.UpSampling2D(interpolation='bilinear', name=f'{name_prefix}{i}_upsamp')(deconv)
    return up



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    true_sum = K.sum(K.square(y_true), -1)
    pred_sum = K.sum(K.square(y_pred), -1)
    return 1 - ((2. * intersection + smooth) / (true_sum + pred_sum + smooth))


def addFeatures(input_tensor):

    def normalizedDifference(c1, c2, name="nd"):
        return layers.Lambda(lambda x: ((x[0] - x[1]) / (x[0] + x[1] + 1e-7)), name=name)([c1, c2])

    def whiteness(r, g, b):
        mean = layers.average([r, g, b], name="vis_mean")
        rx = layers.Lambda(lambda x: K.abs(
            (x[0] - x[1]) / (x[1] + 1e-7)), name="r_centered")([r, mean])
        gx = layers.Lambda(lambda x: K.abs(
            (x[0] - x[1]) / (x[1] + 1e-7)), name="g_centered")([g, mean])
        bx = layers.Lambda(lambda x: K.abs(
            (x[0] - x[1]) / (x[1] + 1e-7)), name="b_centered")([b, mean])
        return layers.add([rx, gx, bx], name="whiteness")

    def addEVI(nir,red,blue):
        #eturn layers.Lambda(lambda x: (2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))), name="evi")([nir,red,blue])
        return layers.Lambda(lambda x: (2.5 * ((x[0]- x[1]) / (x[0] + 6 * x[1] - 7.5 * x[2] + 1))), name="evi")([nir,red,blue])


    def addSAVI(nir,red):
        #return layers.Lambda(lambda x: (nir -red) * (1 + 0.5)/(nir + red + 0.5), name="savi")([nir,red])
        return layers.Lambda(lambda x: (x[0] -x[1]) * (1 + 0.5)/(x[0] + x[1] + 0.5), name="savi")([nir,red])

    def addIBI(nir,swir1,green,red):
        #ibiA = layers.Lambda(lambda x: 2 * swir1 / (swir1 + nir), name="ibia")([swir1,nir])
        #ibiB = layers.Lambda(lambda x: (nir / (nir +  red)) + (green / (green + swir1)), name="ibib")([nir,red,green,swir1])
        ibiA = layers.Lambda(lambda x: 2 * x[0] / (x[0] + x[1]), name="ibia")([swir1,nir])
        ibiB = layers.Lambda(lambda x: (x[0] / (x[0] +  x[1])) + (x[2] / (x[2] + x[3])), name="ibib")([nir,red,green,swir1])


        ibi = normalizedDifference(ibiA,ibiB, name="IBI")
        return ibi


    blue = input_tensor[:, :, :, 0:1]
    green = input_tensor[:, :, :, 1:2]
    red = input_tensor[:, :, :, 2:3]
    nir = input_tensor[:, :, :, 3:4]
    swir1 = input_tensor[:, :, :, 8:9]
    swir2 = input_tensor[:, :, :, 9:10]


    ndvi = normalizedDifference(nir,red, name="ndvi")
    evi = addEVI(nir,red,blue)
    savi = addSAVI(nir,red)
    ibi = addIBI(nir,swir1,green,red)
    mndwi = normalizedDifference(green, swir2, name="mndwi")
    ndbi = normalizedDifference(swir1,nir,name="ndbi")
    white = whiteness(red,green,blue)

    ND_blue_green = normalizedDifference(blue,green, name="ND_blue_green")
    ND_blue_red = normalizedDifference(blue,red, name="ND_blue_red")
    ND_blue_nir = normalizedDifference(blue,nir, name="ND_blue_nir")
    ND_blue_swir1 = normalizedDifference(blue,swir1, name="ND_blue_swir1")
    ND_blue_swir2 = normalizedDifference(blue,swir2, name="ND_blue_swir2")

    ND_green_blue = normalizedDifference(green, blue, name="ND_green_blue")
    ND_green_red = normalizedDifference(green, red, name="ND_green_red")
    ND_green_nir = normalizedDifference(green, nir, name="ND_green_nir")
    ND_green_swir1 = normalizedDifference(green,swir1, name="ND_green_swir1")
    ND_green_swir2 = normalizedDifference(green,swir2, name="ND_green_swir2")

    ND_red_blue = normalizedDifference(red,blue, name="ND_red_blue")
    ND_red_green = normalizedDifference(red,green, name="ND_red_green")
    ND_red_nir = normalizedDifference(red,nir, name="ND_red_nir")
    ND_red_swir1 = normalizedDifference(red,swir1, name="ND_red_swir1")
    ND_red_swir2 = normalizedDifference(red,swir2, name="ND_red_swir2")

    ND_nir_blue = normalizedDifference(nir,blue, name="ND_nir_blue")
    ND_nir_green = normalizedDifference(nir,green, name="ND_nir_green")
    ND_nir_red = normalizedDifference(nir,red, name="ND_nir_red")
    ND_nir_swir1 = normalizedDifference(nir,swir1, name="ND_nir_swir1")
    ND_nir_swir2 = normalizedDifference(nir,swir2, name="ND_nir_swir2")

    ND_swir1_blue = normalizedDifference(swir1,blue, name="ND_swir1_blue")
    ND_swir1_green = normalizedDifference(swir1,green, name="ND_swir1_green")
    ND_swir1_red = normalizedDifference(swir1,red, name="ND_swir1_red")
    ND_swir1_nir = normalizedDifference(swir1,nir, name="ND_swir1_nir")
    ND_swir1_swir2 = normalizedDifference(swir1,swir2, name="ND_swir1_swir2")

    ND_swir2_blue = normalizedDifference(swir2,blue, name="ND_swir2_blue")
    ND_swir2_green = normalizedDifference(swir2,green, name="ND_swir2_green")
    ND_swir2_red = normalizedDifference(swir2,red, name="ND_swir2_red")
    ND_swir2_nir = normalizedDifference(swir2,nir, name="ND_swir2_nir")
    ND_swir2_swir1 = normalizedDifference(swir2,swir1, name="ND_swir2_swir1")

    #return layers.concatenate([input_tensor,ndvi], name='inputs')
    return layers.concatenate([input_tensor,savi, ndvi, mndwi, ndbi, white,ND_blue_green,ND_blue_red,ND_blue_nir,ND_blue_swir1,ND_blue_swir2,ND_green_blue,ND_green_red,ND_green_nir,ND_green_swir1,ND_green_swir2]) #,ND_red_blue,ND_red_green,ND_red_nir,ND_red_swir1,ND_red_swir2,ND_nir_blue,ND_nir_green,ND_nir_red,ND_nir_swir1,ND_nir_swir2,ND_swir1_blue,ND_swir1_green,ND_swir1_red,ND_swir1_nir,ND_swir1_swir2,ND_swir2_blue,ND_swir2_green,ND_swir2_red,ND_swir2_nir,ND_swir2_swir1], name='inputs')




def get_model(in_shape, out_classes, dropout_rate=0.2, noise=1,
              activation='relu', combo='add', **kwargs):

    #out_classes = 8
    #in_tensor = layers.Input(shape=[None,None,14],name="input")
    in_tensor = layers.Input(shape=in_shape, name='input')
    in_tensor = addFeatures(in_tensor)

    vgg19 = keras.applications.VGG19(include_top=False, weights=None, input_tensor=in_tensor)

    base_in = vgg19.input
    base_out = vgg19.output
    concat_layers = ['block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2']
    concat_tensors = [vgg19.get_layer(layer).output for layer in concat_layers]

    decoder0 = decoder_block(
        base_out, nFilters=512, nConvs=3, noise=noise,
        i=0, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )  # 64
    decoder1 = decoder_block(
        decoder0, concat_tensor=concat_tensors[0], nFilters=512, nConvs=3, noise=noise,
        i=1, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )
    decoder2 = decoder_block(
        decoder1, concat_tensor=concat_tensors[1], nFilters=256, nConvs=3, noise=noise,
        i=2, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )
    decoder3 = decoder_block(
        decoder2, concat_tensor=concat_tensors[2], nFilters=128, nConvs=2, noise=noise,
        i=3, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )
    decoder4 = decoder_block(
        decoder3, concat_tensor=concat_tensors[3], nFilters=64, nConvs=2, noise=noise,
        i=4, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )

    outBranch = layers.concatenate([decoder4,concat_tensors[4]],name="out_block_concat1")
    outBranch = layers.SpatialDropout2D(rate=0.2,seed=0,name="out_block_spatialdrop")(outBranch)

    # perform some additional convolutions before predicting probabilites
    outBranch = layers.Conv2D(64, 3, activation='relu',padding='same',name="out_block_conv1")(outBranch)
    outBranch = layers.BatchNormalization(name="out_block_batchnorm1")(outBranch)
    outBranch = layers.Conv2D(64, 3, activation='relu',padding='same',name="out_block_conv2")(outBranch)
    outBranch = layers.BatchNormalization(name="out_block_batchnorm2")(outBranch)
    # final convolution and softmax activation to get output probabilities
    # nodes will equal the number of classes
    outBranch = layers.Conv2D(2, (1, 1),name='final_conv')(outBranch)
    output = layers.Activation("softmax",name="final_out")(outBranch)
    
    model = models.Model(inputs=[base_in], outputs=[output], name='vgg19-unet')
    return model


def build(*args, optimizer=None, loss=None, metrics=None, distributed_strategy=None, **kwargs):
    learning_rate = kwargs.get('learning_rate', 0.001)
    if optimizer == 'sgd_momentum':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        #optimizer = keras.optimizers.Adam()

    if loss is None:
        #loss = dice_loss
        loss = keras.losses.binary_crossentropy
    
    if metrics is None:
        metrics = [
            keras.metrics.categorical_accuracy,
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            dice_coef,
            f1_m
        ]

    if distributed_strategy is not None:
        with distributed_strategy.scope():
            model = get_model(*args, **kwargs)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        model = get_model(*args, **kwargs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
