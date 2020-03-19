
from keras.optimizers import SGD
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Activation
from keras.layers import Add, Subtract, Multiply, merge, Concatenate, Dropout, Conv2DTranspose
from keras.layers import BatchNormalization, SpatialDropout2D,Flatten,Dense
from keras.models import Model
from keras import backend as K



def CONV2D(x, filter_num, kernel_size, activation='relu', **kwargs):
    x = Conv2D(filter_num, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    if activation=='relu': 
        x = Activation('relu', **kwargs)(x)
    elif activation=='sigmoid': 
        x = Activation('sigmoid', **kwargs)(x)
    else:
        x = Activation('softmax', **kwargs)(x)
    return x




def U_Net(shape, classes=1):
    inputs = Input(shape)
    conv0 = BatchNormalization()(inputs)

    conv0 = CONV2D(conv0, 32, (3, 3))
    conv1 = CONV2D(conv0, 32, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv0 = CONV2D(pool1, 64, (3, 3))
    conv2 = CONV2D(conv0, 64, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv0 = CONV2D(pool1, 128, (3, 3))
    conv3 = CONV2D(conv0, 128, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv0 = CONV2D(pool1, 256, (3, 3))
    conv4 = CONV2D(conv0, 256, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv0 = CONV2D(pool1, 512, (3, 3))
    conv5 = CONV2D(conv0, 512, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool5")(conv5)

    conv0 = CONV2D(pool1, 1024, (3, 3))
    conv0 = CONV2D(conv0, 1024, (3, 3))

    pool1 = merge([UpSampling2D(size=(2, 2))(conv0), conv5], mode='concat', concat_axis=3)
    conv0 = CONV2D(pool1, 512, (3, 3))
    conv0 = CONV2D(conv0, 512, (3, 3))

    pool1 = merge([UpSampling2D(size=(2, 2))(conv0), conv4], mode='concat', concat_axis=3)
    conv0 = CONV2D(pool1, 256, (3, 3))
    conv0 = CONV2D(conv0, 256, (3, 3))

    pool1 = merge([UpSampling2D(size=(2, 2))(conv0), conv3], mode='concat', concat_axis=3)
    conv0 = CONV2D(pool1, 128, (3, 3))
    conv0 = CONV2D(conv0, 128, (3, 3))

    pool1 = merge([UpSampling2D(size=(2, 2))(conv0), conv2], mode='concat', concat_axis=3)
    conv0 = CONV2D(pool1, 64, (3, 3))
    conv0 = CONV2D(conv0, 64, (3, 3))

    pool1 = merge([UpSampling2D(size=(2, 2))(conv0), conv1], mode='concat', concat_axis=3)
    conv0 = CONV2D(pool1, 32, (3, 3))
    conv0 = CONV2D(conv0, 32, (3, 3))

    conv0 = CONV2D(conv0, classes, (1, 1), activation='sigmoid')
    model = Model(input=inputs, output=conv0)
    model.summary() 
    return model

# the proposed network
def Our_Net(shape, classes=1):
    inputs = Input(shape)
    conv0 = BatchNormalization()(inputs)

    conv1 = CONV2D(conv0, 32, (3, 3))
    conv1 = CONV2D(conv1, 32, (3, 3))
    pool1a = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1b = MaxPooling2D(pool_size=(4, 4))(conv1)
    
    conv2 = CONV2D(pool1a,64, (3, 3))
    conv2 = CONV2D(conv2, 64, (3, 3))
    pool2a = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2b = MaxPooling2D(pool_size=(4, 4))(conv2)

    merg1 = merge([pool1b, pool2a], mode='concat', concat_axis=3)
    conv3 = CONV2D(merg1, 128, (3, 3))
    conv3 = CONV2D(conv3, 128, (3, 3))
    pool3a = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3b = MaxPooling2D(pool_size=(4, 4))(conv3)

    merg2 = merge([pool2b, pool3a], mode='concat', concat_axis=3)
    conv4 = CONV2D(merg2, 256, (3, 3))
    conv4 = CONV2D(conv4, 256, (3, 3))
    pool4a = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4b = MaxPooling2D(pool_size=(4, 4))(conv4)

    merg3 = merge([pool3b, pool4a], mode='concat', concat_axis=3)
    conv5 = CONV2D(merg3, 512, (3, 3))
    conv5 = CONV2D(conv5, 512, (3, 3))
    pool5a = MaxPooling2D(pool_size=(2, 2))(conv5)

    merg4 = merge([pool4b, pool5a], mode='concat', concat_axis=3)
    conv6 = CONV2D(merg4, 512, (3, 3))
    conv6 = CONV2D(conv6, 512, (3, 3))

    pool3b = CONV2D(pool3b, 512, (3, 3))
    up1 = UpSampling2D(size=(2, 2))(conv6)
    merg5 = merge([up1, conv5, pool3b, Subtract()([up1, conv5]), Subtract()([up1, pool3b])], mode='concat', concat_axis=3)
    conv7 = CONV2D(merg5, 256, (3, 3))
    conv7 = CONV2D(conv7, 256, (3, 3))

    pool2b = CONV2D(pool2b, 256, (3, 3))
    up2 = UpSampling2D(size=(2, 2))(conv7)
    merg6 = merge([up2, conv4, pool2b, Subtract()([up2, conv4]), Subtract()([up2, pool2b])], mode='concat', concat_axis=3)
    conv8 = CONV2D(merg6, 128, (3, 3))
    conv8 = CONV2D(conv8, 128, (3, 3))

    pool1b = CONV2D(pool1b, 128, (3, 3))
    up3 = UpSampling2D(size=(2, 2))(conv8)
    merg7 = merge([up3, conv3, pool1b, Subtract()([up3, conv3]), Subtract()([up3, pool1b])], mode='concat', concat_axis=3)
    conv9 = CONV2D(merg7, 64, (3, 3))
    conv9 = CONV2D(conv9, 64, (3, 3))

    up4 = UpSampling2D(size=(2, 2))(conv9)
    merg8 = merge([up4, conv2, Subtract()([up4, conv2])], mode='concat', concat_axis=3)
    conv10 = CONV2D(merg8,  32, (3, 3))
    conv10 = CONV2D(conv10, 32, (3, 3))

    up5 = UpSampling2D(size=(2, 2))(conv10)
    merg9 = merge([up5, conv1, Subtract()([up5, conv1])], mode='concat', concat_axis=3)
    conv11 = CONV2D(merg9,  32, (3, 3))
    conv11 = CONV2D(conv11, 32, (3, 3))

    conv11 = CONV2D(conv11, classes, (1, 1), activation='sigmoid')

    model = Model(input=inputs, output=conv11)
    model.summary() 
    return model



