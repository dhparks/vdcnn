# for now, this implementation skips the resnet-style short-cuts
# TODO: implement shortcuts for assisting very-deep networks (49+)

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling1D
from keras.models import Model
import tensorflow as tf

# Conneau table 2
# depth 09: layer_spec = [(64, 2),  (128, 2),  (256, 2),  (512, 2)]
# depth 17: layer_spec = [(64, 4),  (128, 4),  (256, 4),  (512, 4)]
# depth 29: layer_spec = [(64, 10), (128, 10), (256, 4),  (512, 4)]
# depth 49: layer_spec = [(64, 16), (128, 16), (256, 10), (512, 6)]

# =============================================================================
#   Helper functions for CharacterCNN
# =============================================================================

def _top_k(tensor, k):
    tensor = tf.transpose(tensor, [0, 2, 1])
    top = tf.nn.top_k(tensor, k=k)[0]
    return tf.transpose(top, [0, 2, 1])


def _make_kmax(tensor, k):
    return


def _conv_block(inputs, filters, kernel_size=3):
    # Conneau Fig 2 (half! figure contains two blocks)
    out = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(inputs)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def _pool(inputs, mode='max', sorted=True):
    if mode not in ('max', 'kmax', 'conv', None):
        raise ValueError('unknown pool_type %s' % pool_type)

    if mode == 'conv':
        # resnet style
        # TODO: need nf
        nf = inputs._keras_shape[-1]
        out = Conv1D(filters=nf, kernel_size=3, strides=2, padding='same')(inputs)
        return BatchNormalization()(out)

    if mode == 'kmax':
        # conneau cites [Kalchbrenner et al., 2014]
        k = int(inputs._keras_shape[1] / 2)
        return _make_kmax(inputs, k)(inputs)

    if mode == 'max':
        # vgg style
        return MaxPooling1D(pool_size=3, strides=2, padding='same')(inputs)

    if mode is None:
        # do nothing
        return inputs


def _build_vdcnn(num_classes, sequence_length, embedding_dim=16,
                 conv_spec=None, pool_type='max', dense_spec=None, kmax=8):

    """
    Instantiate the CharacterCNN. Return keras object.
    TODO: incorporate into a CharacterCNN class

    Parameters:
    -------------------------------------------------------------------------

    num_classes: int
        The number of classes in the response

    sequence_length : int
        The length of the input text, in number of characters. It is expected
        that upstream data manipulation will ensure that sequences shorter
        than sequence_length will have been padded, and that sequences longer
        than sequence_length will have been truncated (or similar).

    embedding_dim : int, optional (default 16)
        Character embedding dimension. Default of 16 taken from Conneau.

    conv_spec : list of 2-tuples, optional (default None)
        Allows the modeler to control the number of convolutional blocks
        in the network. Format is [(filters, nblocks), ...]. Each convolutional
        block is given by the following sequence of layers:

            Conv1D(filters, kernel_size, strides, padding='same')
            BatchNormalization()
            Activation('relu')

        If not supplied, the default value of conv_spec is:
            [(64, 2), (128, 2), (256, 2), (512, 2)]

    pool_type : one of ('max', 'kmax', 'conv', None), optional (default 'max')
        Specifies the type of pooling used to decrease size of the temporal
        dimension. If 'max', we use a standard max pooling layer in VGG style.
        If 'kmax', we use the k-max operation described by Conneau.
        If 'conv', we use a strided convolution in ResNet style.
        If None, we do no pooling.

    dense_spec : list of 2-tuples, optional, (default None)
        Specifies the fully connected layers. The format of the list of
        2-tuples is [(size, dropout_rate), ...]. If not supplied, the
        default value of dense_spec is [(1024, 0.1)].

    kmax: int, optional (default 8)
        This is the number of elements to take from the final convolutional
        layer before doing the dense layers. These are understood as the
        most important "topics" or something like that. There is a temptation
        to make this larger but

    """

    # format of conv_spec is [(nf, n_conv_blocks), ...]
    # each time, nf should double
    # default is conneau's depth-9 spec
    if conv_spec is None:
        conv_spec = [(64, 2), (128, 2), (256, 2), (512, 2)]

    if dense_spec is None:
        dense_spec = [(1024, 0.25)]

    # _check_vdcnn(numclasses, sequence_length, embedding_dim, conv_spec, pool_type, dense_spec)

    # embedding and first convolutional layer are common to all networks
    # Embedding requires encoded (char --> int) values
    inputs = Input(shape=(sequence_length,), name='inputs')
    embed = Embedding(input_dim=sequence_length, output_dim=embedding_dim)(inputs)
    out = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name='1st_conv')(embed)

    # add conv blocks and pooling layers
    for idx, (nf, nblocks) in enumerate(conv_spec):

        for _ in range(nblocks):
            out = _conv_block(out, filters=nf, kernel_size=3)

        if idx < len(conv_spec) - 1:
            out = _pool(out, pool_type)

    # kmax
    out = Lambda(lambda x: _top_k(x, kmax))(out)
    out = Flatten()(out)

    for size, dropout_rate in dense_spec:
        out = Dense(size, activation='relu')(out)
        if dropout_rate > 0:
            out = Dropout(dropout_rate)(out)

    # final activation for class prediction
    out = Dense(num_classes, activation='softmax')(out)

    # optimizer
    # TODO: allow user spec

    # Create model
    model = Model(inputs=inputs, outputs=out)
    return model


def _check_vdcnn(numclasses, sequence_length, embedding_dim, conv_spec,
           pool_type, dense_spec):
    if not isinstance(numclasses, int):
        raise TypeError('numclasses should be int, got %s' % type(numclasses))

    if numclasses < 2:
        raise ValueError('need to have at least two classes')

    if not isinstance(sequence_length, int):
        raise TypeError('sequence_length should be int, got %s' % type(sequence_length))

    if sequence_length < 1:
        raise ValueError('sequence_length of %s makes no sense!' % sequence_length)

    if not isinstance(embedding_dim, int):
        raise TypeError('embedding_dim should be int, got %s' % type(embedding_dim))

    if embedding_dim < 1:
        raise ValueError('embedding_dim of %s makes no sense!' % embedding_dim)

    if not isinstance(conv_spec, list):
        raise TypeError('conv_spec must be list, got %s' % type(conv_spec))

    if len(conv_spec) < 1:
        raise ValueError('conv_spec must contain at least one element')

    for idx, item in enumerate(conv_spec):

        if not isinstance(item, (list, tuple)):
            raise TypeError('conv_spec[%s] must be list or tuple, got %s' % (idx, type(item)))

        if len(item) != 2:
            raise ValueError('conv_spec items must be length-2; item %s is length %s' % (idx, len(item)))

        if not isinstance(item[0], int):
            raise TypeError

        if not isinstance(item[1], int):
            raise TypeError

        if item[0] < 1:
            raise ValueError

        if item[0] < 1:
            raise ValueError

    for idx, item in enumerate(dense_spec):

        if not isinstance(item, (list, tuple)):
            raise TypeError('dense_spec[%s] must be list or tuple, got %s' % (idx, type(item)))

        if len(item) != 2:
            raise ValueError('dense_spec items must be length-2; item %s is length %s' % (idx, len(item)))

        if not isinstance(item[0], int):
            raise TypeError

        if not isinstance(item[1], int):
            raise TypeError

        if item[0] < 1:
            raise ValueError

        if item[0] < 1:
            raise ValueError

