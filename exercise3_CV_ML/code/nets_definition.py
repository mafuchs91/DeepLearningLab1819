from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

import tensorflow.contrib as tc

from layers_slim import *



def FCN_Seg(self, is_training=True):

    #Set training hyper-parameters
    self.is_training = is_training
    self.normalizer = tc.layers.batch_norm
    self.bn_params = {'is_training': self.is_training}


    print("input", self.tgt_image)

    with tf.variable_scope('First_conv'):
        conv1 = tc.layers.conv2d(self.tgt_image, 32, 3, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

        print("Conv1 shape")
        print(conv1.get_shape())

    x = inverted_bottleneck(conv1, 1, 16, 0,self.normalizer, self.bn_params, 1)
    #print("Conv 1")
    #print(x.get_shape())

    #180x180x24
    x = inverted_bottleneck(x, 6, 24, 1,self.normalizer, self.bn_params, 2)
    x = inverted_bottleneck(x, 6, 24, 0,self.normalizer, self.bn_params, 3)

    print("Block One dim ")
    print(x)

    DB2_skip_connection = x
    #90x90x32
    x = inverted_bottleneck(x, 6, 32, 1,self.normalizer, self.bn_params, 4)
    x = inverted_bottleneck(x, 6, 32, 0,self.normalizer, self.bn_params, 5)

    print("Block Two dim ")
    print(x)

    DB3_skip_connection = x
    #45x45x96
    x = inverted_bottleneck(x, 6, 64, 1,self.normalizer, self.bn_params, 6)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 7)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 8)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 9)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 10)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 11)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 12)

    print("Block Three dim ")
    print(x)

    DB4_skip_connection = x
    #23x23x160
    x = inverted_bottleneck(x, 6, 160, 1,self.normalizer, self.bn_params, 13)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 14)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 15)

    print("Block Four dim ")
    print(x)

    #23x23x320
    x = inverted_bottleneck(x, 6, 320, 0,self.normalizer, self.bn_params, 16)

    print("Block Four dim ")
    print(x)


    # Configuration 1 - single upsampling layer
    if self.configuration == 1:

        #input is features named 'x'

        # TODO(1.1) - incorporate a upsample function which takes the features of x
        # and produces 120 output feature maps, which are 16x bigger in resolution than
        # x. Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up5
        current_up5_uncropped = TransitionUp_elu(x,120,16,"decoder_conf1" )

        # crop the image from 304 x 304 to 300 x 300
        # TODO calculate te necessary offsets for cropping, until now ard coded
        current_up5 = crop(current_up5_uncropped, self.tgt_image)


        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    # Configuration 2 - single upsampling layer plus skip connection
    if self.configuration == 2:

        #input is features named 'x'

        # TODO (2.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

        # current_up3_F_up = F_up
        current_up3_F_up = slim.conv2d_transpose(x,160,3,2, scope="decoder_conf2_Fup_for_skip" )
        # crop the image from 304 x 304 to 300 x 300
        print(current_up3_F_up.get_shape())
        # current_up3_F_skip = F_skip
        current_up3_F_skip = DB4_skip_connection
        # crop image
        current_up3_F_up, current_up3_F_skip = crop_image(current_up3_F_up, current_up3_F_skip)

        # add both tensors in var_list
        tensors = []
        tensors.append(current_up3_F_up)
        tensors.append(current_up3_F_skip)
        # concatenate te tensors along the feature dimension
        current_concatenated = tf.concat(tensors, 3, name="concat")
        print(current_concatenated.get_shape())

        current_skipped = slim.conv2d(current_concatenated, 256, [3, 3], scope='convolution')
        print(current_skipped.get_shape())


        # TODO (2.2) - incorporate a upsample function which takes the features from TODO (2.1)
        # and produces 120 output feature maps, which are 8x bigger in resolution than
        # TODO (2.1). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up3
        current_up3_uncropped = slim.conv2d_transpose(current_skipped,120,3,8, scope="upsample_after_skipping")
        # resize tensor
        current_up3, _ = crop_image(current_up3_uncropped, conv1)


        End_maps_decoder1 = slim.conv2d(current_up3, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    # Configuration 3 - Two upsampling layer plus skip connection
    if self.configuration == 3:

        #input is features named 'x'

        # TODO (3.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps
        # current_up3_F_up = F_up


        #--------------------------------------------------------------------------------------------------------------------------
        #FIRST SKIP LAYER


        current_up4_F_up = slim.conv2d_transpose(x,160,3,2, scope="decoder_conf3_Fup_for_skip" )
        # crop the image from 304 x 304 to 300 x 300
        print(current_up4_F_up.get_shape())
        # current_up3_F_skip = F_skip
        current_up4_F_skip = DB4_skip_connection
        # crop image
        current_up4_F_up, current_up4_F_skip = crop_image(current_up4_F_up, current_up4_F_skip)

        # add both tensors in var_list
        tensors = []
        tensors.append(current_up4_F_up)
        tensors.append(current_up4_F_skip)
        # concatenate te tensors along the feature dimension
        current_concatenated = tf.concat(tensors, 3, name="concat")
        print(current_concatenated.get_shape())
        current_skipped = slim.conv2d(current_concatenated, 256, [3, 3], scope='convolution')


#------------------------------------------------------------------------------------------------------------------------------
        # SECOND SKIP LAYER
        # TODO (3.2) - Repeat TODO(3.1) now producing 160 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.

        current_up4_F_up = slim.conv2d_transpose(current_skipped,160,3,2, scope="decoder_conf3_Fup_for_skip2" )
        print(current_up4_F_up.get_shape())
        # current_up3_F_skip = F_skip
        current_up4_F_skip = DB3_skip_connection
        # crop image
        current_up4_F_up, current_up4_F_skip = crop_image(current_up4_F_up, current_up4_F_skip)

        # add both tensors in var_list
        tensors = []
        tensors.append(current_up4_F_up)
        tensors.append(current_up4_F_skip)
        # concatenate te tensors along the feature dimension
        current_concatenated = tf.concat(tensors, 3, name="concat2")
        print(current_concatenated.get_shape())
        current_skipped = slim.conv2d(current_concatenated, 160, [3, 3], scope='convolution2')

        # TODO (3.3) - incorporate a upsample function which takes the features from TODO (3.2)
        # and produces 120 output feature maps which are 4x bigger in resolution than
        # TODO (3.2). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4



        #-----------------------------------------------------------------------------------------------------------------------
        # THIRD SKIP LAYER

        print(current_skipped.get_shape())
        # current_up3
        current_up4_uncropped = slim.conv2d_transpose(current_skipped,120,3,4, scope="upsample_after_skipping3")
        # resize tensor
        current_up4, _ = crop_image(current_up4_uncropped, conv1)



        End_maps_decoder1 = slim.conv2d(current_up4, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    #Full configuration
    if self.configuration == 4:

        ######################################################################################
        ######################################### DECODER Full #############################################


        # TODO (4.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

        # TODO (4.2) - Repeat TODO(4.1) now producing 160 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.

        # TODO (4.3) - Repeat TODO(4.2) now producing 96 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB2_skip_connection) through concatenation.

        # TODO (4.4) - incorporate a upsample function which takes the features from TODO(4.3)
        # and produce 120 output feature maps which are 2x bigger in resolution than
        # TODO(4.3). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4


        #------------------------------------------------------------------------------------------------------------------------------
        #FIRST SKIP LAYER

        current_up4_F_up = slim.conv2d_transpose(x,160,3,2, scope="decoder_conf4_Fup_for_skip" )
        current_up4_F_up = tf.nn.relu(current_up4_F_up)
        current_up4_F_skip = DB4_skip_connection
        current_up4_F_up, current_up4_F_skip = crop_image(current_up4_F_up, current_up4_F_skip)
        tensors = []
        tensors.append(current_up4_F_up)
        tensors.append(current_up4_F_skip)
        current_concatenated = tf.concat(tensors, 3, name="concat")
        print(current_concatenated.get_shape())
        current_skipped = slim.conv2d(current_concatenated, 256, [3, 3], scope='convolution')


        #------------------------------------------------------------------------------------------------------------------------------
        #SECOND SKIP LAYER

        current_up4_F_up = slim.conv2d_transpose(current_skipped,160,3,2, scope="decoder_conf4_Fup_for_skip2" )
        current_up4_F_up = tf.nn.relu(current_up4_F_up)
        current_up4_F_skip = DB3_skip_connection
        current_up4_F_up, current_up4_F_skip = crop_image(current_up4_F_up, current_up4_F_skip)
        tensors = []
        tensors.append(current_up4_F_up)
        tensors.append(current_up4_F_skip)
        current_concatenated = tf.concat(tensors, 3, name="concat2")
        current_skipped = slim.conv2d(current_concatenated, 160, [3, 3], scope='convolution2')


        #------------------------------------------------------------------------------------------------------------------------------
        #THIRD SKIP LAYER

        current_up4_F_up = slim.conv2d_transpose(current_skipped,160,3,2, scope="decoder_conf4_Fup_for_skip3" )
        current_up4_F_up = tf.nn.relu(current_up4_F_up)
        current_up4_F_skip = DB2_skip_connection
        current_up4_F_up, current_up4_F_skip = crop_image(current_up4_F_up, current_up4_F_skip)
        tensors = []
        tensors.append(current_up4_F_up)
        tensors.append(current_up4_F_skip)
        current_concatenated = tf.concat(tensors, 3, name="concat3")
        current_skipped = slim.conv2d(current_concatenated, 96, [3, 3], scope='convolution3')


        #------------------------------------------------------------------------------------------------------------------------------
        #OUTPUT LAYER

        current_up4_uncropped = slim.conv2d_transpose(current_skipped,120,3,2, scope="upsample_after_skipping4")
        current_up5, _ = crop_image(current_up4_uncropped, conv1)



        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    return Reshaped_map


def crop_image(tensor1, tensor2):
    width1 = tensor1.get_shape().as_list()[1]
    width2 = tensor2.get_shape().as_list()[1]
    if (width1 != width2):
        #resize F_up
        print("cropping in progress")
        if (width1 > width2):
            tensor1 = tf.image.crop_to_bounding_box(tensor1, int((width1 - width2)/2),
                                                              int((width1 - width2)/2), width2, width2 )

        # resize F_skip
        else:
            tensor2 = tf.image.crop_to_bounding_box(tensor2,  int((width2 - width1)/2),
                                                         int((width2 - width1)/2), width1, width1 )
    return tensor1, tensor2
