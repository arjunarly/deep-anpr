'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict character sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is test code to run on the
8-item data set in the "sample_data" directory, for those without access to TIMIT.

Author: Jon Rein
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import cv2
import tensorflow as tf
import time
from tensorflow.contrib.ctc import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np

import common
import gen
import train
from train import read_batches, read_batches_for_bdlstm_ctc, unzip
from utils import load_batched_data, target_list_to_sparse_tensor, convert_code_to_spare_tensor

INPUT_PATH = './sample_data/mfcc/'  # directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = './sample_data/char_y/'  # directory of nCharacters 1-D array .npy files

# Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 300

# Network Parameters
nHidden = 128
nClasses = 11  # 11 characters[0,9], plus the "blank" for CTC

report_steps = 100

# Load data
print('Loading data')
# maxTimeStep
# batchedData, maxTimeSteps, totalN = load_batched_data(INPUT_PATH, TARGET_PATH, batchSize)
maxSteps = gen.OUTPUT_SHAPE[1]
nFeatures = gen.OUTPUT_SHAPE[0]
# Define graph
print('Defining graph')

graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(common.INITIAL_LEARNING_RATE,
                                               global_step,
                                               common.DECAY_STEPS,
                                               common.LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    # NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow

    # Graph input
    inputX = tf.placeholder(tf.float32, shape=(maxSteps, common.BATCH_SIZE, nFeatures))
    # Prep input data to fit requirements of rnn.bidirectional_rnn
    # Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
    inputXrs = tf.reshape(inputX, [-1, nFeatures])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    inputList = tf.split(0, maxSteps, inputXrs)
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(common.BATCH_SIZE))

    # Weights & biases
    weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2 * nHidden))))
    biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
    weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2 * nHidden))))
    biasesOutH2 = tf.Variable(tf.zeros([nHidden]))
    weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                     stddev=np.sqrt(2.0 / nHidden)))
    biasesClasses = tf.Variable(tf.zeros([nClasses]))

    # Network
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32,
                                   scope='BDLSTM_H1')
    fbH1rs = [tf.reshape(t, [common.BATCH_SIZE, 2, nHidden]) for t in fbH1]
    outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

    # Optimizing
    logits3d = tf.pack(logits)
    loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)

    # Evaluating
    logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / tf.to_float(
        tf.size(targetY.values))

# Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    test_input, test_codes = unzip(list(train.read_data_for_lstm_ctc("test/*.png"))[:common.TEST_SIZE])
    test_batchInputs, test_batchTargetSparse, test_batchSeqLengths = convert_code_to_spare_tensor(test_input,
                                                                                                  test_codes)
    test_batchTargetIxs, test_batchTargetVals, test_batchTargetShape = test_batchTargetSparse


    def do_report():
        feedDict = {inputX: test_batchInputs, targetIxs: test_batchTargetIxs, targetVals: test_batchTargetVals,
                    targetShape: test_batchTargetShape, seqLengths: test_batchSeqLengths}
        l, pred, errR, steps, lr_rate = session.run([loss, predictions, errorRate, global_step, learning_rate],
                                                    feed_dict=feedDict)

        # num_correct = numpy.sum(numpy.all(r[0] == r[1], axis=1))
        # r_short = (r[0][:common.TEST_SIZE], r[1][:common.TEST_SIZE])
        # print "{} <--> {} ".format("real_value", "predict_value")

        # for pred, real in zip(*r_short):
        #    print "{} <--> {} ".format(vec_to_plate(real), vec_to_plate(pred))

        #          print ("batch:{:3d}, hit_rate:{:2.02f}%,cross_entropy:{}, learning_rate:{},global_step:{} ").format(batch_idx,
        #                                                                                                             100. * num_correct / (
        #                                                                                                                 len(r[
        #                                                                                                                         0])),
        #                                                                                                              r[2], r[3],
        #                                                                                                              r[4])


    for epoch in range(nEpochs):
        print('Epoch', epoch + 1, '...')
        # batchErrors = np.zeros(len(batchedData))
        # batchRandIxs = np.random.permutation(len(batchedData))  # randomize batch order

        last_batch_idx = 0
        last_batch_time = time.time()
        batch_iter = enumerate(read_batches_for_bdlstm_ctc(common.BATCH_SIZE))
        for batch_idx, (ims, codes) in batch_iter:
            batchInputs, batchTargetSparse, batchSeqLengths = convert_code_to_spare_tensor(ims, codes)
            batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            # print(batchTargetVals)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
            print(np.unique(
                lmt))  # print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
            if batch_idx % report_steps == 0:
                do_report()
                batch_time = time.time()
                if last_batch_idx != batch_idx:
                    "time for 60 batches {}s".format(
                        60 * (last_batch_time - batch_time) /
                        (last_batch_idx - batch_idx))
                    last_batch_idx = batch_idx
                    last_batch_time = batch_time
                    #  if (batch % 1) == 0:
                    #      print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                    #      print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)

                    # batchErrors[batch] = er * len(batchSeqLengths)

        """
        for batch, batchOrigI in enumerate(batchRandIxs):
            batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
            batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            print(batchTargetVals)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
            # print(predictions)
            print(np.unique(
                lmt))  # print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batchErrors[batch] = er * len(batchSeqLengths)
        """
        # epochErrorRate = batchErrors.sum() / totalN
        # print('Epoch', epoch + 1, 'error rate:', epochErrorRate)
