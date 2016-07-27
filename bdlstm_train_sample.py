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

# Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 300

# Network Parameters
nHidden = 1024
nClasses = 11  # 11 characters[0,9], plus the "blank" for CTC

report_steps = 50

# Load data
print('Loading data')
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
    inputX = tf.placeholder(tf.float32, shape=(maxSteps, common.BATCH_SIZE, nFeatures))  # maxSteps*batchSize*nFeatures
    # Prep input data to fit requirements of rnn.bidirectional_rnn
    # Reshape to 2-D tensor (nTimeSteps*batchSize, nFeatures)
    inputXrs = tf.reshape(inputX, [-1, nFeatures])  # maxTimeSteps*batchSize,nFeatures
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    inputList = tf.split(0, maxSteps, inputXrs)  # [(batchSize,nFeatures), (batchSize,nFeatures),...]
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(common.BATCH_SIZE))  #

    # Weights & biases
    weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden], stddev=np.sqrt(2.0 / (2 * nHidden))))
    biasesOutH1 = tf.Variable(tf.zeros([nHidden]))

    weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden], stddev=np.sqrt(2.0 / (2 * nHidden))))
    biasesOutH2 = tf.Variable(tf.zeros([nHidden]))

    weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses], stddev=np.sqrt(2.0 / nHidden)))
    biasesClasses = tf.Variable(tf.zeros([nClasses]))

    # Network
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)

    fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32, scope='BDLSTM_H1')
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
    test_input, test_codes = unzip(list(train.read_data_for_lstm_ctc("test/*.png"))[:common.BATCH_SIZE])
    test_input = test_input.swapaxes(0, 1).swapaxes(0, 2)

    test_batchInputs, test_batchTargetSparse, test_batchSeqLengths = convert_code_to_spare_tensor(test_input,
                                                                                                  test_codes)
    test_batchTargetIxs, test_batchTargetVals, test_batchTargetShape = test_batchTargetSparse


    def do_report():
        fDict = {inputX: test_batchInputs, targetIxs: test_batchTargetIxs, targetVals: test_batchTargetVals,
                 targetShape: test_batchTargetShape, seqLengths: test_batchSeqLengths}
        l, pred, errR, steps, lr_rate, lmt = session.run(
            [loss, predictions, errorRate, global_step, learning_rate, logitsMaxTest],
            feed_dict=fDict)
        print("step:", steps, "errorRate:", errorRate, "loss:", l, "lr_rate:", lr_rate, "lmt:", np.unique(lmt))


    batch_iter = enumerate(read_batches_for_bdlstm_ctc(common.BATCH_SIZE))
    train_list = list()
    for i, (ims, codes) in batch_iter:
        train_list.append((ims, codes))
        if i > 100:
            break

    for epoch in range(nEpochs):
        print('Epoch', epoch + 1, '...')
        # batchErrors = np.zeros(len(batchedData))
        # batchRandIxs = np.random.permutation(len(batchedData))  # randomize batch order

        last_batch_idx = 0
        last_batch_time = time.time()

        for batch_idx, (ims, codes) in enumerate(train_list):
            batchInputs, batchTargetSparse, batchSeqLengths = convert_code_to_spare_tensor(ims, codes)
            batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            # print(batchTargetVals)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
            # print(np.unique(lmt))  # print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
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
