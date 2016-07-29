import numpy as np
import tensorflow as tf

import common
import gen
import train
from train import read_batches_for_bdlstm_ctc, unzip

"""
   targetList list of
"""


def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    # print targetList
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1] + 1]
    return (np.array(indices), np.array(vals), np.array(shape))


def test_edit_distance():
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.edit_distance(hyp, truth, normalize=False)

    with tf.Session(graph=graph) as session:
        truthTest = sparse_tensor_feed([[0, 1, 2], [0, 1, 2, 3, 4]])
        hypTest = sparse_tensor_feed([[3, 4, 5], [0, 1, 2, 2]])
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run([editDist], feed_dict=feedDict)
        print(dist)


"""
    inputList has the shape (steps*batchSize*nFeatures)
"""


def convert_code_to_spare_tensor(inputList, targetList):
    assert inputList.shape[1] == len(targetList)
    # print "inputList.shape", inputList.shape
    batch_size = len(targetList)
    # maxSteps = common.LENGTH
    # print "maxSteps", maxSteps
    batchSeqLengths = np.ones(batch_size) * gen.OUTPUT_SHAPE[1]  # the number of timesteps for each sample in batch

    # print batchSeqLengths
    # print batchSeqLengths
    return (inputList, target_list_to_sparse_tensor(targetList), batchSeqLengths)


def data_lists_to_batches(inputList, targetList, batchSize):
    '''
    Takes a list of input matrices and a list of target arrays and returns
       a list of batches, with each batch being a 3-element tuple of inputs,
       targets, and sequence lengths.
       inputList: list of 2-d numpy arrays with dimensions nFeatures x timesteps
       targetList: list of 1-d arrays or lists of ints
       batchSize: int indicating number of inputs/targets per batch
       returns: dataBatches: list of batch data tuples, where each batch tuple (inputs, targets, seqLengths) consists of
                    inputs = 3-d array w/ shape nTimeSteps x batchSize x nFeatures
                    targets = tuple required as input for SparseTensor
                    seqLengths = 1-d array with int number of timesteps for each sample in batch
                maxSteps: maximum number of time steps across all samples
    '''

    assert len(inputList) == len(targetList)
    print inputList[0]
    nFeatures = inputList[0].shape[0]
    maxSteps = 0
    for inp in inputList:
        maxSteps = max(maxSteps, inp.shape[1])

    # rand Ixs is a  shuffled list of range(len(inputList))
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
        batchSeqLengths = np.zeros(batchSize)
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]
        batchInputs = np.zeros((maxSteps, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
            padSecs = maxSteps - inputList[origI].shape[1]
            batchInputs[:, batchI, :] = np.pad(inputList[origI].T, ((0, padSecs), (0, 0)),
                                               'constant', constant_values=0)
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, target_list_to_sparse_tensor(batchTargetList),
                            batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxSteps)


def get_batched_data(batch_size):
    batch_iter = read_batches_for_bdlstm_ctc(batch_size)
    # for ims, codes in batch_iter:
    #    print im.shape, code


def load_batched_data(specPath, targetPath, batchSize):
    import os
    '''
       returns 3-element tuple: batched data (list), max # of time steps (int), and
       total number of samples (int)
    '''
    return data_lists_to_batches([np.load(os.path.join(specPath, fn)) for fn in os.listdir(specPath)],
                                 [np.load(os.path.join(targetPath, fn)) for fn in os.listdir(targetPath)],
                                 batchSize) + \
           (len(os.listdir(specPath)),)


def decode_a_seq(indexes, spars_tensor):
    str_decoded = ''.join([str(spars_tensor[1][m]) for m in indexes])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    return str_decoded


def decode_sparse_tensor(sparse_tensor):
    print(sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    # print("mmmm", decoded_indexes)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result


if __name__ == '__main__':
    """
    INPUT_PATH = './sample_data/mfcc/'  # directory of MFCC nFeatures x nFrames 2-D array .npy files
    TARGET_PATH = './sample_data/char_y/'  # directory of nCharacters 1-D array .npy files

    spec = [np.load(os.path.join(INPUT_PATH, fn)) for fn in os.listdir(INPUT_PATH)]
    target = [np.load(os.path.join(TARGET_PATH, fn)) for fn in os.listdir(TARGET_PATH)]
    for i in range(5):
        print max(target[i])
    load_batched_data(INPUT_PATH, TARGET_PATH, 4)
    # print spec.shape, target.shape
    print target[1]

    batch_iter = read_batches_for_bdlstm_ctc(10)
    for im, code in batch_iter:
        print code
        break
    """
    test_input, test_code = unzip(list(train.read_data_for_lstm_ctc("test/*.png"))[0:4])
    t = test_input.swapaxes(0, 1).swapaxes(0, 2)
    print test_code
    a, b, c = convert_code_to_spare_tensor(t, test_code)
    dd = target_list_to_sparse_tensor(test_code)

    print(decode_sparse_tensor(dd))
