#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines for training the network.

"""

__all__ = (
    'train',
)

import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

import common
import gen
import model


def code_to_vec(code):
    def char_to_vec(c):
        y = numpy.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    c = numpy.vstack([char_to_vec(c) for c in code])
    # print "c.shape():{}".format(c.shape)
    return numpy.concatenate([c.flatten()])


def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = fname.split("/")[1].split("_")[1]
        # print fname.split("/")[1].split("_")[2].replace(".png", "")
        p = fname.split("/")[1].split("_")[2].replace(".png", "") == '1'

        yield im, code_to_vec(code)


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out


def mpgen(f):
    def main(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(10)
        proc = multiprocessing.Process(target=main,
                                       args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()

    return wrapped


@mpgen
def read_batches(batch_size):
    def gen_vecs():
        for im, c, p in gen.generate_ims(batch_size):
            yield im, code_to_vec(c)

    while True:
        yield unzip(gen_vecs())


def train(report_steps, batch_size, initial_weights=None):
    """
    Train the network.

    The function operates interactively: Progress is reported on stdout, and
    training ceases upon `KeyboardInterrupt` at which point the learned weights
    are saved to `weights.npz`, and also returned.

    :param learn_rate:
        Learning rate to use.

    :param report_steps:
        Every `report_steps` batches a progress report is printed.

    :param batch_size:
        The size of the batches used for training.

    :param initial_weights:
        (Optional.) Weights to initialize the network with.

    :return:
        The learned network weights.

    """
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(common.INITIAL_LEARNING_RATE,
                                               global_step,
                                               common.DECAY_STEPS,
                                               common.LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    # y is the predicted value
    x, y, params = model.get_training_model()

    y_ = tf.placeholder(tf.float32, [None, common.LENGTH * len(common.CHARS)])

    digits_loss = tf.nn.softmax_cross_entropy_with_logits(tf.reshape(y, [-1, len(common.CHARS)]),
                                                          tf.reshape(y_, [-1, len(common.CHARS)]))
    cross_entropy = tf.reduce_sum(digits_loss)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    predict = tf.argmax(tf.reshape(y, [-1, common.LENGTH, len(common.CHARS)]), 2)

    real_value = tf.argmax(tf.reshape(y_, [-1, common.LENGTH, len(common.CHARS)]), 2)

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.initialize_all_variables()

    def vec_to_plate(v):
        return "".join(common.CHARS[i] for i in v)

    def do_report():
        r = sess.run([predict, real_value, cross_entropy, learning_rate], feed_dict={x: test_xs, y_: test_ys})
        num_correct = numpy.sum(numpy.all(r[0] == r[1], axis=1))
        r_short = (r[0][:common.TEST_SIZE], r[1][:common.TEST_SIZE])
        print "{} <--> {} ".format("real_value", "predict_value")
        for pred, real in zip(*r_short):
            print "{} <--> {} ".format(vec_to_plate(real), vec_to_plate(pred))

        print ("batch:{:3d}, hit_rate:{:2.02f}%,cross_entropy:{}, learning_rate:{} ").format(batch_idx,
                                                                                             100. * num_correct / (
                                                                                                 len(r[0])),
                                                                                             r[2], r[3])

    def do_batch():
        sess.run(train_step,
                 feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % report_steps == 0:
            do_report()

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data("test/*.png"))[:common.TEST_SIZE])
        # print "test_xs.shape:{}".format(test_xs.shape)

        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                # print "batch_ys.shape():{}".format(batch_ys.shape)
                do_batch()
                if batch_idx % report_steps == 0:
                    batch_time = time.time()
                    if last_batch_idx != batch_idx:
                        print "time for 60 batches {}s".format(
                            60 * (last_batch_time - batch_time) /
                            (last_batch_idx - batch_idx))
                        last_batch_idx = batch_idx
                        last_batch_time = batch_time

        except KeyboardInterrupt:

            last_weights = [p.eval() for p in params]
            numpy.savez("weights.npz", *last_weights)
            return last_weights


if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = numpy.load(sys.argv[1])
        initial_weights = [f[n] for n in sorted(f.files,
                                                key=lambda s: int(s[4:]))]
    else:
        initial_weights = None

    train(report_steps=60,
          batch_size=64,
          initial_weights=initial_weights)
