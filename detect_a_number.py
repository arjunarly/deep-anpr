#!/usr/bin/env python
# encoding=utf-8
# Created by andy on 2016-07-19 21:51.
import cv2
import sys

import numpy

import common
import model
import tensorflow as tf

__author__ = "andy"


def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


def detect(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.

    """

    x, y, params = model.get_detect_model()
    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        feed_dict = {x: [im]}
        feed_dict.update(dict(zip(params, param_vals)))
        predict_values = sess.run(y, feed_dict=feed_dict)
        letter_probs = predict_values.reshape(common.LENGTH, len(common.DIGITS))
        letter_probs = common.softmax(letter_probs)
        code= letter_probs_to_code(letter_probs)
    return code


if __name__ == "__main__":
    im = cv2.imread("00000004_4265936_1.png")
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

    f = numpy.load('weights_7_2048.npz')
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    code = detect(im_gray, param_vals)

    print code
