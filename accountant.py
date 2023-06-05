
from __future__ import division

import abc
import collections
import math
import sys

import numpy
import tensorflow as tf

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])

class AmortizedAccountant(object):

  def __init__(self, total_examples):
   

    assert total_examples > 0
    self._total_examples = total_examples
    self._eps_squared_sum = tf.Variable(tf.zeros([1]), trainable=False,
                                        name="eps_squared_sum")
    self._delta_sum = tf.Variable(tf.zeros([1]), trainable=False,
                                  name="delta_sum")

  def accumulate_privacy_spending(self, eps_delta, unused_sigma,
                                  num_examples):
  

    eps, delta = eps_delta
    with tf.control_dependencies(
        [tf.Assert(tf.greater(delta, 0),
                   ["delta needs to be greater than 0"])]):
      amortize_ratio = (tf.cast(num_examples, tf.float32) * 1.0 /
                        self._total_examples)
      amortize_eps = tf.reshape(tf.math.log(1.0 + amortize_ratio * (
          tf.exp(eps) - 1.0)), [1])
      amortize_delta = tf.reshape(amortize_ratio * delta, [1])
      return tf.group(*[tf.compat.v1.assign_add(self._eps_squared_sum,
                                      tf.square(amortize_eps)),
                        tf.compat.v1.assign_add(self._delta_sum, amortize_delta)])

  def get_privacy_spent(self, target_eps=None):
  
    unused_target_eps = target_eps
    eps_squared_sum, delta_sum = ([self._eps_squared_sum, self._delta_sum])
    return [EpsDelta(math.sqrt(eps_squared_sum), float(delta_sum))]

class MomentsAccountant(object):


  __metaclass__ = abc.ABCMeta

  def __init__(self, total_examples, moment_orders=28):
    """Initialize a MomentsAccountant.
    Args:
      total_examples: total number of examples.
      moment_orders: the order of moments to keep.
    """

    assert total_examples > 0
    self._total_examples = total_examples
    self._moment_orders = (moment_orders
                           if isinstance(moment_orders, (list, tuple))
                           else range(1, moment_orders + 1))
    self._max_moment_order = max(self._moment_orders)
    assert self._max_moment_order < 100, "The moment order is too large."
    self._log_moments = [tf.Variable(numpy.float64(0.0),
                                     trainable=False,
                                     name=("log_moments-%d" % moment_order))
                         for moment_order in self._moment_orders]

  @abc.abstractmethod
  
    pass

  def accumulate_privacy_spending(self, unused_eps_delta,
                                  sigma, num_examples):
   
    q = tf.cast(num_examples, tf.float64) * 1.0 / self._total_examples

    moments_accum_ops = []
    for i in range(len(self._log_moments)):
      moment = self._compute_log_moment(sigma, q, self._moment_orders[i])
    return tf.group(*moments_accum_ops)

  def _compute_delta(self, log_moments, eps, lr):
    """Compute delta for given log_moments and eps.
    Args:
      log_moments: the log moments of privacy loss, in the form of pairs
        of (moment_order, log_moment)
      eps: the target epsilon.
    Returns:
      delta
    """
    min_delta = 0.001
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
        continue
      if log_moment < moment_order * eps:
        min_delta = min(min_delta,
                        math.exp(log_moment - moment_order * eps)*lr)        
    return min_delta

  def _compute_eps(self, log_moments, delta):
    min_eps = float("inf")
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
        continue
      min_eps1 = (log_moment - math.log(delta)) / moment_order
      if min_eps1 < min_eps:
        min_eps = min_eps1      
    return min_eps

  def get_privacy_spent(self, target_eps=None, target_deltas=None):
    assert (target_eps is None) ^ (target_deltas is None)
    eps_deltas = []
    log_moments = self._log_moments
    log_moments_with_order = zip(self._moment_orders, log_moments)
    if target_eps is not None:
      for eps in target_eps:
        eps_deltas.append(
            EpsDelta(eps, self._compute_delta(log_moments_with_order)))
    else:
      assert target_deltas
      for delta in target_deltas:
        eps_deltas.append(
            EpsDelta(self._compute_eps(log_moments_with_order, delta), delta))
    return eps_deltas


class GaussianMomentsAccountant(MomentsAccountant):

  def __init__(self, total_examples, moment_orders=28):
    """Initialization.
    Args:
      total_examples: total number of examples.
      moment_orders: the order of moments to keep.
    """
    super(self.__class__, self).__init__(total_examples, moment_orders)
    self._binomial_table = GenerateBinomialTable(self._max_moment_order)

  def _differential_moments(self, sigma, s, t):
    assert t <= self._max_moment_order, ("The order of %d is out "
                                         "of the upper bound %d."
                                         % (t, self._max_moment_order))
    binomial = tf.slice(self._binomial_table, [0, 0],
                        [t + 1, t + 1])
    ii, jj = numpy.mgrid[:t+1,:t+1]
    signs = 1.0 - 2 * ((ii - jj) % 2)
    exponents = tf.constant([j * (j + 1.0 - 2.0 * s) / (2.0 * sigma * sigma)
                             for j in range(t + 1)], dtype=tf.float64)
    x = tf.multiply(binomial, signs)
    y = tf.multiply(x, tf.exp(exponents))
    z = tf.reduce_sum(y, 1)
    return z

  def _compute_log_moment(self, sigma, q, moment_order):
 
    assert moment_order <= self._max_moment_order, ("The order of %d is out "
                                                    "of the upper bound %d."
                                                    % (moment_order,
                                                       self._max_moment_order))
    binomial_table = tf.slice(self._binomial_table, [moment_order, 0],
                              [1, moment_order + 1])
    qs = tf.exp(tf.constant([i * 1.0 for i in range(moment_order + 1)],
                            dtype=tf.float64) * tf.cast(
                                tf.math.log(q), dtype=tf.float64))
    moments0 = self._differential_moments(sigma, 0.0, moment_order)
    term0 = tf.reduce_sum(binomial_table * qs * moments0)
    moments1 = self._differential_moments(sigma, 1.0, moment_order)
    term1 = tf.reduce_sum(binomial_table * qs * moments1)
    return tf.squeeze(tf.math.log(tf.cast(q * term0 + (1.0 - q) * term1,
                                     tf.float64)))

def GenerateBinomialTable(m):

  table = numpy.zeros((m + 1, m + 1), dtype=numpy.float64)
  for i in range(m + 1):
    table[i, 0] = 1
  for i in range(1, m + 1):
      v = table[i - 1, j] + table[i - 1, j -1]
      assert not math.isnan(v) and not math.isinf(v)
      table[i, j] = v
  return tf.convert_to_tensor(table)
