"""
Commands run

python test_1d_poly.py --deg_true=3 --noise_ratio=0 --fig_prefix=no-noise-deg3-

python test_1d_poly.py --deg_true=10 --noise_ratio=0 --fig_prefix=no-noise-deg10-

python test_1d_poly.py --deg_true=10 --noise_ratio=1 --fig_prefix=noisy-deg10-
"""

import os
import tflearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('deg_true', 5, 'Degree of ground truth polynomial')
tf.app.flags.DEFINE_integer('deg_hypo', 10, 'Degree of hypothesis polynomial')
tf.app.flags.DEFINE_integer('n_train', 11, 'Number of training points')
tf.app.flags.DEFINE_integer('n_test', 50, 'Number of test points')
tf.app.flags.DEFINE_float('x_scale', 1.0, 'The scale of inputs')
tf.app.flags.DEFINE_string('fig_prefix', '', 'The prefix of saved figures')

tf.app.flags.DEFINE_float('noise_ratio', 0.0, 'The ratio of label noise')

tf.app.flags.DEFINE_float('weight_mean', -2.0, 'mean of the weights')
tf.app.flags.DEFINE_float('weight_std', 8.0, 'std of the weights')
tf.app.flags.DEFINE_float('weight_min', 3.0, 'minimum absolute value of weights')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size')
tf.app.flags.DEFINE_integer('n_epoch', 5000000, 'Number of Epochs')


def gen_target():
  weights = FLAGS.weight_std * np.random.randn(FLAGS.deg_true + 1) + FLAGS.weight_mean
  for i in range(len(weights)):
    if weights[i] >= 0 and weights[i] < FLAGS.weight_min:
      weights[i] = FLAGS.weight_min
    if weights[i] < 0 and -weights[i] < FLAGS.weight_min:
      weights[i] = -FLAGS.weight_min
  return weights

def eval_target(xs, target):
  fea = xs.reshape((-1, 1)) ** np.arange(len(target)).reshape((1, -1))
  ys = np.dot(fea, target)

  return ys

def plot_poly(filename, objects):
  plt.figure()
  for obj, name in objects:
    if isinstance(obj, tuple):
      xs, ys = obj
      plt.plot(xs, ys, 'o', label=name)
    else:
      xs = np.linspace(-FLAGS.x_scale, FLAGS.x_scale, 100)
      ys = eval_target(xs, obj)
      plt.plot(xs, ys, '-', label=name)

  plt.legend(loc='best')
  plt.savefig(filename)
  plt.close(plt.gcf())


def eval_on(fea, ys, weights):
  return np.mean((np.dot(fea, weights) - ys)**2)

def add_noise(ys, the_mean=None, the_std=None):
  if the_mean is None:
    the_mean = ys.mean()
    the_std = ys.std()

  noise = the_mean + the_std*np.random.randn(*ys.shape)
  ys = (1-FLAGS.noise_ratio)*ys + FLAGS.noise_ratio * noise
  return (ys, the_mean, the_std)

def do_fitting(xs, ys, validation_set=None):
  if validation_set is None:
    validation_set = (xs, ys)
  xs_tt, ys_tt = validation_set

  fea = xs.reshape((-1, 1)) ** np.arange(FLAGS.deg_hypo + 1).reshape((1, -1))
  fea_tt = xs_tt.reshape((-1, 1)) ** np.arange(FLAGS.deg_hypo + 1).reshape((1, -1))

  # compute the condition number of training matrix
  singular_vals = np.linalg.svd(fea, compute_uv=False)
  cond_num = singular_vals[0] / singular_vals[-1]
  print('#### Condition Number = %g' % cond_num)

  if FLAGS.noise_ratio > 0:
    ys_tt, the_mean, the_std = add_noise(ys_tt)
    ys, the_mean, the_std = add_noise(ys, the_mean, the_std)

  # solve exactly
  weights_solve = np.linalg.solve(fea, ys)

  print('#### Initial distance ||w0 - w*||: %g' % np.linalg.norm(weights_solve))
  return

  if not os.path.exists('figs'):
    os.makedirs('figs')

  weights_sgd = np.zeros(weights_solve.shape)
  for i_epoch in range(FLAGS.n_epoch):
    idx = np.random.permutation(fea.shape[0])
    mean_err = 0.0
    for i_batch in range(fea.shape[0]):
      x = fea[idx[i_batch]]
      y = ys[idx[i_batch]]

      y_pred = np.dot(weights_sgd, x)
      weights_sgd -= FLAGS.learning_rate * (y_pred-y) * x
      mean_err += (y_pred-y)**2
    if i_epoch % 1000 == 0:
      print('Epoch %05d: train error = %.6f' % (i_epoch, mean_err / fea.shape[0]))
      print('          ref train err  = %.6f' % eval_on(fea, ys, weights_solve))
      print('             test error  = %.6f' % eval_on(fea_tt, ys_tt, weights_sgd))
      print('         ref test error  = %.6f' % eval_on(fea_tt, ys_tt, weights_solve))

    if i_epoch < 20 or i_epoch % 50000 == 0:
      plot_poly('figs/%s%05d.png' % (FLAGS.fig_prefix, i_epoch),
                [((xs, ys), 'training samples'),
                 (weights_solve, 'exact solution'),
                 (weights_sgd, 'sgd solution (epoch %d)' % i_epoch)])
    if mean_err < 0.01:
      break

  return weights_solve, weights_sgd


def run_experiment():
  xs = np.linspace(-FLAGS.x_scale, FLAGS.x_scale, FLAGS.n_train)
  np.random.shuffle(xs)
  target = gen_target()
  ys = eval_target(xs, target)

  xs_tt = np.linspace(-FLAGS.x_scale, FLAGS.x_scale, FLAGS.n_test)
  np.random.shuffle(xs_tt)
  ys_tt = eval_target(xs_tt, target)

  weights_solve, weights_sgd = do_fitting(xs, ys, validation_set=(xs_tt, ys_tt))

  print('Target: %s' % target)
  print('Solve:  %s' % weights_solve)
  print('SGD:    %s' % weights_sgd)


if __name__ == '__main__':
  run_experiment()
