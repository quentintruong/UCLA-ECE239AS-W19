import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  Hprime = int((x.shape[2] + 2 * pad - w.shape[2]) / stride) + 1
  Wprime = int((x.shape[3] + 2 * pad - w.shape[3]) / stride) + 1
  out = np.zeros((x.shape[0], w.shape[0], Hprime, Wprime))

  for i, dp in enumerate(x):
    padded_dp = np.pad(dp, pad_width=[(0, 0), (pad, pad), (pad, pad)], mode='constant')
    for j, filter in enumerate(w):
      for xpos in range(Hprime):
        xoffset = xpos * stride
        for ypos in range(Wprime):
          yoffset = ypos * stride
          out[i, j, xpos, ypos] = np.sum(np.multiply(padded_dp[:, xoffset:xoffset + w.shape[2], yoffset:yoffset + w.shape[3]], filter)) + b[j]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  dx = np.zeros(x.shape)
  dxp = np.pad(dx, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)
  for i, padded_dp in enumerate(xpad):
    for j, filter in enumerate(w):
      for xpos in range(dout.shape[2]):
        xoffset = xpos * stride
        for ypos in range(dout.shape[3]):
          yoffset = ypos * stride
          dw[j] += dout[i, j, xpos, ypos] * padded_dp[:, xoffset:xoffset + w.shape[2], yoffset: yoffset + w.shape[3]]
          dxp[i, :, xoffset:xoffset + w.shape[2], yoffset: yoffset + w.shape[3]] += dout[i, j, xpos, ypos] * w[j]
  db = np.sum(np.sum(np.sum(dout, axis=3), axis=2), axis=0)
  dx = dxp[:,:,pad:-pad,pad:-pad]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  Hprime = int((x.shape[2] + - pool_param['pool_height']) / pool_param['stride']) + 1
  Wprime = int((x.shape[3] + - pool_param['pool_width']) / pool_param['stride']) + 1
  out = np.zeros((x.shape[0], x.shape[1], Hprime, Wprime))

  for i, dp in enumerate(x):
    for l, layer in enumerate(dp):
      for xpos in range(Hprime):
        xoffset = xpos * pool_param['stride']
        for ypos in range(Wprime):
          yoffset = ypos * pool_param['stride']
          out[i, l, xpos, ypos] = np.amax(layer[xoffset:xoffset + pool_param['pool_height'], yoffset:yoffset + pool_param['pool_width']])

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  dx = np.zeros(x.shape)
  for i, dp in enumerate(x):
    for l, layer in enumerate(dp):
      for xpos in range(dout.shape[2]):
        xoffset = xpos * pool_param['stride']
        for ypos in range(dout.shape[3]):
          yoffset = ypos * pool_param['stride']
          field = layer[xoffset:xoffset + pool_param['pool_height'], yoffset:yoffset + pool_param['pool_width']]
          ixmax, iymax = np.unravel_index(np.argmax(field, axis=None), field.shape)

          dx[i, l, ixmax + xoffset, iymax + yoffset] = dout[i, l, xpos, ypos]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  # Manipulate shape
  out = np.zeros(x.shape)
  xF = np.array([x[:, j, :, :].reshape(-1) for j in range(x.shape[1])])

  bn_xFT, cache = batchnorm_forward(xF.T, gamma, beta, bn_param) # batchnorm_xflattenedtranspose

  # Unmanipulate shape
  for i in range(bn_xFT.shape[1]): 
    out[:, i, :, :] = bn_xFT[:, i].reshape(out.shape[0], out.shape[2], out.shape[3])
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  # Manipulate shape
  dx = np.zeros(dout.shape)
  dgamma = np.zeros(dout.shape[1])
  dbeta = np.zeros(dout.shape[1])
  doutF = np.array([dout[:,j,:,:].reshape(-1) for j in range(dout.shape[1])])

  dxFT, dgamma, dbeta = batchnorm_backward(doutF.T, cache)

  # Unmanipulate shape
  for i in range(dxFT.shape[1]):
    dx[:, i, :, :] = dxFT[:, i].reshape(dx.shape[0], dx.shape[2], dx.shape[3])
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta