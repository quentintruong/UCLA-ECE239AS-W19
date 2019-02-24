import numpy as np
import pdb

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize all parameters of the network in the self.params dictionary.
    #   The weights and biases of layer 1 are W1 and b1; and in general the 
    #   weights and biases of layer i are Wi and bi. The
    #   biases are initialized to zero and the weights are initialized
    #   so that each parameter has mean 0 and standard deviation weight_scale.
    #
    #   BATCHNORM: Initialize the gammas of each layer to 1 and the beta
    #   parameters to zero.  The gamma and beta parameters for layer 1 should
    #   be self.params['gamma1'] and self.params['beta1'].  For layer 2, they
    #   should be gamma2 and beta2, etc. Only use batchnorm if self.use_batchnorm 
    #   is true and DO NOT batch normalize the output scores.
    # ================================================================ #
    all_dims = [input_dim] + hidden_dims + [num_classes]

    #for d in all_dims:
    #  print(d)

    for layer in range(self.num_layers):
      self.params['W{}'.format(layer + 1)] = np.random.normal(0, weight_scale, (all_dims[layer], all_dims[layer + 1]))
      self.params['b{}'.format(layer + 1)] = np.zeros(all_dims[layer + 1])
      #print("{}x{}".format(all_dims[layer], all_dims[layer+1]))

    for layer in range(self.num_layers - 1):
      if self.use_batchnorm:
        self.params['gamma{}'.format(layer + 1)] = np.ones(all_dims[layer + 1])
        #print('gamma{}'.format(layer + 1))
        self.params['beta{}'.format(layer + 1)] = np.zeros(all_dims[layer + 1])
        #print('beta{}'.format(layer + 1))
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the FC net and store the output
    #   scores as the variable "scores".
    #
    #   BATCHNORM: If self.use_batchnorm is true, insert a bathnorm layer
    #   between the affine_forward and relu_forward layers.  You may
    #   also write an affine_batchnorm_relu() function in layer_utils.py.
    #
    #   DROPOUT: If dropout is non-zero, insert a dropout layer after
    #   every ReLU layer.
    # ================================================================ #
    a = {}
    bn = {}
    h = {}
    d = {}
    h[0] = [X]
    for layer in range(self.num_layers - 1):
      if self.use_batchnorm and not self.use_dropout:
        a[layer + 1] = affine_forward(h[layer][0], self.params['W{}'.format(layer + 1)],  self.params['b{}'.format(layer + 1)])
        #print("a{}".format(layer + 1))
        bn[layer + 1] = batchnorm_forward(a[layer + 1][0], self.params['gamma{}'.format(layer + 1)], self.params['beta{}'.format(layer + 1)], self.bn_params[layer])
        #print("bn{}".format(layer + 1))
        h[layer + 1] = relu_forward(bn[layer + 1][0])
        #print("h{}".format(layer + 1))
      elif not self.use_batchnorm and self.use_dropout:
        if layer == 0:
          d[layer] = [X]
        a[layer + 1] = affine_forward(d[layer][0], self.params['W{}'.format(layer + 1)],  self.params['b{}'.format(layer + 1)])
        h[layer + 1] = relu_forward(a[layer + 1][0])
        d[layer + 1] = dropout_forward(h[layer + 1][0], self.dropout_param)
      elif self.use_batchnorm and self.use_dropout:
        if layer == 0:
          d[layer] = [X]
        a[layer + 1] = affine_forward(d[layer][0], self.params['W{}'.format(layer + 1)],  self.params['b{}'.format(layer + 1)])
        bn[layer + 1] = batchnorm_forward(a[layer + 1][0], self.params['gamma{}'.format(layer + 1)], self.params['beta{}'.format(layer + 1)], self.bn_params[layer])
        h[layer + 1] = relu_forward(bn[layer + 1][0])
        d[layer + 1] = dropout_forward(h[layer + 1][0], self.dropout_param)
      else:
        a[layer + 1] = affine_forward(h[layer][0], self.params['W{}'.format(layer + 1)],  self.params['b{}'.format(layer + 1)])
        #print("a{}".format(layer + 1))
        h[layer + 1] = relu_forward(a[layer + 1][0])
        #print("h{}".format(layer + 1))

    if self.use_dropout:
      a[self.num_layers] = affine_forward(d[self.num_layers - 1][0], self.params['W{}'.format(self.num_layers)],  self.params['b{}'.format(self.num_layers)])
    else:
      a[self.num_layers] = affine_forward(h[self.num_layers - 1][0], self.params['W{}'.format(self.num_layers)],  self.params['b{}'.format(self.num_layers)])
    #print("a{}".format(self.num_layers))
    scores = a[self.num_layers][0]
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backwards pass of the FC net and store the gradients
    #   in the grads dict, so that grads[k] is the gradient of self.params[k]
    #   Be sure your L2 regularization includes a 0.5 factor.
    #
    #   BATCHNORM: Incorporate the backward pass of the batchnorm.
    #
    #   DROPOUT: Incorporate the backward pass of dropout.
    # ================================================================ #
    dl_da = {}
    dl_dbn = {}
    dl_dh = {}
    dl_dd = {}
    dl_dw = {}
    dl_db = {}
    loss, dl_da[self.num_layers] = softmax_loss(scores, y)
    #print("dl_da{}".format(self.num_layers))
    loss += 0.5 * self.reg * np.sum([np.sum(self.params['W{}'.format(layer + 1)] ** 2) for layer in range(self.num_layers)])

    dl_dh[self.num_layers - 1], dl_dw[self.num_layers], dl_db[self.num_layers] = affine_backward(dl_da[self.num_layers], a[self.num_layers][1])
    #print("dl_dh{}".format(self.num_layers - 1))
    for layer in range(self.num_layers)[:0:-1]:
      if self.use_batchnorm and not self.use_dropout:
        dl_dbn[layer] = relu_backward(dl_dh[layer], h[layer][1])
        #print("dl_dbn{}".format(layer))
        dl_da[layer], grads['gamma{}'.format(layer)], grads['beta{}'.format(layer)] = batchnorm_backward(dl_dbn[layer], bn[layer][1])
        #print("dl_da{}".format(layer))
        dl_dh[layer - 1], dl_dw[layer], dl_db[layer] = affine_backward(dl_da[layer], a[layer][1])
        #print("dl_dh{}".format(layer - 1))
      elif not self.use_batchnorm and self.use_dropout:
        if layer == self.num_layers - 1:
          dl_dd[layer] = dl_dh[self.num_layers - 1]
        dl_dh[layer] = dropout_backward(dl_dd[layer], d[layer][1])
        dl_da[layer] = relu_backward(dl_dh[layer], h[layer][1])
        dl_dd[layer - 1], dl_dw[layer], dl_db[layer] = affine_backward(dl_da[layer], a[layer][1])
      elif self.use_batchnorm and self.use_dropout:
        if layer == self.num_layers - 1:
          dl_dd[layer] = dl_dh[self.num_layers - 1]
        dl_dh[layer] = dropout_backward(dl_dd[layer], d[layer][1])
        dl_dbn[layer] = relu_backward(dl_dh[layer], h[layer][1])
        dl_da[layer], grads['gamma{}'.format(layer)], grads['beta{}'.format(layer)] = batchnorm_backward(dl_dbn[layer], bn[layer][1])
        dl_dd[layer - 1], dl_dw[layer], dl_db[layer] = affine_backward(dl_da[layer], a[layer][1])
      else:
        dl_da[layer] = relu_backward(dl_dh[layer], h[layer][1])
        #print("dl_da{}".format(layer))
        dl_dh[layer - 1], dl_dw[layer], dl_db[layer] = affine_backward(dl_da[layer], a[layer][1])
        #print("dl_dh{}".format(layer - 1))

    for layer in range(self.num_layers):
      grads['W{}'.format(layer + 1)] = dl_dw[layer + 1] + self.reg * self.params['W{}'.format(layer + 1)]
      grads['b{}'.format(layer + 1)] = dl_db[layer + 1]

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return loss, grads
