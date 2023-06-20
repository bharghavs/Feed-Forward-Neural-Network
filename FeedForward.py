# Bharghav Srikhakollu
# 05/07/2023

from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
font = {'weight' : 'normal','size'   : 22}
matplotlib.rc('font', **font)
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

######################################################
# Q4 Implement Init, Forward, and Backward For Layers
######################################################

class CrossEntropySoftmax:
  
  # Compute the cross entropy loss after performing softmax
  # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
  # labels -- batch_size x 1 vector of integer label id (0,1,2) where labels[i] is the label for batch element i
  #
  # Output should be a positive scalar value equal to the average cross entropy loss after softmax
  def forward(self, logits, labels):
    # raise Exception('Student error: You haven\'t implemented the forward pass for CrossEntropySoftmax yet.')
    # saving the below to use for back propagation
    self._logits = logits
    self._labels = labels
    # soft_max formula
    soft_max = np.divide(np.exp(logits), np.sum(np.exp(logits), axis = 1, keepdims = True))
    #print(soft_max)
    # cross entropy loss
    #loss = -np.mean(np.log(soft_max[np.arange(len(labels)), labels]))
    log_llhd = -np.log(soft_max[range(labels.shape[0]), labels.flatten()])
    loss = np.sum(np.sum(log_llhd)) / (labels.shape[0] * 3)
    return loss

  # Compute the gradient of the cross entropy loss with respect to the the input logits
  def backward(self):
    # raise Exception('Student error: You haven\'t implemented the backward pass for CrossEntropySoftmax yet.')
    grad_cross = np.divide(np.exp(self._logits), np.sum(np.exp(self._logits), axis = 1, keepdims = True))
    grad_cross[range(self._labels.shape[0]), self._labels.flatten()] -= 1
    return grad_cross

class ReLU:

  # Compute ReLU(input) element-wise
  def forward(self, input):
    # raise Exception('Student error: You haven\'t implemented the forward pass for ReLU yet.')
    # storing the information to make use of them during backpropagation
    self._ReLUinp = input
    self.ReLUout = np.maximum(input, 0)
    return self.ReLUout
 
  # Given dL/doutput, return dL/dinput
  def backward(self, grad):
    # raise Exception('Student error: You haven\'t implemented the backward pass for ReLU yet.')
    # For ReLU function, when input is > 0, output is same as input and the gradient of output will be 1
    # For ReLU function, when input is <= 0, output is 0 and the gradient of output will be 0
    dL_dinput = np.where(self._ReLUinp > 0, grad, 0)
    return dL_dinput
    
  # No parameters so nothing to do during a gradient descent step
  def step(self,step_size):
    return

class LinearLayer:

  # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
  def __init__(self, input_dim, output_dim):
    # raise Exception('Student error: You haven\'t implemented the init for LinearLayer yet.')
    #print("input dim is: ", input_dim)
    #print("output dim is: ", output_dim)
    # He initialization
    self.W = np.random.randn(input_dim, output_dim)*np.sqrt(2.0/input_dim)
    # (1,output_dim) bias vector
    self.b = np.zeros((1, output_dim))
    
    # Initializations for ADAM with weight decay
    self.w_decay, self.epsilon, self.beta1, self.beta2 = 1e-3, 1e-5, 0.9, 0.95
    self.m_hat, self.v_hat = np.zeros((input_dim, output_dim)), np.zeros((input_dim, output_dim))
    
  # During the forward pass, we simply compute XW+b
  def forward(self, input):
    # raise Exception('Student error: You haven\'t implemented the forward pass for LinearLayer yet.')
    # compute XW+b and return the value
    self._Lininp = input
    return np.dot(input, self.W) + (self.b)

  # Inputs:
  #
  # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where 
  #         the i'th row is the gradient of the loss of example i with respect 
  #         to z_i (the output of this layer for example i)

  # Computes and stores:
  #
  # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
  #                       of the loss with respect to the weights of this layer. 
  #                       This is an summation over the gradient of the loss of
  #                       each example with respect to the weights.
  #
  # self.grad_bias dL/dZ--     A (1 x output_dim) matrix storing the gradient
  #                       of the loss with respect to the bias of this layer. 
  #                       This is an summation over the gradient of the loss of
  #                       each example with respect to the bias.
  
  # Return Value:
  #
  # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where
  #               the i'th row is the gradient of the loss of example i with respect 
  #               to x_i (the input of this layer for example i) 

  def backward(self, grad):
      # raise Exception('Student error: You haven\'t implemented the backward pass for LinearLayer yet.')
      # dL/dW = dL/doutput * doutput/dW - where dL/doutput = grad and doutput/dW = input
      #print(grad.shape)
      #print(self._Lininp.shape)
      self.grad_weights = np.dot(self._Lininp.T, grad)
      # dL/dZ = dL/doutput * doutput/dZ
      self.grad_bias = np.sum(grad, axis = 0)
      dL_dX = np.dot(grad, self.W.T)
      return dL_dX
      
  ######################################################
  # Q5 Implement ADAM with Weight Decay
  ######################################################  
  def step(self, step_size):
    # raise Exception('Student error: You haven\'t implemented the step for LinearLayer yet.')
    # m_hat and v_hat calculation
    self.m_hat = (self.beta1 * self.m_hat) + ((1.0 - self.beta1) * (self.grad_weights + (self.w_decay * self.W)))
    self.v_hat = (self.beta2 * self.v_hat) + ((1.0 - self.beta2) * ((self.grad_weights + (self.w_decay * self.W))**2))
    # m_hat and v_hat with bias correction terms
    m_hat_bias = self.m_hat / (1.0 - self.beta1)
    v_hat_bias = self.v_hat / (1.0 - self.beta2)
    # parameter update: weights
    # Updating only weights as per the instruction given in the assignment as below.
    # "Note that weight decay is typically not applied to biases in linear layers."
    self.W -= step_size * (m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon))
    
######################################################
# Q6 Implement Evaluation and Training Loop
###################################################### 

# common accuracy calculation function - for training/validation/test accuracies.
def cal_accuracy(X_val, Y_true):
    Y_pred = np.argmax(X_val, axis = 1)
    Y_pred = Y_pred.reshape(Y_true.shape)
    acc = np.mean(Y_pred == Y_true)
    return acc

# Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
  # raise Exception('Student error: You haven\'t implemented the step for evalute function.')
  loss_function = CrossEntropySoftmax()
  eval_loss = []
  eval_acc = []
  num_ex, input_dim = X_val.shape
  for batch in range(0, num_ex, batch_size):
      p = batch
      q = batch + batch_size
      x, y = X_val[p:q], Y_val[p:q]
      _x_batch = model.forward(x)
      exps_e = np.exp(_x_batch)
      y_predict_e = np.divide(exps_e, np.sum(exps_e,axis = 1, keepdims = True))
      batch_loss = loss_function.forward(y_predict_e, y)
      eval_loss.append(batch_loss)
      evaluate_acc = cal_accuracy(_x_batch, y)
      eval_acc.append(evaluate_acc)
      eval_avg_loss = np.average(eval_loss)
      eval_avg_acc = np.average(eval_acc)
  return eval_avg_loss, eval_avg_acc
     
def main():

  # Set optimization parameters (NEED TO CHANGE THESE)
  batch_size = 256
  max_epochs = 75
  step_size = 1e-3

  number_of_layers = 3
  width_of_layers = 512

  # Load data
  X_train, Y_train, X_val, Y_val, X_test, Y_test = loadCIFAR10Data()
  x_mean = np.mean(X_train, axis = 0)
  x_std = np.std(X_train, axis = 0)
  X_train, X_val, X_test = (X_train - x_mean)/x_std, (X_val - x_mean)/x_std, (X_test - x_mean)/x_std
  # X_train, X_val, X_test = X_train/255.0 - 0.5, X_val/255.0 - 0.5, X_test/255.0 - 0.5
  
  # Some helpful dimensions
  num_examples, input_dim = X_train.shape
  output_dim = 3 # number of class labels

  # Build a network with input feature dimensions, output feature dimension,
  # hidden dimension, and number of layers as specified below. You can edit this as you please.
  net = FeedForwardNeuralNetwork(input_dim,output_dim, width_of_layers, number_of_layers)
  loss_function = CrossEntropySoftmax()
  
  # Some lists for book-keeping for plotting later
  losses = []
  val_losses = []
  accs = []
  val_accs = []
  
  # raise Exception('Student error: You haven\'t implemented the training loop yet.')
  # For each epoch below max epochs
  for epoch in range(max_epochs):
      
    # Scramble order of examples
    scramble_idx = np.random.permutation(num_examples)
    X_train_scramble, Y_train_scramble = X_train[scramble_idx], Y_train[scramble_idx]
    
    # for each batch in data:
    for batch in range(0, num_examples, batch_size):
        
      # Gather batch
      i = batch
      j = batch + batch_size
      X_batch, Y_batch = X_train_scramble[i:j], Y_train_scramble[i:j]

      # Compute forward pass
      _X_batch = net.forward(X_batch)

      # Compute loss
      batch_loss = loss_function.forward(_X_batch, Y_batch)

      # Backward loss and networks
      batch_loss_backward = loss_function.backward()
      net.backward(batch_loss_backward)

      # Take optimizer step
      net.step(step_size)

      # Book-keeping for loss / accuracy
      losses.append(batch_loss)
      exps = np.exp(_X_batch)
      y_predict = np.divide(exps, np.sum(exps,axis = 1, keepdims = True))
      train_acc = cal_accuracy(y_predict, Y_batch)
      accs.append(train_acc)
      
    # Evaluate performance on validation.
    val_loss, val_acc = evaluate(net, X_val, Y_val, batch_size)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    epoch_avg_loss = np.average(losses)
    epoch_avg_acc = np.average(accs)
    ###############################################################
    # Print some stats about the optimization process after each epoch
    ###############################################################
    # epoch_avg_loss -- average training loss across batches this epoch
    # epoch_avg_acc -- average accuracy across batches this epoch
    # vacc -- validation accuracy this epoch
    ###############################################################
    
    logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(epoch,epoch_avg_loss, epoch_avg_acc*100, val_acc*100))

  ###############################################################
  # Code for producing output plot requires
  ###############################################################
  # losses -- a list of average loss per batch in training
  # accs -- a list of accuracies per batch in training
  # val_losses -- a list of average validation loss at each epoch
  # val_acc -- a list of validation accuracy at each epoch
  # batch_size -- the batch size
  ################################################################

  # Plot training and validation curves
  fig, ax1 = plt.subplots(figsize=(16,9))
  color = 'tab:red'
  ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
  ax1.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_losses))], val_losses,c="red", label="Val. Loss")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
  ax1.tick_params(axis='y', labelcolor=color)
  #ax1.set_ylim(-0.01,3)
  
  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:blue'
  ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
  ax2.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_accs))], val_accs,c="blue", label="Val. Acc.")
  ax2.set_ylabel(" Accuracy", c=color)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_ylim(-0.01,1.01)
  
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  ax1.legend(loc="center")
  ax2.legend(loc="center right")
  plt.show()  

  ################################
  # Q7 Tune and Evaluate on Test
  ################################
  
  _, tacc = evaluate(net, X_test, Y_test, batch_size)
  print(tacc)

#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

  def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
 
    if num_layers == 1:
      self.layers = [LinearLayer(input_dim, output_dim)]
    else:
      self.layers = [LinearLayer(input_dim, hidden_dim)]
      self.layers.append(ReLU())
      for i in range(num_layers-2):
        self.layers.append(LinearLayer(hidden_dim, hidden_dim))
        self.layers.append(ReLU())
      self.layers.append(LinearLayer(hidden_dim, output_dim))

  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def backward(self, grad):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def step(self, step_size):
    for layer in self.layers:
      layer.step(step_size)

#####################################################
# Utility Functions for Loading and Visualizing Data
#####################################################

def loadCIFAR10Data():

  with open("cifar10_hst_train", 'rb') as fo:
    data = pickle.load(fo)
  X_train = data['images']
  Y_train = data['labels']

  with open("cifar10_hst_val", 'rb') as fo:
    data = pickle.load(fo)
  X_val = data['images']
  Y_val = data['labels']

  with open("cifar10_hst_test", 'rb') as fo:
    data = pickle.load(fo)
  X_test = data['images']
  Y_test = data['labels']
  
  logging.info("Loaded train: " + str(X_train.shape))
  logging.info("Loaded val: " + str(X_val.shape))
  logging.info("Loaded test: " + str(X_test.shape))
  
  return X_train, Y_train, X_val, Y_val, X_test, Y_test


def displayExample(x):
  r = x[:1024].reshape(32,32)
  g = x[1024:2048].reshape(32,32)
  b = x[2048:].reshape(32,32)
  
  plt.imshow(np.stack([r,g,b],axis=2))
  plt.axis('off')
  plt.show()


if __name__=="__main__":
  main()