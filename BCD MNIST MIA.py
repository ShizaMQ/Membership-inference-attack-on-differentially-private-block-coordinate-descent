from __future__ import print_function, division
import numpy as np
import pandas as pd
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import tensorflow as tf
from accountant1 import *
from sanitizer import *
from pathlib import Path


EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
C=0.01
SIGMA = 4.0
USE_PRIVACY = True
PLOT_RESULTS = True

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Convert to tensor and scale to [0, 1]
ts = transforms.Compose([transforms.ToTensor(), 
                             transforms.Normalize((0,), (1,))])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ts)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ts)


# Manipulate train set
x_d0 = mnist_trainset[0][0].size()[0]
x_d1 = mnist_trainset[0][0].size()[1]
x_d2 = mnist_trainset[0][0].size()[2]
N = x_d3 = len(mnist_trainset)
K = 10
x_train = torch.empty((N,x_d0*x_d1*x_d2), device=device)
y_train = torch.empty(N, dtype=torch.long)
for i in range(N): 
     x_train[i,:] = torch.reshape(mnist_trainset[i][0], (1, x_d0*x_d1*x_d2))
     y_train[i] = mnist_trainset[i][1]
x_train = torch.t(x_train)
y_one_hot = torch.zeros(N, K).scatter_(1, torch.reshape(y_train, (N, 1)), 1)
onehot=y_one_hot
y_one_hot = torch.t(y_one_hot).to(device=device)
y_train = y_train.to(device=device)

# Manipulate test set
N_test = x_d3_test = len(mnist_testset)
x_test = torch.empty((N_test,x_d0*x_d1*x_d2), device=device)
y_test = torch.empty(N_test, dtype=torch.long)
for i in range(N_test): 
     x_test[i,:] = torch.reshape(mnist_testset[i][0], (1, x_d0*x_d1*x_d2))
     y_test[i] = mnist_testset[i][1]
x_test = torch.t(x_test)
y_test_one_hot = torch.zeros(N_test, K).scatter_(1, torch.reshape(y_test, (N_test, 1)), 1)
ytestone=y_test_one_hot
y_test_one_hot = torch.t(y_test_one_hot).to(device=device)
y_test = y_test.to(device=device)


# Initialization of parameters
seed = 40
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.manual_seed(seed)
d0 = x_d0*x_d1*x_d2
d1 = 2000
d2 = K # Layers: input + 2 hidden + output
W1 = 0.01*torch.randn(d1, d0, device=device)
b1 = 0.1*torch.ones(d1, 1, device=device)
W2 = 0.01*torch.randn(d2, d1, device=device)
b2 = 0.1*torch.ones(d2, 1, device=device)
#W3 = 0.01*torch.randn(d3, d2, device=device)
#b3 = 0.1*torch.ones(d3, 1, device=device)
#W4 = 0.01*torch.randn(d4, d3, device=device)
#b4 = 0.1*torch.ones(d4, 1, device=device)

U1 = torch.addmm(b1.repeat(1, N), W1, x_train)
V11 = nn.ReLU()(U1)
U2 = torch.addmm(b2.repeat(1, N), W2, V1)
#V2 = nn.ReLU()(U2)
#U3 = torch.addmm(b3.repeat(1, N), W3, V2)
#V13 = nn.ReLU()(U3) 
#U4 = torch.addmm(b4.repeat(1, N), W4, V3)
V2 = U2

gamma = 1
gamma1 = gamma2 = gamma3  = gamma

rho = 1
rho1 = rho2 = rho3 = rho


alpha = 2
alpha1 = alpha2 = alpha3 = alpha4 = alpha5 = alpha6 = alpha7 \
= alpha8 = alpha9 = alpha10 = alpha

niter = 10
loss1 = np.empty(niter)
loss2 = np.empty(niter)
accuracy_train = np.empty(niter)
accuracy_test = np.empty(niter)
time1 = np.empty(niter)

eps = 0.0
delta = 0.0
max_eps = 0.5
max_delta = 0.00001
target_eps = [0.125,0.25,0.5]
max_target_eps = max(target_eps)
target_delta = [1e-5] #unused
sanitized_grads = []
# Create accountant, sanitizer and metrics

accountant = GaussianMomentsAccountant(len(x_train))
sanitizer = AmortizedGaussianSanitizer(accountant, [C, True])

def updateV(U1,U2,W,b,rho,gamma): 
    #print(torch.tensor(W))   
    #_, d = W.shape
    W= W.numpy()
    W=torch.tensor(W)
    _, d = W.size()
    I = torch.eye(d, device=device)
    U1 = nn.ReLU()(U1)
    _, col_U2 = U2.size()
    Vstar = torch.mm(torch.inverse(rho*(torch.mm(torch.t(W),W)) + gamma*I), \
                     rho*torch.mm(torch.t(W),U2-b.repeat(1,col_U2)) + gamma*U1)
    return Vstar

def updateWb(U, V, W, b, alpha, rho, privacy): 
    d,N = V.size()
    #print(V.size())
    #print(N)
    I = torch.eye(d, device=device)
    _, col_U = U.size()
    eps_delta = EpsDelta(eps, delta)
    
    
    W1=tf.convert_to_tensor(W)
    #b1=tf.convert_to_tensor(b)
    #print("W1")
    #print(W1)
    #x = tf.clip_by_norm(W1, clip_norm=4)
    #print("x")
    #print(x)
    if privacy:
      W2 = sanitizer.sanitize(W1, eps_delta, SIGMA) 
      #b2 = sanitizer.sanitize(b1, eps_delta, SIGMA)    
    
    else:
      W2 = W
    W2= W2.numpy()
    W2=torch.tensor(W2)
    Wstar = torch.mm(alpha*W2 + rho*torch.mm(U - b.repeat(1,col_U),torch.t(V)),\
                     torch.inverse(alpha*I + rho*(torch.mm(V,torch.t(V)))))
    bstar = (alpha*b+rho*torch.sum(U-torch.mm(W2,V), dim=1).reshape(b.size()))/(rho*N + alpha)
    
       
    spent_eps_delta = accountant.get_privacy_spent(target_eps=[max_target_eps])[0]
    #print("spent_eps_delta")
    print(spent_eps_delta)
    
    #print("start")
    #print(sanitized_grad.size())
    #print(sanitized_grad)
    return Wstar, bstar, spent_eps_delta


def relu_prox(a, b, gamma, d, N):
    val = torch.empty(d,N, device=device)
    x = (a+gamma*b)/(1+gamma)
    y = torch.min(b,torch.zeros(d,N, device=device))

    val = torch.where(a+gamma*b < 0, y, torch.zeros(d,N, device=device))
    val = torch.where(((a+gamma*b >= 0) & (b >=0)) | ((a*(gamma-np.sqrt(gamma*(gamma+1))) <= gamma*b) & (b < 0)), x, val)
    val = torch.where((-a <= gamma*b) & (gamma*b <= a*(gamma-np.sqrt(gamma*(gamma+1)))), b, val)
    return val


print('Train on', N, 'samples, validate on', N_test, 'samples')

d=[]
spent_eps_delta = EpsDelta(0, 0)
for k in range(niter):
    start = time.time()
    if (spent_eps_delta.spent_eps > max_eps or spent_eps_delta.spent_delta > max_delta):
                    break
    # update V4
    V2 = (y_one_hot + gamma3*U2 + alpha1*V2)/(1 + gamma3 + alpha1)
    
    # update U4 
    U2 = (gamma3*V2 + rho3*(torch.mm(W2,V1) + b2.repeat(1,N)))/(gamma3 + rho3)

    # update W4 and b4
    #W3, b3 = updateWb(U3,V2,W3,b3,alpha1,rho3, True)
    #W3= W3.numpy()
    #W3=torch.tensor(W3)
    # update V3
    #V2 = updateV(U2,U3,W3,b3,rho3,gamma3)
    
    # update U3
    #U2 = relu_prox(V2,(rho2*torch.addmm(b2.repeat(1,N), W2, V1) + alpha2*U2)/(rho2 + alpha2),(rho2 + alpha2)/gamma2,d2,N)
    
    # update W3 and b3
    W2, b2, h2 = updateWb(U2,V1,W2,b2,alpha3,rho2, True)
    W2= W2.numpy()
    W2=torch.tensor(W2)
    # update V2
    #V2 = updateV(U2,U3,W3,b3,rho3,gamma2)
    
    # update U2
    #U2 = relu_prox(V2,(rho2*torch.addmm(b2.repeat(1,N), W2, V1) + alpha5*U2)/(rho2 + alpha5),(rho2 + alpha5)/gamma2,d2,N)
    
    # update W2 and b2
    #W2, b2 = updateWb(U2,V1,W2,b2,alpha6,rho2, True)
    #W2= W2.numpy()
    #W2=torch.tensor(W2)
    # update V1
    V1 = updateV(U1,U2,W2,b2,rho2,gamma1)
    
    # update U1
    U1 = relu_prox(V1,(rho1*torch.addmm(b1.repeat(1,N), W1, x_train) + alpha7*U1)/(rho1 + alpha7),(rho1 + alpha7)/gamma1,d1,N)

    # update W1 and b1
    W1, b1, h1 = updateWb(U1,x_train,W1,b1,alpha8,rho1, True)
    W1= W1.numpy()
    W1=torch.tensor(W1)

    a1_train = nn.ReLU()(torch.addmm(b1.repeat(1, N), W1, x_train))
    #a2_train = nn.ReLU()(torch.addmm(b2.repeat(1, N), W2, a1_train))
    #a3_train = nn.ReLU()(torch.addmm(b3.repeat(1, N), W3, a2_train))
    #print(torch.addmm(b4.repeat(1, N), W4, a3_train))
    pred = torch.argmax(torch.addmm(b2.repeat(1, N), W2, a1_train), dim=0)
    pred1 = torch.addmm(b2.repeat(1, N), W2, a1_train)

    a1_test = nn.ReLU()(torch.addmm(b1.repeat(1, N_test), W1, x_test))
    #a2_test = nn.ReLU()(torch.addmm(b2.repeat(1, N_test), W2, a1_test))
    #a3_test = nn.ReLU()(torch.addmm(b3.repeat(1, N_test), W3, a2_test))
    pred_test = torch.argmax(torch.addmm(b2.repeat(1, N_test), W2, a1_test), dim=0)
    pred_test1 = torch.addmm(b2.repeat(1, N_test), W2, a1_test)
    
    loss1[k] = gamma3/2*torch.pow(torch.dist(V2,y_one_hot,2),2).cpu().numpy()
    loss2[k] = loss1[k] + rho1/2*torch.pow(torch.dist(torch.addmm(b1.repeat(1,N), W1, x_train),U1,2),2).cpu().numpy() \
    +rho2/2*torch.pow(torch.dist(torch.addmm(b2.repeat(1,N), W2, V1),U2,2),2).cpu().numpy() 
    #+rho3/2*torch.pow(torch.dist(torch.addmm(b3.repeat(1,N), W3, V2),U3,2),2).cpu().numpy() 
    #+rho4/2*torch.pow(torch.dist(torch.addmm(b4.repeat(1,N), W4, V3),U4,2),2).cpu().numpy()
    
    print(h1[1])
    d.insert(k,h1[1])

    # compute training accuracy
    correct_train = pred == y_train
    accuracy_train[k] = np.mean(correct_train.cpu().numpy())
    
    # compute validation accuracy
    correct_test = pred_test == y_test
    accuracy_test[k] = np.mean(correct_test.cpu().numpy())
    
    # compute training time
    stop = time.time()
    duration = stop - start
    time1[k] = duration
    
    # print results
    print('Epoch', k + 1, '/', niter, '\n', 
          '-', 'time:', time1[k], '-', 'sq_loss:', loss1[k], '-', 'tot_loss:', loss2[k], 
          '-', 'acc:', accuracy_train[k], '-', 'val_acc:', accuracy_test[k])
   # if should_terminate:
    #            break


import numpy as np
from typing import Tuple, Text
from scipy import special

import tensorflow as tf
#import tensorflow_datasets as tfds

# Set verbosity.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter(action="ignore", category=ConvergenceWarning)
simplefilter(action="ignore", category=FutureWarning)


import membership_inference_attack1 as mmia

print('Predict on train...')
#pred1=pred1.numpy()
logits_train = pred1
#print(pred1.shape)
print(logits_train.shape)
#print(logits_train)
logits_train=torch.tensor(logits_train)
logits_train=torch.swapaxes(logits_train, 0, 1)
print(logits_train.shape)
#print(logits_train2)
print('Predict on test...')
#print(pred_test1)
#pred_test1=pred_test1.numpy()
logits_test = pred_test1
print(logits_test.shape)
logits_test=torch.tensor(logits_test)
logits_test=torch.swapaxes(logits_test, 0, 1)
print(logits_test.shape)
print('Apply softmax to get probabilities from logits...')
prob_train = special.softmax(logits_train, axis=1)
prob_test = special.softmax(logits_test, axis=1)

print('Compute losses...')
cce = tf.keras.backend.categorical_crossentropy
constant = tf.keras.backend.constant
#y_one_hot=torch.tensor(y_one_hot)
#y_one_hot=torch.swapaxes(y_one_hot, 0, 1)

#y_test_one_hot=torch.tensor(y_test_one_hot)
#y_test_one_hot=swapaxes(y_test_one_hot, 0, 1)
print(onehot.shape)
loss_train = cce(constant(onehot), constant(prob_train), from_logits=False).numpy()
loss_test = cce(constant(ytestone), constant(prob_test), from_logits=False).numpy()


from data_structures import AttackInputData
from data_structures import SlicingSpec
from data_structures import AttackType
from data_structures import SingleMembershipProbabilityResult

import plotting as plotting
import matplotlib.pyplot as plt

y_one_hot1 = tf.convert_to_tensor(onehot)
y_test_one_hot1 = tf.convert_to_tensor(ytestone)

labels_train = np.argmax(y_one_hot1, axis=1)
labels_test = np.argmax(y_test_one_hot1, axis=1)


input = AttackInputData(
  logits_train = logits_train.numpy(),
  logits_test = logits_test.numpy(),
  loss_train = loss_train,
  loss_test = loss_test,
  labels_train = labels_train,
  labels_test = labels_test
)

# Run several attacks for different data slices
attacks_result = mia.run_attacks(input,
                                 SlicingSpec(
                                     entire_dataset = True,
                                     by_class = True,
                                     by_classification_correctness = True
                                 ),
                                 attack_types = [
                                     AttackType.THRESHOLD_ATTACK,
                                     AttackType.LOGISTIC_REGRESSION])


# Plot the ROC curve of the best classifier
fig = plotting.plot_roc_curve(
    attacks_result.get_result_with_max_auc().roc_curve)
#plt.plot(attacks_result.get_result_with_max_auc().roc_curve)
#plt.savefig(fig)
# Print a user-friendly summary of the attacks

print(attacks_result.summary(by_slices = True))
max_auc_attacker = attacks_result.get_result_with_max_auc()
max_advantage_attacker = attacks_result.get_result_with_max_attacker_advantage()
print("Attack type with max AUC: %s, AUC of %.2f, Attacker advantage of %.2f" %
      (max_auc_attacker.attack_type,
       max_auc_attacker.roc_curve.get_auc(),
       max_auc_attacker.roc_curve.get_attacker_advantage()))




