from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.misc
import os
# from ops import *
import math

import tensorflow as tf
import sonnet as snt

import numpy as np
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

# Plotting library.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker

import seaborn as sns
tf.random.set_random_seed(547)
tf.logging.set_verbosity(tf.logging.ERROR)

sns.set(rc={"lines.linewidth": 2.8}, font_scale=2)
sns.set_style("whitegrid")

# Don't forget to select GPU runtime environment in Runtime -> Change runtime type
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

BATCH_SIZE = 64 # @param
NUM_LATENTS = 100 # @param
TRAINING_STEPS =50000 # @param


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_images = mnist.train.images.reshape((-1, 28, 28, 1))
# pdb.set_trace()
test_images = mnist.test.images.reshape((-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(train_images)
batched_dataset = dataset.shuffle(100000).repeat().batch(BATCH_SIZE)
iterator = batched_dataset.make_one_shot_iterator()
real_data = iterator.get_next()

tr_N=int(np.floor(len(mnist.train.images)/BATCH_SIZE)*BATCH_SIZE)
te_N=int(np.floor(len(mnist.test.images)/BATCH_SIZE)*BATCH_SIZE)


train_dataset = tf.data.Dataset.from_tensor_slices(train_images[:tr_N])
batched_train_d = train_dataset.batch(BATCH_SIZE)#.map(lambda x:2*x-1)
iterator_train = batched_train_d.make_initializable_iterator()
train_images = iterator_train.get_next() 


test_dataset = tf.data.Dataset.from_tensor_slices(test_images[:te_N])
batched_test_d = test_dataset.batch(BATCH_SIZE)#.map(lambda x:2*x-1)
iterator_test = batched_test_d.make_initializable_iterator()
test_images = iterator_test.get_next() 


# print(images)
#since there is no normalisation in the argo code 
# real_data = 2 * images - 1
# test_images = 2 * test_images - 1
# train_images = 2 * train_images - 1
'''
        "discriminator" : [
          ("Conv2D", {"output_channels" : 64, "kernel_shape" : [4,4], "stride" : 2},  1),
          ("Conv2D", {"output_channels" : 128, "kernel_shape" : [4,4], "stride" : 2},  0),
          ("BatchNorm", {"offset" : 1, "scale" : 1, "decay_rate" : 0.9},   1),
          ("BatchFlatten", {}, 0),
          ("Linear", {"output_size" : 1024}, 0),
          ("BatchNorm", {"offset" : 1, "scale" : 1, "decay_rate" : 0.9},   1),
          ("Linear", {"output_size" : 2}, 0),
          ],
         "generator" : [
          # ("RandomUniform", {"shape": 62, "minval": -1, "maxval": 1}, 0),
          ("RandomGaussian", {"shape": 62}, 0),#creating the latent space new feature
          ("Linear", {"output_size" : 1024}, 0),
          ("BatchNorm", {"offset" : 1, "scale" : 1, "decay_rate" : 0.9},   1),
          ("Linear", {"output_size" : 128 * 7 * 7}, 0),
          ("BatchNorm", {"offset" : 1, "scale" : 1, "decay_rate" : 0.9},   1),
          ("BatchReshape", {"shape" : (7, 7, 128)}, 0),
          ("Conv2DTranspose" ,{ "output_channels" : 64 ,"output_shape" : [14,14], "kernel_shape" : [4,4], "stride" : 2, "padding":"SAME" },    0),
          ("BatchNorm", {"offset" : 1, "scale" : 1, "decay_rate" : 0.9},   1),
          ("Conv2DTranspose" ,{ "output_channels" : 1 ,"output_shape" : [28,28], "kernel_shape" : [4,4], "stride" : 2, "padding":"SAME" },    0),
          ("Sigmoid", {}, 0)
          ],
'''

class MnistGenerator(snt.AbstractModule):

  def __init__(self, name='MnistGenerator'):
    super(MnistGenerator, self).__init__(name=name)

  def _build(self, inputs):
    """Constructs the generator graph.

    Args:
      inputs: `tf.Tensor` with the input of the generator.

    Returns:
      `tf.Tensor`, the generated samples.
    """
    leaky_relu_activation = lambda x: tf.maximum(0.2* x, x)
    init_dict={'w': tf.truncated_normal_initializer(seed=547,stddev=0.02),'b': tf.constant_initializer(0.3)}
    layer1 = snt.Linear(output_size=1024,initializers=init_dict)(inputs)
    layer2 = leaky_relu_activation(snt.BatchNorm(offset=1,scale=1,decay_rate=0.9)(layer1, is_training=True, test_local_stats=True))
    layer3 = snt.Linear(output_size=128*7*7,initializers=init_dict)(layer2)
    layer4 = leaky_relu_activation(snt.BatchNorm(offset=1,scale=1,decay_rate=0.9)(layer3, is_training=True, test_local_stats=True))
    layer5 = snt.BatchReshape((7, 7, 128))(layer4)
          # ("Conv2DTranspose" ,{ "output_channels" : 64 ,"output_shape" : [14,14], "kernel_shape" : [4,4], "stride" : 2, "padding":"SAME" },    0),
    layer6=snt.Conv2DTranspose(output_channels= 64,output_shape=[14,14],kernel_shape=[4,4],stride=2,padding="SAME",initializers=init_dict)(layer5)
    layer7 = leaky_relu_activation(snt.BatchNorm(offset=1,scale=1,decay_rate=0.9)(layer6, is_training=True, test_local_stats=True))
          # ("Conv2DTranspose" ,{ "output_channels" : 1 ,"output_shape" : [28,28], "kernel_shape" : [4,4], "stride" : 2, "padding":"SAME" },    0),
    layer8=snt.Conv2DTranspose(output_channels= 1,output_shape=[28,28],kernel_shape=[4,4],stride=2,padding="SAME",initializers=init_dict)(layer7)
       # Reshape the data to have rank 4.
    # inputs = leaky_relu_activation(inputs)
    
    # net = snt.nets.ConvNet2DTranspose(
    #     output_channels=[32, 1],
    #     output_shapes=[[14, 14], [28, 28]],
    #     strides=[2],
    #     paddings=[snt.SAME],
    #     kernel_shapes=[[5, 5]],
    #     use_batch_norm=False,
    #     initializers=init_dict)

    # # We use tanh to ensure that the generated samples are in the same range 
    # # as the data.
    return tf.nn.sigmoid(layer8)

class MnistDiscriminator(snt.AbstractModule):

  def __init__(self,
               leaky_relu_coeff=0.2, name='MnistDiscriminator'):
    super(MnistDiscriminator, self).__init__(name=name)
    self._leaky_relu_coeff = leaky_relu_coeff

  def _build(self, input_image):

    leaky_relu_activation = lambda x: tf.maximum(self._leaky_relu_coeff * x, x)
    init_dict={'w': tf.truncated_normal_initializer(seed=547,stddev=0.02),'b': tf.constant_initializer(0.3)}
    layer1=snt.Conv2D(output_channels= 64, kernel_shape= [4,4],stride= 2,initializers=init_dict)(input_image)
    layer2=leaky_relu_activation(layer1)
    layer3=snt.Conv2D(output_channels= 128, kernel_shape= [4,4],stride= 2,initializers=init_dict)(layer2)
    layer4=snt.BatchNorm(offset=1,scale=1,decay_rate=0.9)(layer3, is_training=True, test_local_stats=True)
    layer5=leaky_relu_activation(layer4)
    layer6 = snt.BatchFlatten()(layer5)
    layer7=snt.Linear(output_size=1024,initializers=init_dict)(layer6)
    layer8=snt.BatchNorm(offset=1,scale=1,decay_rate=0.9)(layer7, is_training=True, test_local_stats=True)
    layer9=leaky_relu_activation(layer8)
    classification_logits = snt.Linear(2,initializers=init_dict)(layer9)



    # conv2d = snt.nets.ConvNet2D(
    #     output_channels=[8, 16, 32, 64, 128],
    #     kernel_shapes=[[5, 5]],
    #     strides=[2, 1, 2, 1, 2],
    #     paddings=[snt.SAME],
    #     activate_final=True,
    #     activation=leaky_relu_activation,
    #     use_batch_norm=False,
    #     initializers=init_dict)

    # convolved = conv2d(input_image)
    # # Flatten the data to 2D for the classification layer
    # flat_data = snt.BatchFlatten()(convolved)

    # # We have two classes: one for real, and oen for fake data.
    # classification_logits = snt.Linear(2,initializers=init_dict)(flat_data)
    return classification_logits

def gallery(array, ncols=10):
    """Code adapted from: https://stackoverflow.com/questions/42040747/more-idomatic-way-to-display-images-in-a-grid-with-numpy"""

    # if rescale:
    #     array = (array + 1.) / 2
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

latents = tf.random_normal((BATCH_SIZE, NUM_LATENTS))
generator = MnistGenerator()
samples = generator(latents)


discriminator = MnistDiscriminator()

discriminator_real_data_logits = discriminator(real_data)
# discriminator_real_prob=tf.nn.softmax(discriminator_real_data_logits)

discriminator_generated_data_logits = discriminator(samples)


#loss 
# pdb.set_trace()
real_data_loss=-tf.reduce_mean(discriminator_real_data_logits[:,0])
generated_data_loss=tf.reduce_mean(discriminator_generated_data_logits[:,0])
discriminator_loss=tf.reduce_mean(real_data_loss+generated_data_loss)
generator_loss=-generated_data_loss
########################## evaluate

discriminator_train_logits = discriminator(train_images)
train_loss = -tf.reduce_mean(discriminator_train_logits[:,0])

# Reduce loss over batch dimension
discriminator_loss_train = tf.reduce_mean(train_loss + generated_data_loss)
generator_loss_train = generator_loss


########################## test

discriminator_test_logits = discriminator(test_images)

test_loss = -tf.reduce_mean(discriminator_test_logits[:,0])

discriminator_loss_test = tf.reduce_mean(test_loss +generated_data_loss)
generator_loss_test = generator_loss

discriminator_probabilities = tf.nn.softmax(discriminator_generated_data_logits)

#accuracy 
total_fake=tf.equal(tf.argmax(discriminator_probabilities,axis=1),tf.zeros(shape=BATCH_SIZE,dtype=tf.int64))
total_fake=tf.cast(total_fake,tf.int32)
total_fake_=tf.reduce_sum(total_fake)

#accuracy train 
discriminator_prob_train=tf.nn.softmax(discriminator_train_logits)
total_train=tf.equal(tf.argmax(discriminator_prob_train,axis=1),tf.ones(shape=BATCH_SIZE,dtype=tf.int64))
total_train=tf.cast(total_train,tf.int32)
total_train_=tf.reduce_sum(total_train)

accuracy_train=((total_fake_+total_train_)/(2*BATCH_SIZE))
# pdb.set_trace()

#accuracy test
discriminator_prob_test=tf.nn.softmax(discriminator_test_logits)
total_test=tf.equal(tf.argmax(discriminator_prob_test,axis=1),tf.ones(shape=BATCH_SIZE,dtype=tf.int64))
total_test=tf.cast(total_test,tf.int32)
total_test_=tf.reduce_sum(total_test)
accuracy_test=((total_fake_+total_test_)/(2*BATCH_SIZE))





discrimiantor_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00005)
generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00005)


discriminator_update_op = discrimiantor_optimizer.minimize(discriminator_loss, var_list=discriminator.get_all_variables())
generator_update_op = generator_optimizer.minimize(generator_loss, var_list=generator.get_all_variables())
# pdb.set_trace()

clip_value = 0.01
clip_discriminator_vars = [p.assign(tf.clip_by_value(p, -clip_value, clip_value)) for p in discriminator.get_all_variables()]
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, clip_discriminator_vars)

# pdb.set_trace()
'''
self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
'''

config = tf.ConfigProto()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

disc_losses = []
gen_losses = []

disc_losses_train = []
gen_losses_train = []

acc_train=[]
acc_test=[]

disc_losses_test = []
gen_losses_test = []

import time
way=os.getcwd()
timestr=time.strftime("%d%m_%H%M")
check=os.path.join(way,'tmlss_notebook_original'+timestr)
try:
    os.mkdir(check)
except OSError:  
    print ("Creation of the directory %s failed" % check)
else:  
    print ("Successfully created the directory %s " % check)
g_i=os.path.join(check,'generated_images')
try:
    os.mkdir(g_i)
except OSError:
    print("Creation of the directory %s failed" % g_i)
else :
    print("Successfully created the dictionary %s" %g_i)
# # with open(check +'/FID','a') as f ,open(check + '/IS','a') as g:
# # with open(check +'/FID','a') as f :

for i in range(TRAINING_STEPS):
    for j in range(5):
        _=sess.run(discriminator_update_op)
        _=sess.run(clip_discriminator_vars)
   
    _=sess.run(generator_update_op)
    # real_logits=sess.run(discriminator_real_data_logits)
    # fake_logits=sess.run(discriminator_generated_data_logits)
    # pdb.set_trace()
    # print(real_logits)

    if i%1000 == 0:
        final_samples = sess.run(samples)
        scipy.misc.imsave(g_i +'/fake_image{}.png'.format(i/1000),gallery(final_samples, ncols=8).squeeze(axis=2))

    
  
    if i % 500 == 0: 
        disc_loss,gen_loss = sess.run([discriminator_loss,generator_loss])
        # print(disc_loss)
        # print(gen_loss)
        # pdb.set_trace()
        disc_losses.append(disc_loss)
        gen_losses.append(gen_loss)

        disc_loss_train_s=0
        g_loss_train_s=0
        acc_tr_s=0
        count_train=0
        sess.run(iterator_train.initializer)
            
        while True:
            try:
                    
                disc_loss_train, gen_loss_train,accuracy_train_= sess.run([discriminator_loss_train, generator_loss_train,accuracy_train])
                # pdb.set_trace()
                #print(disc_loss_train, gen_loss_train)
                disc_loss_train_s += disc_loss_train 
                g_loss_train_s += gen_loss_train 
                acc_tr_s +=accuracy_train_
                #                 print(count)
                count_train=count_train+1
            except tf.errors.OutOfRangeError:
                break

        disc_losses_train.append(disc_loss_train_s/count_train)
        gen_losses_train.append(g_loss_train_s/count_train)
        acc_train.append(acc_tr_s/count_train)

        disc_loss_test_s=0
        g_loss_test_s=0
        acc_te_s=0
        count_test=0
        
        sess.run(iterator_test.initializer)
        while True:
            try:
                disc_loss_test, gen_loss_test, accuracy_test_ = sess.run([discriminator_loss_test,generator_loss_test,accuracy_test])
                #print(disc_loss_test, gen_loss_test)
                disc_loss_test_s += disc_loss_test
                g_loss_test_s += gen_loss_test
                acc_te_s = acc_te_s+accuracy_test_
                count_test=count_test+1
            except tf.errors.OutOfRangeError:
                break
        disc_losses_test.append(disc_loss_test_s/count_test)
        gen_losses_test.append(g_loss_test_s/count_test)
        acc_test.append(acc_te_s/count_test)
        
        print('At iteration {} out of {}'.format(i, TRAINING_STEPS))
# f.close()
# g.close()

# pdb.set_trace()
fig1=plt.figure(figsize=(11,9))
ax1=fig1.add_subplot(111)
ax1.set_ylabel('generator_loss')
ax1.plot(gen_losses,'g',label='update')
ax1.plot(gen_losses_train,'b',label='train')
ax1.plot(gen_losses_test,'r',label='test')
ax1.legend(loc='upper left')

fig2=plt.figure(figsize=(11,9))
ax2=fig2.add_subplot(111)
ax2.set_ylabel("discriminator loss")
ax2.plot(disc_losses,'g',label='update')
ax2.plot(disc_losses_train,'b',label='train')
ax2.plot(disc_losses_test,'r',label='test')
ax2.plot([np.log(2)] * len(disc_losses), 'r--', label='Discriminator is being fooled')
ax2.legend(loc='upper left')

fig3=plt.figure(figsize=(11,9))
ax3=fig3.add_subplot(111)
ax3.set_ylabel("accuracy")
# ax3.ylim(0.001,0.005)
ax3.plot(acc_train,'b',label='train')
ax3.plot(acc_test,'r',label='test')
ax3.legend(loc='upper left')

plt.show()
fig1.savefig(check +'/generator_loss.png')
fig2.savefig(check +'/discriminator_loss.png')
fig3.savefig(check + '/accu.png')

# fig.savefig(check+'/loss')

real_data_examples = sess.run(real_data)
scipy.misc.imsave(check + '/real_image.png',gallery(real_data_examples, ncols=8).squeeze(axis=2))





sess.close()