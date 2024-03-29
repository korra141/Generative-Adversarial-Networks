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
from tensorflow.python.ops import array_ops
os.environ["CUDA_VISIBLE_DEVICES"] ="2"

tfgan = tf.contrib.gan

# Plotting library.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker

from evaluation_score import inception_score,frechet_distance 

import seaborn as sns
tf.random.set_random_seed(547)
tf.logging.set_verbosity(tf.logging.ERROR)

sns.set(rc={"lines.linewidth": 2.8}, font_scale=2)
sns.set_style("whitegrid")

# Don't forget to select GPU runtime environment in Runtime -> Change runtime type
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

BATCH_SIZE = 64 # @param
NUM_LATENTS = 100 # @param
TRAINING_STEPS =100000 # @param
#SCORE_EPOCH= 1000#@param number of epochs after which FID is to be calculated.


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_images_original = mnist.train.images.reshape((-1, 28, 28, 1))
# pdb.set_trace()
test_images_original = mnist.test.images.reshape((-1, 28, 28, 1))
# pdb.set_trace()

dataset = tf.data.Dataset.from_tensor_slices(train_images_original)
batched_dataset = dataset.shuffle(100000).repeat().batch(BATCH_SIZE)
iterator = batched_dataset.make_one_shot_iterator()
real_data = iterator.get_next()

tr_N=int(np.floor(len(mnist.train.images)/BATCH_SIZE)*BATCH_SIZE)
te_N=int(np.floor(len(mnist.test.images)/BATCH_SIZE)*BATCH_SIZE)


train_dataset = tf.data.Dataset.from_tensor_slices(train_images_original[:tr_N])
batched_train_d = train_dataset.batch(BATCH_SIZE)#.map(lambda x:2*x-1)
iterator_train = batched_train_d.make_initializable_iterator()
train_images = iterator_train.get_next() 

test_dataset = tf.data.Dataset.from_tensor_slices(test_images_original[:te_N])
batched_test_d = test_dataset.batch(BATCH_SIZE)#.map(lambda x:2*x-1)
iterator_test = batched_test_d.make_initializable_iterator()
test_images = iterator_test.get_next() 

################################
#####   SCORE ITERATORS   ######
################################

BATCH_SIZE_SCORE= 1
NUMBER_IMAGES=50
tr_N_SCORE = int(np.floor(len(mnist.train.images)/BATCH_SIZE_SCORE)*BATCH_SIZE_SCORE)
te_N_SCORE = int(np.floor(len(mnist.test.images)/BATCH_SIZE_SCORE)*BATCH_SIZE_SCORE)



train_dataset_score = tf.data.Dataset.from_tensor_slices(train_images_original[:NUMBER_IMAGES])
batched_train_d_score = train_dataset_score.batch(NUMBER_IMAGES)#.map(lambda x:2*x-1)
iterator_train_score =batched_train_d_score.make_initializable_iterator()
train_images_score = iterator_train_score.get_next() 

test_dataset_score = tf.data.Dataset.from_tensor_slices(test_images_original[:NUMBER_IMAGES])
batched_test_d_score=test_dataset_score.batch(NUMBER_IMAGES)#.map(lambda x:2*x-1)
iterator_test_score =batched_test_d_score.make_initializable_iterator()
test_images_score = iterator_test_score.get_next() 


# print(images)
#since there is no normalisation in the argo code 
# real_data = 2 * images - 1
# test_images = 2 * test_images - 1
# train_images = 2 * train_images - 1


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
    linear = snt.Linear(7 * 7 * 64,initializers=init_dict)
    inputs = linear(inputs)
    
    # Reshape the data to have rank 4.
    inputs = snt.BatchReshape((7, 7, 64))(inputs)
    inputs = leaky_relu_activation(inputs)
    
    net = snt.nets.ConvNet2DTranspose(
        output_channels=[32, 1],
        output_shapes=[[14, 14], [28, 28]],
        strides=[2],
        paddings=[snt.SAME],
        kernel_shapes=[[5, 5]],
        use_batch_norm=False,
        initializers=init_dict)

    # We use tanh to ensure that the generated samples are in the same range 
    # as the data.
    return tf.nn.sigmoid(net(inputs))

class MnistDiscriminator(snt.AbstractModule):

  def __init__(self,
               leaky_relu_coeff=0.2, name='MnistDiscriminator'):
    super(MnistDiscriminator, self).__init__(name=name)
    self._leaky_relu_coeff = leaky_relu_coeff

  def _build(self, input_image):

    leaky_relu_activation = lambda x: tf.maximum(self._leaky_relu_coeff * x, x)
    init_dict={'w': tf.truncated_normal_initializer(seed=547,stddev=0.02),'b': tf.constant_initializer(0.3)}
    conv2d = snt.nets.ConvNet2D(
        output_channels=[8, 16, 32, 64, 128],
        kernel_shapes=[[5, 5]],
        strides=[2, 1, 2, 1, 2],
        paddings=[snt.SAME],
        activate_final=True,
        activation=leaky_relu_activation,
        use_batch_norm=False,
        initializers=init_dict)

    convolved = conv2d(input_image)
    # Flatten the data to 2D for the classification layer
    flat_data = snt.BatchFlatten()(convolved)

    # We have two classes: one for real, and oen for fake data.
    classification_logits = snt.Linear(2,initializers=init_dict)(flat_data)
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
discriminator_real_data_labels = tf.ones(shape=BATCH_SIZE, dtype=tf.int32)
discriminator_real_prob=tf.nn.softmax(discriminator_real_data_logits)

# np.full((1,BATCH_SIZE))
#loss 

real_data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=discriminator_real_data_logits, labels=discriminator_real_data_labels)

discriminator_samples_logits = discriminator(samples)
discriminator_samples_labels = tf.zeros(shape=BATCH_SIZE, dtype=tf.int32)
samples_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=discriminator_samples_logits, labels=discriminator_samples_labels)

discriminator_loss = tf.reduce_mean(real_data_loss + samples_loss)

discriminator_probabilities = tf.nn.softmax(discriminator_samples_logits)
generator_loss = tf.reduce_mean(- tf.log(discriminator_probabilities[:, 1]))

########################## evaluate

discriminator_train_logits = discriminator(train_images)
discriminator_train_labels = tf.ones(shape=BATCH_SIZE, dtype=tf.int32)

train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=discriminator_train_logits, labels=discriminator_train_labels)

# Reduce loss over batch dimension
discriminator_loss_train = tf.reduce_mean(train_loss + samples_loss)
generator_loss_train = generator_loss


########################## test

discriminator_test_logits = discriminator(test_images)
discriminator_test_labels = tf.ones(shape=BATCH_SIZE, dtype=tf.int32)

test_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=discriminator_test_logits, labels=discriminator_test_labels)

discriminator_loss_test = tf.reduce_mean(test_loss + samples_loss)
generator_loss_test = generator_loss



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

latents_score = tf.random_normal((NUMBER_IMAGES, NUM_LATENTS))
samples_score = generator(latents_score)
#pdb.set_trace()
fid_train=frechet_distance(train_images_score,samples_score,BATCH_SIZE_SCORE)
fid_test=frechet_distance(test_images_score,samples_score,BATCH_SIZE_SCORE)

is_train=inception_score(samples_score,BATCH_SIZE_SCORE)
is_test=inception_score(samples_score,BATCH_SIZE_SCORE)

# pdb.set_trace()
discrimiantor_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9)
generator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.9)

# Optimize the discrimiantor.
discriminator_update_op = discrimiantor_optimizer.minimize(
    discriminator_loss, var_list=discriminator.get_all_variables())

# Optimize the generator..
generator_update_op = generator_optimizer.minimize(
    generator_loss, var_list=generator.get_all_variables())


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7,allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

disc_losses = []
gen_losses = []

disc_losses_train = []
gen_losses_train = []

acc_train=[]
acc_test=[]

disc_losses_test = []
gen_losses_test = []

fid_train_score = []
fid_test_score = []

is_train_score = []
is_test_score = []

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
# with open(check +'/FID','a') as f ,open(check + '/IS','a') as g:
# with open(check +'/FID','a') as f :

for i in range(TRAINING_STEPS):
    _=sess.run(discriminator_update_op)
    _=sess.run(generator_update_op)


    if i%1000 == 0:
        final_samples = sess.run(samples)
        scipy.misc.imsave(g_i +'/fake_image{}.png'.format(i/1000),gallery(final_samples, ncols=8).squeeze(axis=2))
    if i % 1000 == 0: 
        disc_loss,gen_loss = sess.run([discriminator_loss,generator_loss])

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
                disc_loss_train_s += disc_loss_train 
                g_loss_train_s += gen_loss_train 
                acc_tr_s +=accuracy_train_
                count_train = count_train+1
            except tf.errors.OutOfRangeError:
                break
        disc_losses_train.append(disc_loss_train_s/count_train)
        gen_losses_train.append(g_loss_train_s/count_train)
        acc_train.append(acc_tr_s/count_train)

        fid_train_s=0
        is_train_s=0
        count_train_score=0
        sess.run(iterator_train_score.initializer)
        while True:
            try:
                fid_train_,is_train_=sess.run([fid_train,is_train])
                count_train_score=count_train_score+1
                fid_train_s+=fid_train_
                is_train_s+=is_train_
            except tf.errors.OutOfRangeError:
                break
        fid_train_score.append(fid_train_s/count_train_score)
        is_train_score.append(is_train_s/count_train_score)



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

        fid_test_s=0
        count_test_score=0
        is_test_s=0
        sess.run(iterator_test_score.initializer)
        while True:
            try:
                fid_test_,is_test_=sess.run([fid_test,is_test])
                count_test_score=count_test_score+1
                fid_test_s+=fid_test_
                is_test_s+=is_test_
            except tf.errors.OutOfRangeError:
                break
        fid_test_score.append(fid_test_s/count_test_score)
        is_test_score.append(is_test_s/count_test_score)

        print('At iteration {} out of {}'.format(i, TRAINING_STEPS))
# f.close()
# g.close()

# # pdb.set_trace()
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

fig4=plt.figure(figsize=(11,9))
ax4=fig4.add_subplot(111)
ax4.set_ylabel("FID SCORE")
# ax3.ylim(0.001,0.005)
ax4.plot(fid_train_score,'b',label='train')
ax4.plot(fid_test_score,'r',label='test')
ax4.legend(loc='upper left')

fig5=plt.figure(figsize=(11,9))
ax5=fig5.add_subplot(111)
ax5.set_ylabel("IS SCORE")
# ax3.ylim(0.001,0.005)
ax5.plot(is_train_score,'b',label='train')
ax5.plot(is_test_score,'r',label='test')
ax5.legend(loc='upper left')

plt.show()
fig1.savefig(check +'/generator_loss.png')
fig2.savefig(check +'/discriminator_loss.png')
fig3.savefig(check + '/accu.png')
fig4.savefig(check + '/fid_score.png')
fig5.savefig(check + '/is_score.png')

# # fig.savefig(check+'/loss')

real_data_examples = sess.run(real_data)
scipy.misc.imsave(check + '/real_image.png',gallery(real_data_examples, ncols=8).squeeze(axis=2))





sess.close()
