# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm 
from time import time
import horovod.tensorflow as hvd

def get_args():
    parser = ArgumentParser(description='TensorFlow MNIST Example')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=16, metavar='N',
                        help='number of epochs to train (default: 16)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    return parser.parse_args()

def get_dataset(batch_size):
    (mnist_images, mnist_labels), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
        tf.cast(mnist_labels, tf.int64)
    ))
    test_dset = tf.data.Dataset.from_tensor_slices((
        tf.cast(x_test[..., tf.newaxis] / 255.0, tf.float32),
        tf.cast(y_test, tf.int64)
    ))

    # shuffle the dataset, with shuffle buffer to be 10000
    dataset = dataset.repeat().shuffle(10000).batch(batch_size)
    test_dset  = test_dset.repeat().batch(batch_size)

    dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())
    test_dset = test_dset.shard(num_shards=hvd.size(), index=hvd.rank())

    return dataset, test_dset

def train_model(batch_size, epochs, dataset, test_dset, nsamples, ntests, mnist_model, loss, opt):
    @tf.function
    def training_step(_mnist_model, _images, _labels, _loss, _opt):
        with tf.GradientTape() as tape:
            probs = _mnist_model(_images, training=True)
            loss_value = _loss(_labels, probs)
            pred = tf.math.argmax(probs, axis=1)
            equality = tf.math.equal(pred, _labels)
            accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
        
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss_value, _mnist_model.trainable_variables)
        _opt.apply_gradients(zip(grads, _mnist_model.trainable_variables))
        return loss_value, accuracy

    @tf.function
    def validation_step(_mnist_model, _images, _labels, _loss):
        probs = _mnist_model(_images, training=False)
        pred = tf.math.argmax(probs, axis=1)
        equality = tf.math.equal(pred, _labels)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
        loss_value = _loss(_labels, probs)
        return loss_value, accuracy

    def training_epoch(_dataset, _test_dset, _mnist_model, _loss, _optimizer, _nstep, _ntest_step, _epochNum):
        train_loss = 0.0
        train_acc = 0.0
        for batchNum, (images, labels) in enumerate(_dataset.take(_nstep)):
            loss_value, acc = training_step(_mnist_model, images, labels, _loss, _optimizer)
            train_loss += loss_value / _nstep
            train_acc += acc / _nstep

            # HVD - broadcast model and parameters from rank 0 to the other ranks
            if (_epochNum == 0 and batchNum==0):
                hvd.broadcast_variables(_mnist_model.variables, root_rank=0)
                hvd.broadcast_variables(_optimizer.variables(), root_rank=0)

            # if batchNum % 100 == 0 and hvd.rank() == 0:
            #     print('Epoch - %d, step #%d/%d\tLoss: %.6f\tAcc: %.6f' % (_epochNum, batchNum, _nstep, loss_value, acc))

        # HVD - average the training metrics 
        mean_train_loss = hvd.allreduce(train_loss, average=True)
        mean_train_acc = hvd.allreduce(train_acc, average=True)

        # Testing
        test_acc = 0.0
        test_loss = 0.0
        for batchNum, (images, labels) in enumerate(_test_dset.take(_ntest_step)):
            loss_value, acc = validation_step(_mnist_model, images, labels, _loss)
            test_acc += acc / _ntest_step
            test_loss += loss_value / _ntest_step
        
        # HVD - average the test metrics 
        mean_test_loss = hvd.allreduce(test_loss, average=True)
        mean_test_acc = hvd.allreduce(test_acc, average=True)
        
        return mean_train_loss, mean_train_acc, mean_test_loss, mean_test_acc


    nstep = int(nsamples / batch_size / hvd.size())
    ntest_step = int(ntests / batch_size / hvd.size())
    metrics={}
    metrics['train_acc'] = []
    metrics['valid_acc'] = []
    metrics['train_loss'] = []
    metrics['valid_loss'] = []
    metrics['time_per_epochs'] = []
    for ep in range(epochs):
        tt0 = time()
        training_loss, training_acc, test_loss, test_acc = training_epoch(dataset, test_dset, mnist_model, loss, opt, nstep, ntest_step, ep)
        tt1 = time()
        if hvd.rank() == 0: 
            print('E[%d], train Loss: %.6f, training Acc: %.3f, val loss: %.3f, val Acc: %.3f\t Time: %.3f seconds' % (ep, training_loss, training_acc, test_loss, test_acc, tt1 - tt0))
            metrics['train_acc'].append(training_acc.numpy())
            metrics['train_loss'].append(training_loss.numpy())
            metrics['valid_acc'].append(test_acc.numpy())
            metrics['valid_loss'].append(test_loss.numpy())
            metrics['time_per_epochs'].append(tt1 - tt0)
        
    if hvd.rank() == 0: 
        np.savetxt(f"other/metrics{hvd.size()}.dat", np.array([metrics['train_acc'], metrics['train_loss'], metrics['valid_acc'], metrics['valid_loss'], metrics['time_per_epochs']]).transpose())

if __name__ == "__main__":
    args = get_args()

    hvd.init()
    print("# I am rank %d of %d" %(hvd.rank(), hvd.size()))

    # Get the list of GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Ping GPU to the rank
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset, test_dset = get_dataset(args.batch_size)
    nsamples = 60000
    ntests = 10000

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss = tf.losses.SparseCategoricalCrossentropy()
    opt = tf.optimizers.Adam(learning_rate=args.lr*hvd.size())

    t0 = time()
    train_model(args.batch_size, args.epochs, dataset, test_dset, nsamples, ntests, mnist_model, loss, opt)
    t1 = time()
    
    if hvd.rank() == 0: 
        print("Total training time: %f seconds" %(t1 - t0))
