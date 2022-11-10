from os import system, remove
from matplotlib import pyplot
import numpy as np

if __name__ == "__main__":
    pngPath = '6_mnist_hvd__loss_acc2.png'
    numEpochs = 16
    xs = np.arange(numEpochs)

    fig, ax = pyplot.subplots(2, 2, figsize=(12, 8))
    
    for i, numGPUs in enumerate([1, 2, 4, 8]):
        system(f"mpirun -n {numGPUs} python 6_tensorflow2_mnist_hvd.py --epochs {numEpochs}")
        dataPath = f'other/metrics{numGPUs}.dat'
        yAccTrain = np.loadtxt(dataPath, usecols=0, dtype='float')
        yLossTrain = np.loadtxt(dataPath, usecols=1, dtype='float')
        yAccTest = np.loadtxt(dataPath, usecols=2, dtype='float')
        yLossTest = np.loadtxt(dataPath, usecols=3, dtype='float')

        ax[0][0].plot(xs, yLossTrain, linewidth=2, label=f'{numGPUs} GPU(s)')
        ax[0][0].set_title('Training')
        ax[0][0].set_ylabel('Loss')
        ax[0][0].legend()

        ax[0][1].plot(xs, yLossTest, linewidth=2)
        ax[0][1].set_title('Testing')

        ax[1][0].plot(xs, yAccTrain, linewidth=2)
        ax[1][0].set_xlabel('Batch #')
        ax[1][0].set_ylabel('Accuracy')

        ax[1][1].plot(xs, yAccTest, linewidth=2)
        ax[1][1].set_xlabel('Batch #')

    fig.savefig(pngPath)