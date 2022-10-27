from pandas import read_csv
from matplotlib import pyplot

dataPath = 'resnet34_loss_acc.csv'

data = read_csv(dataPath)

fig, ax = pyplot.subplots(1, 2, figsize=(18, 6), dpi=80) # Create a 1 by 2 plot grid.

ax[0].plot(data['batch'], data['loss'])

ax[1].plot(data['batch'], data['acc'])