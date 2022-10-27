from pandas import read_csv
from matplotlib import pyplot

dataPath = 'homework/other/resnet34_loss_acc.csv'
pngPath = 'homework/4_resnet34_loss_acc.png'

data = read_csv(dataPath)

fig, ax = pyplot.subplots(2, 1, figsize=(16, 6), dpi=80, sharex=True)

step = 15
ax[0].plot(data['batch'][::step], data['loss'][::step])
ax[0].set_title('ResNet34 (1 epoch, batch size = 256)')
ax[0].set_ylabel('Loss')

ax[1].plot(data['batch'][::step], data['accuracy'][::step])
ax[1].set_xlabel('Batch #')
ax[1].set_ylabel('Accuracy')

fig.savefig(pngPath)