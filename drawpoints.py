import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DrawPoints:
    def __init__(self):
        self.color = ["red", "salmon", "black", "brown", "darkgray", "darkgreen", "darkmagenta", "gold",
                      "blue", "deeppink"]

    def draw(self, features, labels, epoch, batch, pictures_path, loss):
        plt.ion()
        plt.clf()
        ax = plt.subplot(projection='3d')
        for i in range(10):
            ax.scatter3D(features[labels == i, 0], features[labels == i, 1], features[labels == i, 2], '.',
                         c=self.color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.title("epoch={0},batch={1},loss={2}".format(epoch, batch, loss))
        plt.savefig("{0}{1}-{2}.jpg".format(pictures_path, epoch, batch))
        plt.draw()
        plt.pause(0.1)
        plt.ioff()
