'''
simple vis for digit MNIST

Usage:
    python mnist_vis.py <filename> [<img index> default 0]
'''
import matplotlib.pyplot as plt
import h5py
import sys

if len(sys.argv) == 0 or '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(0)


def show_example(img, idx, imgname='mnist'):
    fig_wid = 8
    fig_height = 8
    fig = plt.figure(figsize=(fig_wid, fig_height))
    plt.imshow(img)
    ax = plt.gca()
    ax.axis('on')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    figname = imgname + '_%06d.pdf' % (idx)
    plt.savefig(figname, bbox_inches='tight')
    plt.close()


file_name = sys.argv[1]
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
f = h5py.File(file_name, 'r')

img = f['features'][idx, 0, :, :]
show_example(img, idx)
