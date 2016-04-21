__author__ = 'thushv89'
import numpy as np

def create_image_grid(filters,fig_id):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.cm as cm
    from math import ceil
    filt_w,filt_h = int(ceil(len(filters)**0.5)),int(ceil(len(filters)**0.5))

    fig = plt.figure(1,figsize=(filt_w,filt_h)) # use the same figure, otherwise get main thread is not in main loop
    grid = ImageGrid(fig, 111, nrows_ncols=(filt_w, filt_h), axes_pad=0.1, share_all=True)

    for i in range(len(filters)):
        grid[i].set_xticks([])
        grid[i].set_yticks([])
        grid[i].set_xticklabels([])
        grid[i].set_yticklabels([])
        grid[i].imshow(filters[i], cmap = cm.Greys_r)  # The AxesGrid object work as a list of axes.

    plt.savefig('test_filters'+str(fig_id)+'.jpg')
    plt.close()

if __name__ == '__main__':

    filters1 = []
    for i in range(100):
        filters1.append(np.arange(100).reshape(10,10))

    filters2 = []
    for i in range(115):
        filters2.append(np.arange(100).reshape(10,10))

    filters3 = []
    for i in range(215):
        filters3.append(np.arange(100).reshape(10,10))

    for i in range(10):
        print 'Iteration ',i
        create_image_grid(filters1,1)
        create_image_grid(filters2,2)
        create_image_grid(filters3,3)