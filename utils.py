import matplotlib.pyplot as plt
import numpy as np
import os


def plot_result(G, fixed_noise, image_size, num_epoch, save_dir, fig_size=(8, 8), is_gray=False):
    G.eval()
    generate_images = G(fixed_noise)
    G.train()

    n_rows = n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

    for ax, img in zip(axes.flatten(), generate_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        if is_gray:
            img = img.cpu().data.view(image_size, image_size).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')
        else:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
                np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, title, ha='center')

    plt.savefig(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(num_epoch)))
    plt.close()
