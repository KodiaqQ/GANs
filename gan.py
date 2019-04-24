from configparser import ConfigParser

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from network import Generator, Discriminator
from generator import Pipes
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')

    path = config['Data']['path']
    size = (int(config['Data']['size']), int(config['Data']['size']))
    z_size = int(config['Data']['z_size'])
    channel_size = int(config['Data']['channel_size'])
    ngf = int(config['Data']['ngf'])
    ndf = int(config['Data']['ndf'])
    save = config['Data']['save']
    dataset = config['Data']['dataset']

    batch_size = int(config['Train']['batch_size'])
    num_workers = int(config['Train']['num_workers'])
    lr = float(config['Train']['lr'])
    epoches = int(config['Train']['epoches'])

    generator = Pipes({'path': path, 'size': size})

    # generator = dset.ImageFolder(root=path,
    #                              transform=transforms.Compose([
    #                                  transforms.Resize(size[0]),
    #                                  transforms.CenterCrop(size[0]),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                              ]))

    device = torch.device('cuda:0')

    G = Generator(z_size=z_size, out_size=channel_size, ngf=ngf).to(device)
    print('G network structure')
    print(G)

    D = Discriminator(in_size=channel_size, ndf=ndf).to(device)
    print('D network structure')
    print(D)

    train_loader = DataLoader(dataset=generator,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    criterion = nn.BCELoss().cuda()

    optimizerG = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.FloatTensor(1 * 1, z_size, 1, 1).normal_(0, 1)
    fixed_noise = Variable(fixed_noise.cuda(), volatile=True)

    real_label = 1
    fake_label = 0

    for epoch in range(epoches):
        for i, data in enumerate(train_loader):

            D.zero_grad()
            real_cpu = data.to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = D(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, z_size, 1, 1, device=device)
            fake = G(noise)
            label.fill_(fake_label)
            output = D(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            G.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = D(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epoches, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            # plot_result(G, fixed_noise, size, epoch + 1, save, is_gray=(channel_size == 1))
            if epoch % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % save,
                                  normalize=True)
                fake = G(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/%s_fake_epoch_%03d.png' % (save, dataset, epoch),
                                  normalize=True)

        torch.save(G.state_dict(), '%s/%s_netG_epoch_%d.pth' % (save, dataset, epoch))
        torch.save(D.state_dict(), '%s/%s_netD_epoch_%d.pth' % (save, dataset, epoch))
    print('Done')
