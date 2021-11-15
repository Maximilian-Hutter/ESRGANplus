import numpy as np
import torch
import torch as nn
from torch.nn.modules.loss import L1Loss
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import socket
import argparse
from prefetch_generator import BackgroundGenerator
#from torchort import ORTModule
import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import itertools

from ESRGANplus import ESRGANplus
from Models import Discriminator
from get_data import ImageDataset
from feature_extract import FeatureExtractor

if __name__ == '__main__':
    # settings that can be changed with console command options
    parser = argparse.ArgumentParser(description='PyTorch ESRGANplus')
    #parser.add_argument('--ray_tune', type=bool, default=False, help=("Use ray tune to tune parameters"))
    parser.add_argument('--upsample', type=int, default=2, help="super resolution upscale factor")
    parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
    parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')   ### Epochs default 20 !!!!
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning Rate. Default=0.01')
    parser.add_argument('--gpu_mode', type=bool, default=False) ##### cuda default False !!!!
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')  #### default 0 more than 0 gives if __nam__ ?? '__main__' Error
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    parser.add_argument('--dataset_name', type=str, default='DIV2K')
    parser.add_argument('--data_augmentation', type=bool, default=False) 
    parser.add_argument('--model_type', type=str, default='ESRGANplus')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--disc_save_folder',type=str, default='weights/', help='Location to save Discriminator')
    parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')
    parser.add_argument('--channels',type=int, default=3, help='number of channels R,G,B for img / number of input dimensions 3 times 2dConv for img')
    parser.add_argument('--beta1',type=float, default=0.9, help='decay of first order momentum of gradient')
    parser.add_argument('--beta2',type=float, default=0.999, help='decay of first order momentum of gradient')
    parser.add_argument('--hr_height', type=int, default= 2048, help='high res. image height')
    parser.add_argument('--hr_width', type=int, default= 1080, help='high res. image width')
    parser.add_argument('--n_resblock', type=int, default=23, help='number of Res Blocks')
    parser.add_argument('--perceplambda',type=float, default=1, help='perceptionloss weight')
    parser.add_argument('--contlambda',type=float, default=5e-3, help='contentloss weight')
    parser.add_argument('--advlambda',type=float, default=1e-2, help='adverserialloss weight')
    parser.add_argument('--filters',type=int, default=64, help='number of filters')
    parser.add_argument('--mini_batch',type=int, default=16, help='mini batch size')
    parser.add_argument('--sample_interval',type=int, default=100, help='Number of epochs for learning rate decay')
    parser.add_argument('--resume',type=bool, default=False, help='resume training/ load checkpoint')

    opt = parser.parse_args()
    np.random.seed(opt.seed)    # set seed to default 123 or opt
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    gpus_list = range(opt.gpus)
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(opt)  # print the chosen parameters

    # set Image shape
    hr_shape = (opt.hr_height, opt.hr_width)

    # data loading
    print('==> Loading Datasets')
    dataloader = DataLoader(ImageDataset("../../data/%s/train_HR" % opt.dataset_name, hr_shape = hr_shape), batch_size=opt.batchSize, shuffle=True, num_workers=opt.threads)

    # instantiate model (n_feat = filters)
    Generator = ESRGANplus(opt.channels, filters=opt.filters, num_upsample = opt.upsample, n_resblock = opt.n_resblock)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set model to inference mode
    feature_extractor.eval()

    # loss
    adverserial_criterion = torch.nn.BCEWithLogitsLoss()
    perceptual_criterion = L1Loss()
    content_criterion = L1Loss()

    # run on gpu
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    if cuda:
        Generator = Generator.cuda(gpus_list[0])
        discriminator = discriminator.cuda(gpus_list[0])
        feature_extractor = feature_extractor.cuda(gpus_list[0])
        adverserial_criterion = adverserial_criterion.cuda(gpus_list[0])
        perceptual_criterion = perceptual_criterion.cuda(gpus_list[0])
        content_criterion = content_criterion.cuda(gpus_list[0])

    # optimizer
    optimizer_G = optim.Adam(Generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))  # ESRGANplus / generator optimizer
    optimizer_D = optim.Adam(discriminator.parameters(),lr=opt.lr, betas=(opt.beta1, opt.beta2)) # Discriminator optimizer


    # load checkpoint/load model
    star_n_iter = 0
    start_epoch = 0
    if opt.resume:
        checkpointG = torch.load(opt.save_folder) ## look at what to load
        checkpointD = torch.load(opt.disc_save_folder) 
        Generator.load_state_dict(checkpointG['net'])
        discriminator.load_state_dict(checkpointD['net'])
        start_epoch = checkpointG['epoch']
        start_n_iter = checkpointG['n_iter']
        optimizer_G.load_state_dict(checkpointG['optim'])
        optimizer_D.load_state_dict(checkpointD['optim'])
        print("last checkpoint restored")

    # multiple gpu run
    Generator = torch.nn.DataParallel(Generator, device_ids=gpus_list)
    discriminator = torch.nn.DataParallel(Generator, device_ids=gpus_list)
    feature_extractor = torch.nn.DataParallel(Generator, device_ids=gpus_list)

    # tensor board
    writer = SummaryWriter()

    # define Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    for epoch in range(start_epoch, opt.nEpochs):
        epoch_loss = 0
        Generator.train()

        # prefetch generator and tqdm for iterating trough data
        #pbar = tqdm(enumerate(BackgroundGenerator(dataloader,1)), total = len(dataloader)),
        start_time = time.time()


        for imgs in enumerate(BackgroundGenerator(dataloader,1)):   #  for data in pbar
            # data preparation

            imgs_lr = Variable(imgs["lr"].type(Tensor)) # get low res images from dataloader
            imgs_hr = Variable(imgs["hr"].type(Tensor)) # get high res images from dataloader
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            if cuda:    # put variables to gpu
                imgs_lr = imgs_lr.cuda
                imgs_hr = imgs_hr.cuda


            # keep track of prepare time
            prepare_time = start_time-time.time()

            #train generator  
            optimizer_G.zero_grad()

            gen_img = Generator(imgs_lr)
            
            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_img)

            adverserial_loss = adverserial_criterion(pred_fake - pred_real.mean(0, keepdim=True),valid)

            perceptual_loss = perceptual_criterion(gen_img, imgs_hr)

            gen_features = feature_extractor(gen_img)
            real_features = feature_extractor(imgs_hr).detach()
            content_loss = content_criterion(gen_features, real_features)

            Generatorloss = perceptual_loss * opt.perceplambda + content_loss * opt.contlambda + adverserial_loss * opt.advlambda
            epoch_loss += Generatorloss.data
            Generatorloss.backward()
            optimizer_G.step()

            # train Discriminator

            optimizer_D.zero_grad()
            
            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_img)

            adverserial_loss_rf = adverserial_criterion(pred_real - pred_fake.mean(0, keepdim=True),valid)
            adverserial_loss_fr = adverserial_criterion(pred_fake - pred_real.mean(0, keepdim=True),valid)
            discriminatorloss = (adverserial_loss_rf + adverserial_loss_fr) / 2
            
            epoch_loss += discriminatorloss.data
            discriminatorloss.backward()
            optimizer_D.step()


            # update tensorboard
            writer.add_scalar()

            #compute time and compute efficiency and print information
            process_time = start_time-time.time()-prepare_time
            #pbar.set_description("Compute efficiency. {:.2f}, epoch: {}/{}".format(process_time/(process_time+prepare_time),epoch, opt.epoch))
            start_time = time.time()
            print("===> Epoch {} Complete: Avg. loss: {:.4f}".format(epoch, epoch_loss / len(dataloader)))

        def print_network(net):
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            print(net)
            print('Total number of parameters: %d' % num_params)

        def checkpointG(epoch):
            model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_'+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
            torch.save(Generator.state_dict(), model_out_path)
            print("Checkpoint saved to {}".format(model_out_path))

        def checkpointD(epoch):
            model_discriminator_out_path = opt.disc_save_folder+str(opt.upscale_factor)+'x_'+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
            torch.save(discriminator.state_dict(), model_discriminator_out_path)
            print("Checkpoint saved to {}".format(model_discriminator_out_path))

    

        print('===> Building Model ', opt.model_type)
        if opt.model_type == 'ESRGANplus':
            Generator = ESRGANplus()
            discriminator = Discriminator()

        print('----------------Network architecture----------------')
        print_network(ESRGANplus)
        print('----------------------------------------------------')

        for epoch in range(opt.start_epoch, opt.nEpochs + 1):

            # learning rate is delayed
            if (epoch+1) % opt.sample_interval == 0:
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] /= 10.0
                print('Learning rate decay: lr={}'.format(optimizer_G.param_groups[0]['lr']))
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] /= 10.0
                print('Learning rate decay: lr={}'.format(optimizer_D.param_groups[0]['lr']))

            if (epoch+1) % (opt.snapshots) == 0:
                checkpointG(epoch)