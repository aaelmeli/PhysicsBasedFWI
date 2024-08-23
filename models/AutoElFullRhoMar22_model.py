import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import sys
# sys.path.append('/disk/student/adhara/Fall2021/deepwave/.')
# sys.path.append('/disk/student/adhara/Fall2021/deepwave/deepwave/.')
sys.path.append('./functions/.')
import deepwave
import multiprocessing
from joblib import Parallel, delayed
#import ray
import time
import os
#import scipy
import torchgeometry as tgm
from seisgan.optimizers import MALA, SGHMC
from LBFGS import FullBatchLBFGS
#import bruges as bg
#from vgg import Vgg16

#from skimage import metrics


class AutoElFullRhoMar22Model(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='AutoElFullRhoMar22',
                            dataset_mode='unalignedVelABCDEl', ngf='32')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float,
                                default=1.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        print("number of cuda devices:", torch.cuda.device_count())

        # torch.cuda.set_device(2)
        #del self.device
        # print(self.device)
        # Start Ray.
        #os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4,5,6,7"
        #ray.init(num_cpus=48,num_gpus=7)

        self.device1 = torch.device('cuda:{}'.format(
             self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_MSE', 'M_MSE', 'V_MSE', 'K_MSE', 'VP', 'VS', 'RHO','L_MSE']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_Vp','real_Vp','fake_Vs', 'real_Vs','fake_Rho','real_Rho']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer_G)
            self.criterionMSE = torch.nn.MSELoss(reduction='sum')
        else:
            print("----test data----")
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss(reduction='sum')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device1)
        print ("shape of self.real_A =",self.real_A.shape)
        self.real_B = input['B' if AtoB else 'A'].to(self.device1)
        print ("shape of self.real_B =",self.real_B.shape)
        print("max of vp=",torch.max(self.real_B[:,0,:,:]))
        self.real_C = input['C'].to(self.device1)
        self.real_D = input['D'].to(self.device1)  
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def forward2(self, epoch1, lstart, freq):
        if (epoch1 == 1):
            self.latent = torch.ones(1,1,1,1)
        [self.fake_Vp,self.fake_Vs,self.fake_Rho,self.grad,self.latent,self.vp_grad,self.vs_grad,self.rho_grad,self.loss_D_MSE] = self.netG(self.real_B,self.real_A,lstart,epoch1,self.latent,self.real_C,self.real_D,freq)  # G(A)
        #self.loss_D_MSE = self.loss_D_MSE*(10**30)
        self.real_Vp = torch.unsqueeze(self.real_B[:,0,:,:],1)
        self.real_Vs = torch.unsqueeze(self.real_B[:,1,:,:],1)
        self.real_Rho = torch.unsqueeze(self.real_B[:,2,:,:],1)
        
        self.low_Vp = torch.unsqueeze(self.real_C[:,0,:,:],1)
        self.low_Vs = torch.unsqueeze(self.real_C[:,1,:,:],1)
        self.low_Rho = torch.unsqueeze(self.real_C[:,2,:,:],1)

        self.fake_B = torch.cat((self.fake_Vp,self.fake_Vs,self.fake_Rho),axis=1)
        
        self.vp_grad = torch.unsqueeze(self.vp_grad,0)
        self.vs_grad = torch.unsqueeze(self.vs_grad,0)
        self.rho_grad = torch.unsqueeze(self.rho_grad,0)
        
        self.grad = torch.cat((self.vp_grad,self.vs_grad,self.rho_grad),dim=0)
        self.grad = torch.unsqueeze(self.grad,0)
        return self.loss_D_MSE

    def forward(self,epoch1,lstart,freq,idx,it):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if (epoch1 == 1):
            self.latent = torch.ones(1,1,1,1)
        [self.fake_Vp,self.fake_Vs,self.fake_Rho, self.grad,self.latent,self.vp_grad,self.vs_grad,self.rho_grad,self.loss_D_MSE] = self.netG(self.real_B,self.real_A,lstart,epoch1,self.latent,self.real_C,self.real_D,freq,idx,it)  # G(A)
        self.real_Vp = torch.unsqueeze(self.real_B[:,0,:,:],1)
        self.real_Vs = torch.unsqueeze(self.real_B[:,1,:,:],1)
        self.real_Rho = torch.unsqueeze(self.real_B[:,2,:,:],1)
        
        self.low_Vp = torch.unsqueeze(self.real_C[:,0,:,:],1)
        self.low_Vs = torch.unsqueeze(self.real_C[:,1,:,:],1)
        self.low_Rho = torch.unsqueeze(self.real_C[:,2,:,:],1)
        
        self.fake_Rho = self.fake_Rho/1
        self.fake_Vp = self.fake_Vp
        self.fake_Vs = self.fake_Vs
        self.fake_B = torch.cat((self.fake_Vp,self.fake_Vs,self.fake_Rho),axis=1)
        
        self.vp_grad = torch.unsqueeze(self.vp_grad,0)
        self.vs_grad = torch.unsqueeze(self.vs_grad,0)
        self.rho_grad = torch.unsqueeze(self.rho_grad,0)
        
        self.grad = torch.cat((self.vp_grad,self.vs_grad,self.rho_grad),dim=0)
        self.grad = torch.unsqueeze(self.grad,0)
           

    def forwardT(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        False_lstart = 1
        False_epoch = -1
        freq = 3

        self.latentT = torch.ones(1,1,1,1)
        num_shots = 35 #Abdo:: this needs to be changed
        idx = torch.randperm(num_shots)
        [self.VpT,self.VsT,self.RhoT,self.gradT,self.latentT,_,_,_,_] = self.netG(self.real_B,self.real_A,False_lstart,False_epoch,self.latentT,self.real_C,self.real_D,freq,idx,1)  # G(A)
        self.fake_BT = torch.cat((self.VpT,self.VsT,self.RhoT),axis=1)


        

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # Second, G(A) = B

        diff_size = self.real_B.size()
        self.loss_M_MSE = (self.criterionMSE(self.fake_B, self.real_B)) * \
            100/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        self.loss_D_MSE = 0.0
        # combine loss and calculate gradients
        self.loss_G = self.loss_M_MSE
        self.loss_G.backward()
    
    def backward_GKL(self,epoch1):
        """Calculate MSE loss along with KL divergence"""
                # First, G(A) should fake the discriminator
        # Second, G(A) = B
        #print("real B shape")
        diff_size = self.real_B.size()
        self.loss_M_MSE = (self.criterionMSE(self.fake_B, self.real_B)) * \
            100/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        self.loss_D_MSE = 0.0
        # combine loss and calculate gradients
        self.loss_G = self.loss_M_MSE
        self.loss_G.backward()
        
        
    def backward_G11(self, epoch1, batch, lstart, initerror, currenterror, idx):
            
        """Calculate GAN and L1 loss for the generator"""
        self.loss_M1_MSE = 0.0
        self.loss_K_MSE = 0.0
        
        self.loss_M_MSE = self.criterionMSE(self.real_Vp, self.fake_Vp) + self.criterionMSE(self.real_Vs, self.fake_Vs) + self.criterionMSE(self.real_Rho, self.fake_Rho)
        self.loss_VP = self.criterionMSE(self.real_Vp, self.fake_Vp)
        self.loss_VS = self.criterionMSE(self.real_Vs, self.fake_Vs)
        self.loss_RHO = self.criterionMSE(self.real_Rho, self.fake_Rho)
        
        self.loss_L_MSE = self.criterionMSE(self.low_Vp, self.fake_Vp) + self.criterionMSE(self.low_Vs, self.fake_Vs) + self.criterionMSE(self.low_Rho, self.fake_Rho)
        
        self.loss_VPLOW = self.criterionMSE(self.real_Vp, self.low_Vp)
        self.loss_VSLOW = self.criterionMSE(self.low_Vs, self.fake_Vs)
        self.loss_RHOLOW = self.criterionMSE(self.low_Rho, self.fake_Rho)

        self.loss_D_MSE = self.loss_D_MSE*(10**15) #this is just to visualize the data loss but we dont backpropagate on this, we backpropagate the gradient instead.

        print("loss_M_MSE :", self.loss_M_MSE)
        print("loss_D_MSE :", self.loss_D_MSE)

        lambda1 = 1
        lambda2 = 0
     
        self.loss_G = lambda1 * self.loss_L_MSE 
        ####self.loss_G = lambda2 * self.loss_M1_MSE
        
        if (epoch1>lstart):
            print("2nd epoch1 :", epoch1)
            #self.loss_G.backward(retain_graph=True)
            #self.grad = self.grad*(10**7)   #####(10**5) works for marmousi model
     
            self.vp_grad = torch.unsqueeze(self.vp_grad,0)*(10**12)
            self.vp_grad = self.vp_grad.cuda(self.fake_Vp.get_device())
            self.fake_Vp.backward(self.vp_grad,retain_graph=True)
            
            self.vs_grad = torch.unsqueeze(self.vs_grad,0)*(10**10)
            self.vs_grad = self.vs_grad.cuda(self.fake_Vs.get_device())
            self.fake_Vs.backward(self.vs_grad, retain_graph=True)

            self.rho_grad = torch.unsqueeze(self.rho_grad,0)*(10**8)
            self.rho_grad = self.rho_grad.cuda(self.fake_Rho.get_device())
            self.fake_Rho.backward(self.rho_grad)


        
    def closure(self,epoch,lstart,batch,freq):
        loss = self.forward2(epoch, lstart, freq)
        loss = loss
        loss = np.array(loss)
        loss = loss.astype(float)
        self.optimizer_G.zero_grad()
        self.backward_G11(epoch,batch,lstart)
        loss1 = torch.from_numpy(loss)
        loss1 = loss1.float()
        loss1 = loss1.to(self.device1)
        return loss1


    def optimize_parameters(self, epoch, batch, lstart, freq, initerror, currenterror):
        num_shots = 46
        idx = np.random.permutation(num_shots)
        num_batches = 1
        for it in range(num_batches):
            self.forward(epoch,lstart,freq,idx,it)                   # compute fake images: G(A)
            # update G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G11(epoch,batch,lstart,initerror,currenterror,idx)   
            self.optimizer_G.step()

    def compute_loss_only(self):
        diff_size = self.real_B.size()
        lossMSE = self.criterionMSE(
            self.fake_BT, self.real_B[:,0:3,:,:])/(diff_size[0]*diff_size[1]*diff_size[2]*diff_size[3])
        self.loss_V_MSE = lossMSE
        print("Loss RMSE : "+str(lossMSE.cpu().float().numpy()))
        pass

    def update_epoch(self, epoch):
        # update epoch number
        self.epoch1 = epoch
        print("epoch numbers : "+str(self.epoch1))



        
