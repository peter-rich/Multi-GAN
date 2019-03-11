import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

std = 0.1

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A_En = networks.define_GEn(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_A_Dn = networks.define_GDn(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.netG_B_En = networks.define_GEn(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B_Dn = networks.define_GDn(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A_En, 'G_A_En', which_epoch)
            self.load_network(self.netG_A_Dn, 'G_A_Dn', which_epoch)
            self.load_network(self.netG_B_En, 'G_B_En', which_epoch)
            self.load_network(self.netG_B_Dn, 'G_B_Dn', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionSelf = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionConsistence = mse_loss

            # initialize optimizers
            # the chain become longer as the decoder and encoder push into the list
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A_En.parameters(),self.netG_A_Dn.parameters(), self.netG_B_En.parameters(), self.netG_B_Dn.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A_En)#### Add
        networks.print_network(self.netG_A_Dn)#### Add
        networks.print_network(self.netG_B_En)#### Add
        networks.print_network(self.netG_B_Dn)#### Add
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        middle_AB = self.netG_A_En(real_A)
        fake_B = self.netG_A_Dn(middle_AB)
        self_A = self.netG_B_Dn(middle_AB)

        middle_BA = self.netG_B_En(fake_B)
        self.rec_A = self.netG_B_Dn(middle_BA).data
        self.fake_B = fake_B.data
        self.self_A = self_A.data

        real_B = Variable(self.input_B, volatile=True)
        middle_BA = self.netG_B_En(real_B)
        fake_A = self.netG_B_Dn(middle_BA)
        self_B = self.netG_A_Dn(middle_BA)

        middle_AB = self.netG_A_En(fake_A)
        self.rec_B = self.netG_A_Dn(middle_AB).data
        self.fake_A = fake_A.data
        self.self_B = self_B.data
    
    def test_general(self):
        real_A = Variable(self.input_A, volatile=True)
        middle_AB = self.netG_A_En(real_A)
        fake_B = self.netG_A_Dn(middle_AB)
        middle_BA = self.netG_B_En(fake_B)
        self.fake_B = fake_B.data
        self_A = self.netG_B_Dn(middle_AB)

        t = torch.cuda.FloatTensor(1,256,64,64).zero_()
        d = torch.cuda.FloatTensor(1,256,64,64).fill_(std)
        m = torch.distributions.Normal(t,d)

        noise = Variable(m.sample())
        #print(noise)
        
        middle_general_1 = middle_AB + noise

        noise = Variable(m.sample())
        middle_general_2 = middle_AB + noise
        

        self_A_1 = self.netG_B_Dn(middle_general_1)
        fake_B_1 = self.netG_A_Dn(middle_general_1)
        self_A_2 = self.netG_B_Dn(middle_general_2)
        fake_B_2 = self.netG_A_Dn(middle_general_2)        
        
        self.fake_B = fake_B.data
        self.rec_A = self.netG_B_Dn(middle_general_1).data
        
        self.self_A = self_A.data
        self.self_A_1 = self_A_1.data
        self.fake_B_1 = fake_B_1.data
        self.self_A_2 = self_A_2.data
        self.fake_B_2 = fake_B_2.data
        
        real_B = Variable(self.input_B, volatile=True)
        middle_BA = self.netG_B_En(real_B)
        fake_A = self.netG_B_Dn(middle_BA)
        middle_AB = self.netG_A_En(fake_A)
        self_B = self.netG_A_Dn(middle_BA)
        self.fake_A = fake_A.data

        noise = Variable(m.sample())
        middle_general_1 = middle_BA + noise

        noise = Variable(m.sample())
        middle_general_2 = middle_BA + noise

        self_B_1 = self.netG_A_Dn(middle_general_1)
        fake_A_1 = self.netG_B_Dn(middle_general_1)
        self_B_2 = self.netG_A_Dn(middle_general_2)
        fake_A_2 = self.netG_B_Dn(middle_general_2)

        self.self_B_1 = self_B_1.data
        self.fake_A_1 = fake_A_1.data
        self.self_B_2 = self_B_2.data
        self.fake_A_2 = fake_A_2.data

        self.rec_B = self.netG_A_Dn(middle_general_2).data
        self.fake_A = fake_A.data
        self.self_B = self_B.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        general_A_B = self.fake_B_pool.query(self.general_A_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        loss_D_A_noise = self.backward_D_basic(self.netD_A, self.real_B, general_A_B)

        self.loss_D_A = loss_D_A.item() + loss_D_A_noise.item()


    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        general_B_A = self.fake_A_pool.query(self.general_B_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        loss_D_B_noise = self.backward_D_basic(self.netD_B, self.real_A, general_B_A)

        self.loss_D_B = loss_D_B.item() + loss_D_B_noise.item()

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            
            self.middle_AB = self.netG_A_En(self.real_B)
            
            idt_A = self.netG_A_Dn(self.middle_AB)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.

            self.middle_BA = self.netG_B_En(self.real_B)
            
            idt_B = self.netG_B_Dn(self.middle_BA)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.item()
            self.loss_idt_B = loss_idt_B.item()
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        #
        #
        #  change a lot
        ###
        # Initialize a  random noise
        t = torch.cuda.FloatTensor(1,256,64,64).zero_()
        d = torch.cuda.FloatTensor(1,256,64,64).fill_(std)
        m = torch.distributions.Normal(t,d)

        

        #  GAN loss D_A(G_A(A))
        middle_AB = self.netG_A_En(self.real_A)
        fake_B = self.netG_A_Dn(middle_AB)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        
        #realA + noise -> B
        noise = Variable(m.sample())
        middle_noise_A_B = middle_AB + noise
        general_A_B = self.netG_A_Dn(middle_noise_A_B)
        pred_fake = self.netD_A(general_A_B)
        loss_G_A_nosie = self.criterionGAN(pred_fake, True)

        #realA + noise -> A
        #noise = Variable(m.sample())
        #middle_noise_A_A = middle_AB + noise
        #general_A_A = self.netG_B_Dn(middle_noise_A_A)


        # Forward cycle loss
        middle_BA = self.netG_B_En(fake_B)
        rec_A = self.netG_B_Dn(middle_BA)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        consistency_loss_AB = self.criterionConsistence(middle_AB,middle_BA)*lambda_A
        
        self_A = self.netG_B_Dn(middle_AB) 
        loss_self_A = self.criterionSelf (self_A, self.real_A) * lambda_A
        
        # middle_noise and the general loss
        middle_general_A_B = self.netG_B_En(general_A_B)
        #middle_general_A_A = self.netG_A_En(general_A_A)

        consistency_noise_loss_AB = self.criterionConsistence(middle_noise_A_B,middle_general_A_B)*lambda_A
        #consistency_noise_loss_AA = self.criterionConsistence(middle_noise_A_A,middle_general_A_A)*lambda_A
        ####################################################################################################
        # GAN loss D_B(G_B(B))
        middle_BA = self.netG_B_En(self.real_B)
        fake_A = self.netG_B_Dn(middle_BA)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        #realB + noise -> A
        noise = Variable(m.sample())
        middle_noise_B_A = middle_BA + noise
        general_B_A = self.netG_B_Dn(middle_noise_B_A)
        pred_fake = self.netD_B(general_B_A)
        loss_G_B_noise = self.criterionGAN(pred_fake, True)

        #realB + noise -> B
        #noise = Variable(m.sample())
        #middle_noise_B_B = middle_BA + noise
        #general_B_B = self.netG_A_Dn(middle_noise_B_B)


        # Backward cycle loss
        middle_AB = self.netG_A_En(fake_A)
        rec_B = self.netG_A_Dn(middle_AB)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        consistency_loss_BA = self.criterionConsistence(middle_BA,middle_AB)*lambda_B
        
        self_B = self.netG_B_Dn(middle_BA)
        loss_self_B = self.criterionSelf (self_B, self.real_B) * lambda_B
        
        
        # middle_noise and the general loss

        middle_general_B_A = self.netG_B_En(general_B_A)
        #middle_general_B_B = self.netG_A_En(general_B_B)

        consistency_noise_loss_BA = self.criterionConsistence(middle_noise_B_A,middle_general_B_A)*lambda_B
        #consistency_noise_loss_BB = self.criterionConsistence(middle_noise_B_B,middle_general_B_B)*lambda_B


        #
        #
        # combined loss    I have added four Loss into the function which is so quite easy, that's all the  thing i have now in the whole method, haha, haha
        #  
        #        Zhanfu Yang    865031716@qq.com
        #        2018.3.9  21:24
        #
        para = 0.5
        loss_G = loss_G_A + loss_G_B + para*(loss_G_A_nosie + loss_G_B_noise) + loss_idt_A + loss_idt_B + \
        (loss_cycle_A + loss_cycle_B + loss_self_A + loss_self_B) + \
        consistency_loss_AB + consistency_loss_BA + para*(consistency_noise_loss_AB + consistency_noise_loss_BA)
        
        loss_G.backward()
        
        self.general_A_B = general_A_B.data
        self.general_B_A = general_B_A.data

        self.self_A = self_A.data
        self.self_B = self_B.data
        self.loss_G = loss_G
        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.item()
        self.loss_G_B = loss_G_B.item()
        self.loss_cycle_A = loss_cycle_A.item()
        self.loss_cycle_B = loss_cycle_B.item()
        self.consistency_noise_loss_AB = consistency_noise_loss_AB.item()
        self.consistency_noise_loss_BA = consistency_noise_loss_BA.item()
        self.loss_self_A = loss_self_A.item()
        self.loss_self_B = loss_self_B.item()
        self.consistency_loss_AB = consistency_loss_AB.item()
        self.consistency_loss_BA = consistency_loss_BA.item()

    
    def get_current_loss(self):
        return self.loss_G
    
    def get_current_loss_G_A(self):
        return self.loss_G_A

    def get_current_loss_G_B(self):
        return self.loss_G_B

    def get_current_loss_Cyc_A(self):
        return self.loss_cycle_A

    def get_current_loss_Cyc_B(self):
        return self.loss_cycle_B

    def get_current_loss_Cos_A(self):
        return self.consistency_loss_AB

    def get_current_loss_Cos_B(self):
        return self.consistency_loss_BA

    def get_current_loss_Self_A(self):
        return self.loss_self_A

    def get_current_loss_Self_B(self):
        return self.loss_self_B

    def get_current_loss_Idt_A(self):
        return self.loss_idt_A

    def get_current_loss_Idt_B(self):
        return self.loss_idt_B
    
    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A), 
                                  ('Cos_A', self.consistency_loss_AB), ('Self_A', self.loss_self_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B', self.loss_cycle_B),
                                  ('Cos_B', self.consistency_loss_BA), ('Self_B', self.loss_self_B)])
                                  
        if self.opt.lambda_identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        self_A = util.tensor2im(self.self_A)
        self_B = util.tensor2im(self.self_B)

        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),('self_A',self_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),('self_B',self_B)])
        if self.opt.isTrain and self.opt.lambda_identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def get_current_visuals_general(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        self_A = util.tensor2im(self.self_A)
        self_B = util.tensor2im(self.self_B)
        self_A_1 = util.tensor2im(self.self_A_1)
        self_A_2 = util.tensor2im(self.self_A_2)
        self_B_1 = util.tensor2im(self.self_B_1)
        self_B_2 = util.tensor2im(self.self_B_2)
        fake_A_1 = util.tensor2im(self.fake_A_1)
        fake_A_2 = util.tensor2im(self.fake_A_2)
        fake_B_1 = util.tensor2im(self.fake_B_1)
        fake_B_2 = util.tensor2im(self.fake_B_2)

        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),('self_A',self_A), ('self_A_1',self_A_1), ('self_A_2',self_A_2), ('fake_B_1',fake_B_1), ('fake_B_2',fake_B_2),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),('self_B',self_B), ('self_B_1',self_B_1), ('self_B_2',self_B_2), ('fake_A_1',fake_A_1), ('fake_A_2',fake_A_2)])
        if self.opt.isTrain and self.opt.lambda_identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A_En, 'G_A_En', label, self.gpu_ids)
        self.save_network(self.netG_A_Dn, 'G_A_Dn', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B_En, 'G_B_En', label, self.gpu_ids)
        self.save_network(self.netG_B_Dn, 'G_B_Dn', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
