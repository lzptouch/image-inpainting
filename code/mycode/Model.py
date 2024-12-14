import os

from dataset import Dataset
from model_ import Genetator,Discriminator
import torch
import torch.optim as optim
import torch.nn as nn

from utils import imsave


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

        if config.MODE == 2:  # test mode
            self.test_dataset = Dataset(config, config.TEST_MASK_FLIST, augment=False, training=False)
        # else:
        #     self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
        #     self.vali_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False,training=False)
        #
        # if config.MODE == 2:
        #     self.test_loader = DataLoader(
        #         dataset=self.test_dataset,
        #         batch_size=1,
        #         shuffle=False
        #     )
        #
        # self.samples_path = os.path.join(config.PATH, 'samples')
        # self.results_path = os.path.join(config.PATH, 'result1_')
        # self.log_file = os.path.join(config.PATH, 'log_' + self.name + '.dat')

    def load(self):
        # 根据网络参数文件是否存在，决定加载参数
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])

            self.iteration = data['iteration']


        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])



    def cuda(self,*args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255
        img = img.permute(0, 2, 3, 1)
        return img.int()

class PasteModel(BaseModel):

    def __init__(self, config):
        super(PasteModel, self).__init__('PasteModel', config)
        self.config = config

        self.generator = Genetator().to(config.DEVICE)
        self.discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge').to(config.DEVICE)
        # if len(config.GPU) > 1:
        #     self.generator = nn.DataParallel(self.generator, config.GPU)
        #     self.discriminator = nn.DataParallel(self.discriminator, config.GPU)

        # self.l1loss = nn.L1Loss().to(config.DEVICE)
        # self.perceptual_loss = PerceptualLoss().to(config.DEVICE)
        # self.adversarial_loss = AdversarialLoss(type=config.GAN_LOSS).to(config.DEVICE)
        # self.style_loss = StyleLoss().to(config.DEVICE)
        #
        # self.auxloss = Auxloss().to(config.DEVICE)


        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )


    def process(self, images, masks):
        self.iteration += 1


        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()
        outputs, f32, f64, f128 = self(images, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss

        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, dis_real_feat = self.discriminator(dis_input_real)  # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)  # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)  # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)* self.config.ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        gen_restruction_loss = (self.l1loss(outputs, images)+self.l1loss(outputs*masks, images*masks))* self.config.RESTRUCTION_LOSS_WEIGHT

        gen_loss += gen_restruction_loss

        auxloss = self.auxloss(images,f32,f64,f128)*self.config.AUX_LOSS_WEIGHT
        gen_loss+=auxloss


        gen_content_loss = self.perceptual_loss(outputs, images)* self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_re", gen_restruction_loss.item()),
            ("l_aux",auxloss.item()),
            ("l_contend",gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs


    def forward(self, images, masks):
        images_masked = (images * (1 - masks).float())+masks
        inputs = images_masked

        output = self.postprocess(inputs)[0]
        imsave(output, "model.jpg")

        outputs, f32, f64, f128 = self.generator(inputs)
        return outputs,  f32, f64, f128

    def backward(self, gen_loss=None, dis_loss=None,gen_loss64=None, gen_loss128=None):

        if dis_loss is not None:
            dis_loss.backward()
        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()
        self.dis_optimizer.step()