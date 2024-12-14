import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import kpn.utils as kpn_utils
# import kpn.config as kpn_config
import torchvision
import lpips



class MISF():
    def __init__(self, config):
        self.config = config

        # if config.MODEL == 2:
        model_name = 'inpaint'


        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
            print('test dataset:'.format(len(self.test_dataset)))
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            # self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=False)
            # self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

            print('train dataset:{}'.format(len(self.train_dataset)))
            # print('eval dataset:{}'.format(len(self.val_dataset)))

        # self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = config.RESULTS


        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load(self.config.MODEL_LOAD)

        else:
            self.edge_model.load('EdgeModel')
            self.inpaint_model.load('InpaintingModel')

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()
        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        max_psnr = 0
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.edge_model.train()
                self.inpaint_model.train()

                images, images_gray, edges, masks = self.cuda(*items)

                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                    # metrics
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))
                    logs.append(('gen_loss', gen_loss.item()))

                    # backward
                    self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration


                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                    outputs_merged = (outputs * masks) + images * (1 - masks)

                    # # metrics
                    # psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    # mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    # logs.append(('psnr', psnr.item()))
                    # logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # inpaint with edge model
                elif model == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, edges, masks)
                        outputs = outputs * masks + edges * (1 - masks)
                    else:
                        outputs = edges

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # joint model
                else:
                    # train
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpaint_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs


                # sample
                # if iteration % self.config.TRAIN_SAMPLE_INTERVAL == 0:
                #     img_list2 = [images * (1 - masks), outputs_merged, outputs, images]
                #     name_list2 = ['in', 'pred_2', 'pre_1', 'gt']
                #     kpn_utils.save_sample_png(sample_folder=self.config.TRAIN_SAMPLE_SAVE,
                #                               sample_name='ite_{}_{}'.format(self.inpaint_model.iteration,
                #                                                              0), img_list=img_list2,
                #                               name_list=name_list2, pixel_max_cnt=255, height=-1,
                #                               width=-1)


                # save model at checkpoints
                if iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                # evaluate model at checkpoints
                # if iteration % self.config.EVAL_INTERVAL == 0:
                #     print('\nstart eval...\n')
                #     cur_psnr = self.eval()
                #     self.inpaint_model.iteration = iteration
                #
                #     if cur_psnr > max_psnr:
                #         max_psnr = cur_psnr
                #         self.save()
                #         print('---increase-iteration:{}'.format(iteration))

                print(logs)

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=False
        )

        model = self.config.MODEL

        self.edge_model.eval()
        self.inpaint_model.eval()

        psnr_all = []
        ssim_all = []
        l1_list = []
        lpips_list = []

        iteration = self.inpaint_model.iteration
        with torch.no_grad():
            for items in val_loader:
                images, images_gray, edges, masks = self.cuda(*items)

                # edge model
                if model == 1:

                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)
                    outputs = outputs * masks + edges * (1 - masks)

                    psnr, ssim = self.metric(edges, outputs)
                    psnr_all.append(psnr)
                    ssim_all.append(ssim)

                # inpaint model
                elif model == 2:
                    # eval
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                    outputs_merged = (outputs * masks) + images * (1 - masks)

                    psnr, ssim = self.metric(images, outputs_merged)
                    psnr_all.append(psnr)
                    ssim_all.append(ssim)

                    l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
                    l1_list.append(l1_loss)

                    pl = 1.0
                    lpips_list.append(pl)


                # inpaint with edge model
                elif model == 3:
                    # eval
                    outputs = self.edge_model(images_gray, edges, masks)
                    outputs = outputs * masks + edges * (1 - masks)

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))


                # joint model
                else:
                    # eval
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                # sample
                if len(psnr_all) % self.config.EVAL_SAMPLE_INTERVAL == 0:
                    img_list2 = [images * (1 - masks), outputs_merged, outputs, images]
                    name_list2 = ['in', 'pred2', 'pre1', 'gt']
                    kpn_utils.save_sample_png(sample_folder=self.config.EVAL_SAMPLE_SAVE,
                                          sample_name='ite_{}_{}'.format(iteration, len(psnr_all)), img_list=img_list2,
                                          name_list=name_list2, pixel_max_cnt=255, height=-1,
                                          width=-1)



                print('psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips{}/{}  {}/{}'.format(psnr, np.average(psnr_all),
                                                                                   ssim, np.average(ssim_all),
                                                                                   l1_loss, np.average(l1_list),
                                                                                   pl, np.average(lpips_list),
                                                                                   len(psnr_all), len(self.val_dataset)))

                if len(psnr_all) >= 1000:
                    break

            print('iteration:{} ave_psnr:{}  ave_ssim:{} ave_l1:{}  ave_lpips:{}'.format(
                iteration,
                np.average(psnr_all),
                np.average(ssim_all),
                np.average(l1_list),
                np.average(lpips_list)
            ))

            return np.average(psnr_all)

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )


        index = 0
        with torch.no_grad():
            for items in test_loader:
                images, images_gray, edges, masks = self.cuda(*items)
                # index += 1
                name = self.test_dataset.load_name(index)
                # edge model
                if model == 1:
                    outputs = self.edge_model(images_gray, edges, masks)
                    outputs_merged = (outputs * masks) + (edges * (1 - masks))

                # inpaint model
                elif model == 2:
                    outputs = self.inpaint_model(images, edges, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))
                    # outputs_merged = outputs

                # inpaint with edge model / joint model
                else:
                    edges = self.edge_model(images_gray, edges, masks).detach()
                    outputs = self.inpaint_model(images, edges, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))


                output = self.postprocess(outputs_merged)[0]
                path = os.path.join(self.results_path, name)
                print(index, name)

                imsave(output, path)
                index += 1
                if index >= 3000:
                    exit()




    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)

        return psnr, ssim