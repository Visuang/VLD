
import torchvision.transforms as transforms
from data_loader.dataset import SYSUData, RegDBData, TestData, process_query_sysu, process_gallery_sysu, \
    process_test_regdb
from data_loader.processing import ChannelRandomErasing, ChannelAdapGray, ChannelExchange
from data_loader.sampler import GenIdx, IdentitySampler

import torch.utils.data as data
from data_loader.vcm_data_manager import VCM
from data_loader.vcm_dataloader import *
from data_loader.bupt_dataloader import get_dataloader


class Loader:

    def __init__(self, config):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((config.img_h, config.img_w)),
            transforms.ToTensor(),
            normalize])

        self.transform_color1_vcm  = transforms.Compose([
            transforms.Resize((config.img_h, config.img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((config.img_h, config.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)])

        self.transform_color2_vcm  = transforms.Compose([
            transforms.Resize((config.img_h, config.img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((config.img_h, config.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)])

        self.transform_thermal_vcm  = transforms.Compose([
            transforms.Resize((config.img_h, config.img_w)),
            transforms.Pad(10),
            transforms.RandomCrop((config.img_h, config.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])


        self.dataset = config.dataset
        self.img_w = config.img_w
        self.img_h = config.img_h
        self.num_pos = config.num_pos
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.seq_lenth = config.sequence_length
        self.test_batch = config.test_batch
        self.config = config
        self._loader()

    def _loader(self):
        if self.dataset == 'vcm':
            samples = VCM(self.config)
            self.color_pos, self.thermal_pos = GenIdx(samples.rgb_label, samples.ir_label)
            self.samples = samples

            self.query_loader = data.DataLoader(
                VideoDataset_test(samples.query, seq_len=self.seq_lenth, sample='video_test', transform=self.transform_test),
                batch_size=self.test_batch, shuffle=False, num_workers=self.num_workers)

            self.gallery_loader = data.DataLoader(
                VideoDataset_test(samples.gallery, seq_len=self.seq_lenth, sample='video_test', transform=self.transform_test),
                batch_size=self.test_batch, shuffle=False, num_workers=self.num_workers)

            # ----------------visible to infrared----------------
            self.query_loader_1 = data.DataLoader(
                VideoDataset_test(samples.query_1, seq_len=self.seq_lenth, sample='video_test', transform=self.transform_test),
                batch_size=self.test_batch, shuffle=False, num_workers=self.num_workers)

            self.gallery_loader_1 = data.DataLoader(
                VideoDataset_test(samples.gallery_1, seq_len=self.seq_lenth, sample='video_test', transform=self.transform_test),
                batch_size=self.test_batch, shuffle=False, num_workers=self.num_workers)

            self.nquery_1 = samples.num_query_tracklets_1
            self.ngall_1 = samples.num_gallery_tracklets_1

            self.n_class = samples.num_train_pids
            self.nquery = samples.num_query_tracklets
            self.ngall = samples.num_gallery_tracklets
        elif self.dataset == 'bupt':
            self.query_loader, _ = get_dataloader(self.config, 'query',transform=self.transform_test, show=False)
            self.gallery_loader, _ = get_dataloader(self.config, 'gallery', transform=self.transform_test, show=False)
            self.dataloader_train, self.class_num = get_dataloader(self.config, 'train',transform=self.transform_color1_vcm,transform2=self.transform_color2_vcm,transform3=self.transform_thermal_vcm, show=True)
            self.train_iter = IterLoader(self.dataloader_train)

    def get_train_loader(self):
        if self.config.dataset=='vcm':
            print('==> Preparing Data Loader...')
            sampler = IdentitySampler(self.samples.ir_label, self.samples.rgb_label, self.color_pos, self.thermal_pos, self.num_pos, self.batch_size)
            index1 = sampler.index1
            index2 = sampler.index2

            loader_batch = self.batch_size * self.num_pos

            train_loader = data.DataLoader(
                VideoDataset_train(self.samples.train_ir, self.samples.train_rgb, seq_len=self.seq_lenth, sample='video_train',
                                   transform1=self.transform_color1_vcm, transform2=self.transform_color2_vcm,
                                   transform3=self.transform_thermal_vcm, index1=index1, index2=index2),
                sampler=sampler,
                batch_size=loader_batch, num_workers=self.num_workers,
                drop_last=True,
            )
            train_iter = IterLoader(train_loader)
            return train_iter
        else:
            sampler = IdentitySampler(self.samples.train_color_label, self.samples.train_thermal_label, self.color_pos,
                                      self.thermal_pos, self.num_pos, int(self.batch_size / self.num_pos))
            self.samples.cIndex = sampler.index1
            self.samples.tIndex = sampler.index2
            train_loader = data.DataLoader(self.samples, batch_size=self.batch_size,
                                           sampler=sampler, num_workers=self.num_workers, drop_last=True)
            return train_loader


class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)