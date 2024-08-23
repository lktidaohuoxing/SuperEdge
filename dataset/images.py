#-*-coding:utf8-*-
import os
import glob
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.params import dict_update
from dataset.utils.homographic_augmentation import homographic_aug_pipline
from dataset.utils.photometric_augmentation import PhotoAugmentor
from utils.keypoint_op import compute_keypoint_map
from dataset.utils.photometric_augmentation import *


class IMGDataset(torch.utils.data.Dataset):
    def __init__(self, config,model_config, is_train, device='cpu'):

        super(IMGDataset, self).__init__()
        self.device = device
        self.is_train = is_train
        self.resize = tuple(config['resize'])
        self.photo_augmentor = PhotoAugmentor(config['augmentation']['photometric'])
        # load config
        self.config = config #dict_update(getattr(self, 'default_config', {}), config)
        self.model_config = model_config
        # get images
        if self.is_train:
            self.samples = self._init_data(config['image_train_path'], config['label_train_path'])
        else:
            self.samples = self._init_data(config['image_test_path'], config['label_test_path'])


    def _init_data(self, image_path, label_path=None):
        ##
        if not isinstance(image_path,list):
            image_paths, label_paths = [image_path,], [label_path,]
        else:
            image_paths, label_paths = image_path, label_path

        image_types = ['jpg','jpeg','bmp','png']
        samples = []
        for im_path, lb_path in zip(image_paths, label_paths):
            for it in image_types:
                temp_im = glob.glob(os.path.join(im_path, '*.{}'.format(it)))
                if lb_path is not None:
                    temp_lb = [os.path.join(lb_path, os.path.basename(imp)+'.npy') for imp in temp_im]
                else:
                    temp_lb = [None,]*len(temp_im)
                temp = [{'image':imp, 'label':lb} for imp, lb in zip(temp_im, temp_lb)]
                samples += temp
        ##
        return samples

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        '''load raw data'''
        data_path = self.samples[idx]#raw image path of processed image and point path
        img = cv2.imread(data_path['image'], 0)#Gray image
        pri_W,pri_H = img.shape[0], img.shape[1] # image size
        W, H = int(pri_W/8) * 8,int(pri_H/8)*8
        img = cv2.resize(img, (H,W))
        img_name = data_path['image'].rsplit('/', 1)[-1][:-4]
        pts = None if data_path['label'] is None else np.load(data_path['label'])#N*2,yx
        if pts is not None:
            pts = pts.astype(np.float32)

        #superedge
        if( data_path['label'] != None):
            index = data_path['label'].rfind("/") 
            new_path = data_path['label'][:index + 1] + "obj" + data_path['label'][index + 1:] 
        else:
            new_path = None

        if self.model_config['name'] != 'superedgev1':    
            kp_pts = None if data_path['label'] is None else np.load(new_path) 
        else:
            kp_pts = None
            
        # init data dict
        img_tensor = torch.as_tensor(img.copy(), dtype=torch.float, device=self.device)
        kpts_tensor = None if pts is None else torch.as_tensor(pts, device=self.device)

        #superedge
        kp_pts = None if kp_pts is None else  torch.as_tensor(kp_pts, dtype=torch.float32, device=self.device)#N2
        kp_kp_map = None if kp_pts is None else compute_keypoint_map(kp_pts, img_tensor.shape, device=self.device)#HW

        kpts_map = None if pts is None else compute_keypoint_map(kpts_tensor, img.shape, device=self.device)
        valid_mask = torch.ones(img.shape, device=self.device)

        data = {'raw':{'img': img_tensor,
                        'pri_H':pri_H,
                        'pri_W':pri_W,
                       'kpts': kpts_tensor,
                       'kp_kpts' : kp_pts, #lk, N,2
                       'kpts_map':kpts_map,
                       'kp_kpts_map':kp_kp_map, #h,W
                       'mask':valid_mask,
                       'img_name':(img_name[:])
                       },
                'warp':None,
                'homography':torch.eye(3,device=self.device)}
        data['warp'] = deepcopy(data['raw'])

        ##
        if self.is_train:
            photo_enable = self.config['augmentation']['photometric']['train_enable']
            homo_enable = self.config['augmentation']['homographic']['train_enable']
        else:
            photo_enable = self.config['augmentation']['photometric']['test_enable']
            homo_enable = self.config['augmentation']['homographic']['test_enable']


        if homo_enable and data['raw']['kpts'] is not None:#homographic augmentation
            # return dict{warp:{img:[H,W], point:[N,2], valid_mask:[H,W], homography: [3,3]; tensors}}
            data_homo = homographic_aug_pipline(data['warp']['img'],
                                                data['warp']['kpts'],
                                                data['warp']['kp_kpts'],
                                                self.config['augmentation']['homographic'],
                                                device=self.device)
            data.update(data_homo)

        if photo_enable:
            photo_img = data['warp']['img'].cpu().numpy().round().astype(np.uint8)
            photo_img = self.photo_augmentor(photo_img)
            data['warp']['img'] = torch.as_tensor(photo_img, dtype=torch.float,device=self.device)

        ##normalize
        data['raw']['img'] = data['raw']['img']/255.
        data['warp']['img'] = data['warp']['img']/255.

        return data#img:HW, kpts:N2,kp_kpts_map, kpts_map:HW, valid_mask:HW, homography:HW

    def batch_collator(self, samples):
        """
        :param samples:a list, each element is a dict with keys
        like `img`, `img_name`, `kpts`, `kpts_map`,
        `valid_mask`, `homography`...
        img:H*W, kpts:N*2, kpts_map:HW, valid_mask:HW, homography:HW
        :return:
        """
        if(self.model_config['name'] == 'superedgev1'):
            sub_data = {'img': [], 'kpts_map': [],'mask': [],'img_name':[],'pri_H':[],'pri_W':[]}
        else:
            sub_data = {'img': [], 'kpts_map': [],'kp_kpts_map':[],'mask': [],'img_name':[],'pri_H':[],'pri_W':[]}#remove kpts
        batch = {'raw':sub_data, 'warp':deepcopy(sub_data), 'homography': []}

        for s in samples:
            batch['homography'].append(s['homography'])

            for k in sub_data:
                if k=='img':
                    batch['raw'][k].append(s['raw'][k].unsqueeze(dim=0))
                    if 'warp' in s:
                        batch['warp'][k].append(s['warp'][k].unsqueeze(dim=0))
                else:
                    batch['raw'][k].append(s['raw'][k])
                    if 'warp' in s:
                        s['warp']['img_name'] = s['raw']['img_name']
                        batch['warp'][k].append(s['warp'][k])
        ##
        batch['homography'] = torch.stack(batch['homography'])
        for k0 in ('raw','warp'):
            for k1 in sub_data:#`img`, `img_name`, `kpts`, `kpts_map`...
                if k1=='kpts'  or k1=='kp_kpts' or k1 == 'img_name' or k1 == 'pri_H' or k1 == 'pri_W':
                    continue

                batch[k0][k1] = None if batch[k0][k1][0] is None else torch.stack(batch[k0][k1])

        return batch


if __name__=='__main__':
    import yaml
    import matplotlib.pyplot as plt
    from dataset.utils.photometric_augmentation import *
    with open('../config/superpoint_train.yaml','r') as fin:
        config = yaml.load(fin)

    coco = COCODataset(config['data'],True)
    cdataloader = DataLoader(coco,collate_fn=coco.batch_collator,batch_size=1,shuffle=True)

    for i,d in enumerate(cdataloader):
        if i>=10:
            break
        img = (d['raw']['img']*255).cpu().numpy().squeeze().astype(np.int).astype(np.uint8)
        img_warp = (d['warp']['img']*255).cpu().numpy().squeeze().astype(np.int).astype(np.uint8)
        img = cv2.merge([img, img, img])
        img_warp = cv2.merge([img_warp, img_warp, img_warp])
        ##
        kpts = np.where(d['raw']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(np.int)
        for kp in kpts:
            cv2.circle(img, (kp[1], kp[0]), radius=3, color=(0,255,0))
        kpts = np.where(d['warp']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(np.int)
        for kp in kpts:
            cv2.circle(img_warp, (kp[1], kp[0]), radius=3, color=(0,255,0))

        mask = d['raw']['mask'].cpu().numpy().squeeze().astype(np.int).astype(np.uint8)*255
        warp_mask = d['warp']['mask'].cpu().numpy().squeeze().astype(np.int).astype(np.uint8)*255

        img = cv2.resize(img, (640,480))
        img_warp = cv2.resize(img_warp,(640,480))

        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.subplot(2,2,2)
        plt.imshow(mask)
        plt.subplot(2,2,3)
        plt.imshow(img_warp)
        plt.subplot(2,2,4)
        plt.imshow(warp_mask)
        plt.show()

    print('Done')