#-*-coding:utf8-*-
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import yaml
import argparse
from tqdm import tqdm
from dataset.images import IMGDataset
from dataset.synthetic_shapes import SyntheticShapes
from torch.utils.data import DataLoader
from model.superedgev1 import SuperEdgeV1
from model.superedge import SuperEdge

from solver.loss import loss_func


#training
def train_eval(model, dataloader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['solver']['base_lr'])

    try:
        # start training
        for epoch in range(config['solver']['epoch']):
            model.train()
            mean_loss = []
            for i, data in tqdm(enumerate(dataloader['train'])):
                prob, desc, prob_warp, desc_warp = None, None, None, None
                if config['model']['name']=='superedge' and config['data']['name']=='coco':
                    data['raw'] = data['warp']
                    data['warp'] = None

                raw_outputs = model(data['raw'])
                prob = raw_outputs 

                ##loss
                loss = loss_func(config['solver'],config['model'], data, prob,
                                 prob_warp, device)

                mean_loss.append(loss.item())
                #reset
                model.zero_grad()
                loss.backward()
                optimizer.step()

                if (i%50==0):
                    print('Epoch [{}/{}], Step [{}/{}], LR [{}], Loss: {:.3f}'
                          .format(epoch, config['solver']['epoch'], i, len(dataloader['train']),
                                  optimizer.state_dict()['param_groups'][0]['lr'], np.mean(mean_loss)))
                    mean_loss = []

                ##do evaluation
                save_iter = int( 0.5 * len(dataloader['train']))#half epoch
                if (i!= 0 and i%save_iter==0 ) or (i+1)==len(dataloader['train']):
                    model.eval()
                    eval_loss = do_eval(model, dataloader['test'], config, device)
                    model.train()

                    save_path = os.path.join(config['solver']['save_dir'],
                                             config['solver']['model_name'] + '_{}_{}.pth').format(round(eval_loss, 3),epoch )
                    torch.save(model.state_dict(), save_path)
                    print('Epoch [{}/{}], Step [{}/{}], Eval loss {:.3f}, Checkpoint saved to {}'
                          .format(epoch, config['solver']['epoch'], i, len(dataloader['train']), eval_loss, save_path))
                    mean_loss = []

    except KeyboardInterrupt:
        torch.save(model.state_dict(), "./export/key_interrupt_model.pth")

@torch.no_grad()
def do_eval(model, dataloader, config, device):
    mean_loss = []
    truncate_n = max(int(0.1 * len(dataloader)), 100)  # 0.1 of test dataset for eval

    for ind, data in tqdm(enumerate(dataloader)):
        if ind>truncate_n:
            break
        prob, desc, prob_warp, desc_warp = None, None, None, None
        if config['model']['name'] == 'superedge' and config['data']['name'] == 'coco':
            data['raw'] = data['warp']
            data['warp'] = None

        raw_outputs = model(data['raw'])
        prob = raw_outputs

        # compute loss
        loss = loss_func(config['solver'],config['model'], data, prob,
                                 prob_warp, device)


        mean_loss.append(loss.item())
    mean_loss = np.mean(mean_loss)

    return mean_loss



if __name__=='__main__':

    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()

    config_file = args.config
    assert (os.path.exists(config_file))
    ##
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)

    if not os.path.exists(config['solver']['save_dir']):
        os.makedirs(config['solver']['save_dir'])

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    data_loaders = None
    if config['data']['name'] == 'coco':
        datasets = {k: IMGDataset(config['data'],config['model'], is_train=True if k == 'train' else False, device=device)
                    for k in ['test', 'train']}
        data_loaders = {k: DataLoader(datasets[k],
                                      config['solver']['{}_batch_size'.format(k)],
                                      collate_fn=datasets[k].batch_collator,
                                      shuffle=True) for k in ['train', 'test']}
    elif config['data']['name'] == 'synthetic':
        datasets = {'train': SyntheticShapes(config['data'], task=['training', 'validation'], device=device),
                    'test': SyntheticShapes(config['data'], task=['test', ], device=device)}
        data_loaders = {'train': DataLoader(datasets['train'], batch_size=config['solver']['train_batch_size'], shuffle=True,
                                            collate_fn=datasets['train'].batch_collator),
                        'test': DataLoader(datasets['test'], batch_size=config['solver']['test_batch_size'], shuffle=True,
                                           collate_fn=datasets['test'].batch_collator)}
    ##Make model
    if config['model']['name'] == 'superedge':
        model = SuperEdge(config['model'], device=device, using_bn=config['model']['using_bn'])
    elif config['model']['name'] == 'superedgev1':
        model = SuperEdgeV1(config['model'], device=device,using_bn=config['model']['using_bn'])

    ##Load Pretrained Model
    if os.path.exists(config['model']['pretrained_model']):
        pre_model_dict = torch.load(config['model']['pretrained_model'])
        model_dict = model.state_dict()
        for k,v in pre_model_dict.items():
            if k in model_dict.keys() and v.shape==model_dict[k].shape:
                model_dict[k] = v
        model.load_state_dict(model_dict)
    model.to(device)
    train_eval(model, data_loaders, config)
    print('Done')
