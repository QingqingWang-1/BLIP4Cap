import torch as torch
from torch.utils.data import DataLoader 
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
#from torchvision.transforms import RandomAugment
from data.transform import RandomAugment #self-defined
from data.coco_caption_dataset import coco_train, coco_caption_eval
from data.pretrain_dataset import pretrain_dataset

def create_sampler(datasets, shuffles, world_size, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=shuffle) #num_replicas is number of processes participating in distributed training, default as world_size; rank is the global_rank of current process
        samplers.append(sampler)
    return samplers

def create_dataset(dataset, config, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([                        
        transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        transforms.ToTensor(),
        normalize,
    ]) 
    transform_test = transforms.Compose([
    transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])
    if dataset == 'pretrain':
        dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)
        return dataset 
    if dataset == 'caption_coco':
        train_dataset = coco_train(transform_train, config['train_image_root'], config['train_ann_json'], prompt=config['prompt'])
        val_dataset = coco_caption_eval(transform_test, config['val_image_root'], config['val_ann_json'])
        test_dataset = coco_caption_eval(transform_test, config['val_image_root'], config['val_ann_json'])
        return train_dataset, val_dataset, test_dataset
    else:
        raise NotImplementedError(f'dataset type {dataset} is not implemented')

def create_loader(datasets, samplers, batch_sizes, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_sizes, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None) #sampler has shuffled for training
            drop_last = True 
        else:
            shuffle = False 
            drop_last = False 
        loader = DataLoader(dataset, batch_size=bs, num_workers=n_worker, pin_memory=True, sampler=sampler, shuffle=shuffle, 
                            collate_fn=collate_fn, drop_last=drop_last)
        loaders.append(loader)
    return loaders


