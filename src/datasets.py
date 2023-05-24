import warnings

from util import misc


warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, copy, torch, random, os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import scipy.io
from torchvision.datasets import ImageFolder
from timm.data import create_transform
from sampler import RecallKDistributedSampler

def give_dataloaders(dataset, opt, config):
    if opt.dataset=='vehicle_id':
        datasets = give_VehicleID_datasets(opt)
    elif opt.dataset=='Inaturalist':
        datasets = give_inaturalist_datasets(opt,config)
    elif opt.dataset == 'cars196':
        datasets = give_cars196_datasets(opt)
    elif opt.dataset=='sop':
        datasets = give_sop_datasets(opt)
    elif opt.dataset=='cub':
        datasets = give_cub_datasets(opt)
    else:
        raise Exception('No Dataset >{}< available!'.format(dataset))
    
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    dataloaders = {}
    for key, dataset in datasets.items():
        if isinstance(dataset, TrainDatasetrsk) and key == 'training':
            sampler = RecallKDistributedSampler(dataset=dataset, num_replicas=num_tasks, rank=global_rank, batchsize=opt.bs, shuffle=False, drop_last=True)
            dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=int(opt.bs/opt.world_size),
                    num_workers=opt.kernels, sampler=sampler, pin_memory=True, drop_last=True)
            # dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=opt.bs,
            #                                                num_workers=opt.kernels,
            #                                                sampler=torch.utils.data.SequentialSampler(dataset),
            #                                                pin_memory=True, drop_last=True)
        else:
            is_val = dataset.is_validation
            if key == 'training':
                dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, 
                        num_workers=opt.kernels, shuffle=not is_val, pin_memory=True, drop_last=not is_val)
            else: 
                dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=opt.bs_base,
                        num_workers=6, shuffle=not is_val, pin_memory=True, drop_last=not is_val)
    return dataloaders


def give_cars196_datasets(opt):
        train_image_dict, test_image_dict = {}, {}
        all_image_dict = {}
        data = scipy.io.loadmat(os.path.join(opt.source_path, 'cars_annos.mat'))['annotations'][0]
        for entry in data:
                data_set = entry[6][0][0]
                im_path = os.path.join(opt.source_path, entry[0][0])
                class_id = entry[5][0][0]
                if class_id not in all_image_dict.keys():
                        all_image_dict[class_id] = []
                all_image_dict[class_id].append(im_path)
        train_classes = list(all_image_dict.keys())[:98]
        val_classes = list(all_image_dict.keys())[98:]
        for given_class in train_classes:
                train_image_dict[given_class] = all_image_dict[given_class]
        for given_class in val_classes:
                test_image_dict[given_class] = all_image_dict[given_class]
        train_dataset = TrainDatasetrsk(train_image_dict, opt)
        val_dataset = BaseTripletDataset(test_image_dict, opt, is_validation=True)
        eval_dataset = BaseTripletDataset(train_image_dict, opt, is_validation=True)
        return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}

def give_sop_datasets(opt):
	train_image_dict, test_image_dict = {}, {}
	train_data = open(os.path.join(opt.source_path, 'Ebay_train.txt'), 'r').read().splitlines()[1:]
	test_data = open(os.path.join(opt.source_path, 'Ebay_test.txt'), 'r').read().splitlines()[1:]
	for entry in train_data:
		info = entry.split(' ')
		class_id = info[1]
		im_path = os.path.join(opt.source_path, info[3])
		if class_id not in train_image_dict.keys():
			train_image_dict[class_id] = []
		train_image_dict[class_id].append(im_path)
	for entry in test_data:
		info = entry.split(' ')
		class_id = info[1]
		im_path = os.path.join(opt.source_path, info[3])
		if class_id not in test_image_dict.keys():
			test_image_dict[class_id] = []
		test_image_dict[class_id].append(im_path)

	new_train_dict = {}
	class_ind_ind = 0
	for cate in train_image_dict:
		new_train_dict[class_ind_ind] = train_image_dict[cate]
		class_ind_ind += 1
	train_image_dict = new_train_dict
	new_test_dict = {}
	class_ind_ind = 0
	for cate in test_image_dict:
		new_test_dict[class_ind_ind] = test_image_dict[cate]
		class_ind_ind += 1
	test_image_dict = new_test_dict
	train_dataset = TrainDatasetrsk(train_image_dict, opt)
	val_dataset = BaseTripletDataset(test_image_dict,   opt, is_validation=True)
	eval_dataset = BaseTripletDataset(train_image_dict,   opt, is_validation=True)
	return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}

def give_cub_datasets(opt):
        train_image_dict, test_image_dict = {}, {}
        all_image_dict = {}
        images_data = open(os.path.join(opt.source_path, 'images.txt'), 'r').read().splitlines()
        split_data = open(os.path.join(opt.source_path, 'train_test_split.txt'), 'r').read().splitlines()
        data_dict = {}
        for given_sample in images_data:
                key = given_sample.split(' ')[0]
                path = given_sample.split(' ')[1]
                data_dict[key] = [path]
        for given_sample in split_data:
                key = given_sample.split(' ')[0]
                split = int(given_sample.split(' ')[1])
                data_dict[key].append(split)
        for entry in data_dict.keys():
                given_sample = data_dict[entry]
                class_id = int(given_sample[0].split('.')[0])
                im_path = os.path.join(opt.source_path, 'images', given_sample[0])
                split = given_sample[1]
                if class_id not in all_image_dict.keys():
                        all_image_dict[class_id] = []
                all_image_dict[class_id].append(im_path)
        train_classes = list(all_image_dict.keys())[:100]
        val_classes = list(all_image_dict.keys())[100:]
        for given_class in train_classes:
                train_image_dict[given_class] = all_image_dict[given_class]
        for given_class in val_classes:
                test_image_dict[given_class] = all_image_dict[given_class]
        train_dataset = TrainDatasetsmoothap(train_image_dict, opt)
        val_dataset = BaseTripletDataset(test_image_dict, opt, is_validation=True)
        eval_dataset = BaseTripletDataset(train_image_dict, opt, is_validation=True)
        return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}

def give_inaturalist_datasets(opt,config):
    train_image_dict= {}

    dataset = ImageFolder(root=opt.source_path)
    
    for path, label in dataset.samples:
        if label not in train_image_dict:
            train_image_dict[label] = []
        train_image_dict[label].append(path)
    train_dataset = TrainDatasetrsk(train_image_dict, opt,config)
    return {'training':train_dataset}

def give_VehicleID_datasets(opt):
    train       = np.array(pd.read_table(opt.source_path+'/train_test_split/train_list.txt', header=None, delim_whitespace=True))
    small_test  = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_800.txt', header=None, delim_whitespace=True))
    medium_test = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_1600.txt', header=None, delim_whitespace=True))
    big_test    = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_2400.txt', header=None, delim_whitespace=True))
    lab_conv_train = {x:i for i,x in enumerate(np.unique(train[:,1]))}
    train[:,1] = np.array([lab_conv_train[x] for x in train[:,1]])
    lab_conv = {x:i for i,x in enumerate(np.unique(np.concatenate([small_test[:,1], medium_test[:,1], big_test[:,1]])))}
    small_test[:,1]  = np.array([lab_conv[x] for x in small_test[:,1]])
    medium_test[:,1] = np.array([lab_conv[x] for x in medium_test[:,1]])
    big_test[:,1]    = np.array([lab_conv[x] for x in big_test[:,1]])
    train_image_dict    = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))
    small_test_dict = {}
    for img_path, key in small_test:
        if not key in small_test_dict.keys():
            small_test_dict[key] = []
        small_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))
    medium_test_dict    = {}
    for img_path, key in medium_test:
        if not key in medium_test_dict.keys():
            medium_test_dict[key] = []
        medium_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))
    big_test_dict    = {}
    for img_path, key in big_test:
        if not key in big_test_dict.keys():
            big_test_dict[key] = []
        big_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))
    attribute = np.array(pd.read_table(opt.source_path+'/attribute/model_attr.txt', header=None, delim_whitespace=True))
    new_dict = {}
    not_found = 0
    for thing in attribute:
        if lab_conv_train[thing[0]] not in train_image_dict:
            not_found +=1
        else:
            if thing[1] not in new_dict:
                new_dict[thing[1]] = []
            new_dict[thing[1]].append(lab_conv_train[thing[0]])
    train_dataset = TrainDatasetrsk(train_image_dict, opt)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt,    is_validation=True)
    val_small_dataset     = BaseTripletDataset(small_test_dict, opt,  is_validation=True)
    val_medium_dataset    = BaseTripletDataset(medium_test_dict, opt, is_validation=True)
    val_big_dataset       = BaseTripletDataset(big_test_dict, opt,    is_validation=True)
    return {'training':train_dataset, 'testing_set1':val_small_dataset, 'testing_set2':val_medium_dataset, \
            'testing_set3':val_big_dataset, 'evaluation':eval_dataset}

class BaseTripletDataset(Dataset):
    def __init__(self, image_dict, opt, samples_per_class=8, is_validation=False):
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])
        self.is_validation = is_validation
        self.pars        = opt
        self.image_dict  = image_dict
        self.avail_classes    = sorted(list(self.image_dict.keys()))
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))
        if not self.is_validation:
            self.samples_per_class = samples_per_class
            self.current_class   = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf_list = []
        if not self.is_validation:
            transf_list.extend([transforms.RandomResizedCrop(size=224) if opt.arch=='resnet50' or opt.arch=='ViTB16' or opt.arch=='ViTB32' or opt.arch=='DeiTB' else transforms.RandomResizedCrop(size=227),
                                transforms.RandomHorizontalFlip(0.5)])
        else:
            transf_list.extend([transforms.Resize(256),
                                transforms.CenterCrop(224) if opt.arch=='resnet50' or opt.arch=='ViTB16' or opt.arch=='ViTB32' or opt.arch=='DeiTB' else transforms.CenterCrop(227)])
        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)
        self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]
        self.is_init = True

    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.pars.loss == 'recallatk':
            if self.is_init:
                self.is_init = False
            if not self.is_validation:
                if self.samples_per_class==1:
                    return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))
                if self.n_samples_drawn==self.samples_per_class:
                    counter = copy.deepcopy(self.avail_classes)
                    for prev_class in self.classes_visited:
                        if prev_class in counter: counter.remove(prev_class)
                    self.current_class   = counter[idx%len(counter)]
                    self.classes_visited = self.classes_visited+[self.current_class]
                    self.n_samples_drawn = 0
                class_sample_idx = idx%len(self.image_dict[self.current_class])
                self.n_samples_drawn += 1
                out_img = self.transform(self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
                return self.current_class,out_img
            else:
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))
        else:
            if self.is_init:
                self.current_class = self.avail_classes[idx%len(self.avail_classes)]
                self.is_init = False
            if not self.is_validation:
                if self.samples_per_class==1:
                    return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))
                if self.n_samples_drawn==self.samples_per_class:
                    counter = copy.deepcopy(self.avail_classes)
                    for prev_class in self.classes_visited:
                        if prev_class in counter: counter.remove(prev_class)
                    self.current_class   = counter[idx%len(counter)]
                    self.classes_visited = self.classes_visited[1:]+[self.current_class]
                    self.n_samples_drawn = 0
                class_sample_idx = idx%len(self.image_dict[self.current_class])
                self.n_samples_drawn += 1
                out_img = self.transform(self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
                return self.current_class,out_img
            else:
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

    def __len__(self):
        return self.n_files

flatten = lambda l: [item for sublist in l for item in sublist]

class TrainDatasetrsk(Dataset):
    def __init__(self, image_dict, opt, config):
        self.image_dict = image_dict
        self.dataset_name = opt.dataset
        self.batch_size = opt.bs
        self.samples_per_class = opt.samples_per_class
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((sub, instance))
            self.image_dict[sub] = newsub
        self.avail_classes = [*self.image_dict]
        mean = [0.4702532,0.48587758,0.38928695]
        std = [0.19859357,0.19675725,0.19619795]

        self.transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            scale=(0.2, 1.0),
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=mean,
            std=std,
        )
        self.reshuffle()
        self.count = 0

    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img

    def reshuffle(self):
        image_dict = copy.deepcopy(self.image_dict)
        print('shuffling data')
        for sub in image_dict:
            random.shuffle(image_dict[sub])
        classes = [*image_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >=self.samples_per_class) and (len(batch) < self.batch_size/self.samples_per_class):
                    batch.append(image_dict[sub_class][:self.samples_per_class])
                    image_dict[sub_class] = image_dict[sub_class][self.samples_per_class:] 
            if len(batch) == self.batch_size/self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                finished = 1
        
        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))
        

    def __getitem__(self, idx):
        batch_item = self.dataset[idx]
        if self.dataset_name in ['Inaturalist']:
            cls = int(batch_item[0])
        else:
            cls = batch_item[0]
        img = Image.open(batch_item[1])
        # self.count+=1
        # print(self.count)
        return cls, self.transform(self.ensure_3dim(img))

    def __len__(self):
        return len(self.dataset)
