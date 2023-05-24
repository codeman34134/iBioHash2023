import argparse
import pickle
from collections import OrderedDict

from torchvision.datasets import VisionDataset

import torch
import os

from PIL import Image

import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from tqdm import tqdm

from config import get_config
from models.swin_transformer_v2 import SwinTransformerV2
from netlib import ViTB16
from util.datasets import build_transform

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    return filename.lower().endswith(extensions)

def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:

    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index, fname
                    instances.append(item)
    return instances

class DatasetFolder(VisionDataset):

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,fname = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, fname

    def __len__(self) -> int:
        return len(self.samples)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

def build_dataset(is_train, args, config):
    transform = build_transform(is_train, config)

    root = '{}'.format(args.data_path)
    dataset = ImageFolder(root, transform=transform)
    print(dataset)

    return dataset



parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml',type=str,  help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
# easy config modification
parser.add_argument('--batch-size',default=50, type=int, help="batch size for single GPU")
parser.add_argument('--data-path',default='/home/dataset/iBioHash', type=str, help='path to dataset')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--accumulation-steps',default=64, type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used (deprecated!)')
parser.add_argument('--output', default='output', type=str, metavar='PATH',
                    help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
# distributed training
parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
# for acceleration
parser.add_argument('--fused_window_process', action='store_true',
                    help='Fused window shift & window partition, similar for reversed part.')
parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
parser.add_argument('--optim', type=str,
                    help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')


parser.add_argument('--dataset',      default='Inaturalist',   type=str, help='Dataset to use.', choices=['Inaturalist','vehicle_id', 'sop', 'cars196', 'cub'])
parser.add_argument('--lr',                default=0.00001,  type=float, help='Learning Rate for network parameters.')
parser.add_argument('--fc_lr_mul',         default=0,        type=float, help='OPTIONAL: Multiply the embedding layer learning rate by this value. If set to 0, the embedding layer shares the same learning rate.')
parser.add_argument('--n_epochs',          default=400,       type=int,   help='Number of training epochs.')
parser.add_argument('--kernels',           default=16,        type=int,   help='Number of workers for pytorch dataloader.')
parser.add_argument('--bs',                default=112 ,     type=int,   help='Mini-Batchsize to use.')
parser.add_argument('--bs_base',                default=200 ,     type=int,   help='Mini-Batchsize to use for evaluation and for chunks in two feed-forward setup.')
parser.add_argument('--samples_per_class', default=4,        type=int,   help='Number of samples in one class drawn before choosing the next class')
parser.add_argument('--seed',              default=1,        type=int,   help='Random seed for reproducibility.')
parser.add_argument('--scheduler',         default='step',   type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
parser.add_argument('--gamma',             default=0.3,      type=float, help='Learning rate reduction after tau epochs.')
parser.add_argument('--decay',             default=0.0004,   type=float, help='Weight decay for optimizer.')
parser.add_argument('--tau',               default= [10,20,30],nargs='+',type=int,help='Stepsize(s) before reducing learning rate.')
parser.add_argument('--infrequent_eval', default=1,type=int, help='only compute evaluation metrics every 10 epochs')
parser.add_argument('--opt', default = 'adam',help='adam or sgd')
parser.add_argument('--loss',         default='recallatk', type=str)
parser.add_argument('--mixup', default=0, type=int, help='Gompertzap: use mixup')
parser.add_argument('--sigmoid_temperature', default=0.01, type=float, help='RS@k: the temperature of the sigmoid used to estimate ranks')
parser.add_argument('--k_vals',       nargs='+', default=[1,2,4,8], type=int, help='Recall @ Values.')
parser.add_argument('--k_vals_train',       nargs='+', default=[1,2,4,8,16], type=int, help='Training recall@k vals.')
parser.add_argument('--k_temperatures',       nargs='+', default=[1,2,4,8,16], type=int, help='Temperature for training recall@k vals.')
parser.add_argument('--resume', default='/home/hdd/ct_RecallatK_surrogate/src/Training_Results/Inaturalist/_25/checkpoint_3.pth.tar', type=str, help='path to checkpoint to load weights from (if empty then ImageNet pre-trained weights are loaded')
parser.add_argument('--embed_dim',    default=48,         type=int,   help='Embedding dimensionality of the network')
parser.add_argument('--arch',         default='SwinL',  type=str,   help='Network backend choice: resnet50, googlenet, BNinception')
parser.add_argument('--grad_measure',                      action='store_true', help='If added, gradients passed from embedding layer to the last conv-layer are stored in each iteration.')
parser.add_argument('--dist_measure',                      action='store_true', help='If added, the ratio between intra- and interclass distances is stored after each epoch.')
parser.add_argument('--not_pretrained',                    action='store_true', help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')
parser.add_argument('--gpu',          default=2,           type=int,   help='GPU-id for GPU to use.')
parser.add_argument('--savename',     default='',          type=str,   help='Save folder name if any special information is to be included.')
parser.add_argument('--source_path',  default='/home/hdd/iBioHash_Train',         type=str, help='Path to data')
parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save the checkpoints')

opt = parser.parse_args()
args, unparsed = parser.parse_known_args()
config = get_config(args)

dataset_train = build_dataset(is_train=False, args=args,config=config)
sampler_train = torch.utils.data.SequentialSampler(
            dataset_train
        )

data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=3,
        pin_memory=True,
        drop_last=False,
    )

model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=opt.embed_dim,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
weights = torch.load(opt.resume,map_location=torch.device('cpu'))
new_dictionary = OrderedDict()
for key, value in weights['state_dict'].items():
    new_key = key.replace('module.', '')
    new_dictionary[new_key] = value
model.load_state_dict(new_dictionary)

model.to('cuda')
model.eval()


df_gallery = {}
df_query = {}


for index,(img,fold,fname) in tqdm(enumerate(data_loader_train)):
    img = img.to('cuda')
    with torch.no_grad():
        output = model(img).cpu().detach().tolist()
    codes = output
    for i in range (args.batch_size):
        if fold[i] == 1:
            # df_query.loc[df_query['image_id'] == fname[i],['hashcode']] = '\''+ codes[i] + '\''
            df_query[fname[i]]=codes[i]
        if fold[i] == 0:
            # df_gallery.loc[df_gallery['image_id'] == fname[i], ['hashcode']] = '\'' + codes[i] + '\''
            df_gallery[fname[i]]=codes[i]


with open("df_query"+opt.resume.split("_")[-1].split(".")[0]+"_swinL_ddp_384-48.pkl", "wb") as f:
    # 将DataFrame对象写入文件中
    pickle.dump(df_query, f)

with open("df_gallery"+opt.resume.split("_")[-1].split(".")[0]+"_swinL_ddp_384-48.pkl", "wb") as f:
    # 将DataFrame对象写入文件中
    pickle.dump(df_gallery, f)