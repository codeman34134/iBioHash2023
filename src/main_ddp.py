import warnings

from config import get_config
from util.misc import NativeScalerWithGradNormCount

warnings.filterwarnings("ignore")
import os, numpy as np, argparse, random, matplotlib, datetime
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from pathlib import Path
matplotlib.use('agg')
from tqdm import tqdm
import auxiliaries as aux
import datasets as data
import netlib as netlib
import losses as losses
import evaluate as eval
import time
import copy
from tensorboardX import SummaryWriter
import torch.multiprocessing
import torch.distributed as dist

from mixup import *
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      default='Inaturalist',   type=str, help='Dataset to use.', choices=['Inaturalist','vehicle_id', 'sop', 'cars196', 'cub'])
parser.add_argument('--lr',                default=0.00001,  type=float, help='Learning Rate for network parameters.')
parser.add_argument('--fc_lr_mul',         default=0,        type=float, help='OPTIONAL: Multiply the embedding layer learning rate by this value. If set to 0, the embedding layer shares the same learning rate.')
parser.add_argument('--n_epochs',          default=400,       type=int,   help='Number of training epochs.')
parser.add_argument('--kernels',           default=2,        type=int,   help='Number of workers for pytorch dataloader.')
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
parser.add_argument('--mixup', default=1, type=int, help='Gompertzap: use mixup')
parser.add_argument('--sigmoid_temperature', default=0.08, type=float, help='RS@k: the temperature of the sigmoid used to estimate ranks')
parser.add_argument('--k_vals',       nargs='+', default=[1,2,4,8], type=int, help='Recall @ Values.')
parser.add_argument('--k_vals_train',       nargs='+', default=[1,2,4,8,12,16,20,24,28,32], type=int, help='Training recall@k vals.')
parser.add_argument('--k_temperatures',       nargs='+', default=[1,2,4,8,12,16,20,24,28,32], type=int, help='Temperature for training recall@k vals.')
parser.add_argument('--resume', default='RecallatK_surrogate/src/Training_Results/Inaturalist/_22/checkpoint_0.pth.tar', type=str, help='path to checkpoint to load weights from (if empty then ImageNet pre-trained weights are loaded')
parser.add_argument('--embed_dim',    default=512,         type=int,   help='Embedding dimensionality of the network')
parser.add_argument('--arch',         default='SwinL',  type=str,   help='Network backend choice: resnet50, googlenet, BNinception')
parser.add_argument('--grad_measure',                      action='store_true', help='If added, gradients passed from embedding layer to the last conv-layer are stored in each iteration.')
parser.add_argument('--dist_measure',                      action='store_true', help='If added, the ratio between intra- and interclass distances is stored after each epoch.')
parser.add_argument('--not_pretrained',                    action='store_true', help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')
parser.add_argument('--gpu',          default=6,           type=int,   help='GPU-id for GPU to use.')
parser.add_argument('--savename',     default='',          type=str,   help='Save folder name if any special information is to be included.')
parser.add_argument('--source_path',  default='../../iBioHash_Train',         type=str, help='Path to data')
parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save the checkpoints')

parser.add_argument('--cfg', default='configs/swinv2/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.yaml',type=str,  help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)

# easy config modification
parser.add_argument('--batch-size',default=2, type=int, help="batch size for single GPU")
parser.add_argument('--data-path',default='../../iBioHash_Train', type=str, help='path to dataset')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--pretrained',type=str, default='../../Swin-Transformer/checkpoint/swinv2_large_patch4_window12_192_22k.pth',
                    help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
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
args, unparsed = parser.parse_known_args()
config = get_config(args)
opt = parser.parse_args()
# opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset

if config.AMP_OPT_LEVEL:
    print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    opt.world_size = world_size
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
else:
    rank = -1
    world_size = -1
torch.cuda.set_device(config.LOCAL_RANK)
torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.distributed.barrier()

if opt.dataset== 'Inaturalist':
    opt.k_vals = [1,4,16,32]
    opt.bs = 4000
    opt.n_epochs = 90
    if opt.arch == 'resnet50':
        opt.tau = [40,70]
        opt.bs_base = 20
        opt.lr = 0.0001
        opt.opt = 'adam'
    if opt.arch == 'ViTB32':
        opt.tau = [10,40,70]
        opt.bs_base = 200
        opt.lr = 0.00005
        opt.opt = 'adamW'
    if opt.arch == 'ViTB16':
        opt.tau = [10,40,70]
        opt.bs_base = 2
        opt.lr = 0.00005
        opt.opt = 'adamW'
    if opt.arch == 'SwinL':
        opt.tau = [10,40,70]
        opt.bs_base = 5
        opt.lr = 0.00002
        opt.opt = 'adamW'

if opt.dataset=='sop':
    opt.tau = [25,50]
    opt.k_vals = [1,10,100,1000]
    opt.bs = 4000
    opt.n_epochs = 55
    if opt.arch == 'resnet50':
        opt.bs_base = 200
        opt.lr = 0.0002
        opt.opt = 'adam'
    if opt.arch == 'ViTB32':
        opt.bs_base = 200
        opt.lr = 0.00005
        opt.opt = 'adamW'
    if opt.arch == 'ViTB16':
        opt.bs_base = 100
        opt.lr = 0.00005
        opt.opt = 'adamW'

if opt.dataset=='vehicle_id':
    opt.tau = [40,70]
    opt.k_vals = [1,5]
    opt.bs = 4000
    opt.n_epochs = 90
    if opt.arch == 'resnet50':
        opt.bs_base = 200
        opt.lr = 0.0001
        opt.opt = 'adam'
    if opt.arch == 'ViTB32':
        opt.bs_base = 200
        opt.lr = 0.0001
        opt.opt = 'adamW'
    if opt.arch == 'ViTB16':
        opt.bs_base = 100
        opt.lr = 0.00005
        opt.opt = 'adamW'

if opt.dataset == 'cars196':
    opt.k_vals = [1,2,4,8,16]
    opt.bs = 392
    opt.bs_base = 98
    if opt.arch == 'resnet50':
        opt.n_epochs = 170
        opt.tau = [80, 140]
        opt.lr = 0.0001
        opt.opt = 'adam'
    if opt.arch == 'ViTB32':
        opt.n_epochs = 50
        opt.tau = [20,30,40]
        opt.lr = 0.00003
        opt.opt = 'adamW'
    if opt.arch == 'ViTB16':
        opt.n_epochs = 50
        opt.tau = [20,30,40]
        opt.lr = 0.00001
        opt.opt = 'adamW'

if opt.dataset == 'cub':
    opt.k_vals = [1,2,4,8,16]
    opt.bs = 400
    opt.bs_base = 100
    if opt.arch == 'resnet50':
        opt.n_epochs = 40
        opt.tau = [10,20,30]
        opt.lr = 0.0001
        opt.opt = 'adam'
    if opt.arch == 'ViTB32':
        opt.n_epochs = 40
        opt.tau = [10,20,30]
        opt.lr = 0.00003
        opt.opt = 'adamW'
    if opt.arch == 'ViTB16':
        opt.n_epochs = 40
        opt.tau = [10,20,30]
        opt.lr = 0.00001
        opt.opt = 'adamW'

timestamp = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
exp_name = aux.args2exp_name(opt)
opt.save_name = f"weights_{exp_name}" +'/'+ timestamp
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)


tensorboard_path = Path(f"./logs/logs_{exp_name}") / f"{timestamp}_rank_{args.local_rank}"
tensorboard_path.parent.mkdir(exist_ok=True, parents=True)
global writer;
writer = SummaryWriter(tensorboard_path)
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)

opt.device = torch.device('cuda')
model      = netlib.networkselect(opt,config)
checkpoint = torch.load(opt.resume,map_location='cpu')

# 继续训练
state_dict = checkpoint['state_dict']
from collections import OrderedDict
new_dictionary = OrderedDict()
for key, value in state_dict.items():
    new_key = key.replace('module.', '')
    new_dictionary[new_key] = value
msg = model.load_state_dict(new_dictionary, strict=False)
print(msg)

_          = model.to(opt.device)
model_without_ddp = model
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)

if 'fc_lr_mul' in vars(opt).keys() and opt.fc_lr_mul!=0:
    all_but_fc_params = list(filter(lambda x: 'last_linear' not in x[0],model.named_parameters()))
    for ind, param in enumerate(all_but_fc_params):
        all_but_fc_params[ind] = param[1]
    fc_params         = model.model.last_linear.parameters()
    to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
                         {'params':fc_params,'lr':opt.lr*opt.fc_lr_mul,'weight_decay':opt.decay}]
else:
    to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
dataloaders      = data.give_dataloaders(opt.dataset, opt, config)
print("debug: data loaders created", len(dataloaders['training']))
opt.num_classes  = len(dataloaders['training'].dataset.avail_classes)
metrics_to_log = aux.metrics_to_examine(opt.dataset, opt.k_vals)
LOG = aux.LOGGER(opt, metrics_to_log, name='Base', start_new=True)

criterion, to_optim = losses.loss_select(opt.loss, opt, to_optim)
_ = criterion.to(opt.device)

if opt.grad_measure:
    grad_measure = eval.GradientMeasure(opt, name='baseline')
if opt.dist_measure:
    distance_measure = eval.DistanceMeasure(dataloaders['evaluation'], opt, name='Train', update_epochs=1)

if opt.opt == 'adam':
    optimizer    = torch.optim.Adam(to_optim)
elif opt.opt == 'adamW':
    optimizer    = torch.optim.AdamW(to_optim)
elif opt.opt == 'sgd':
    optimizer    = torch.optim.SGD(to_optim)
elif opt.opt == 'rmsprop':
    optimizer = torch.optim.RMSprop(to_optim)
else:
    raise Exception('unknown optimiser')
if opt.scheduler=='exp':
    scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
elif opt.scheduler=='step':
    scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
elif opt.scheduler=='none':
    print('No scheduling used!')
else:
    raise Exception('No scheduling option for input: {}'.format(opt.scheduler))
loss_scaler = torch.cuda.amp.GradScaler()

start_epoch = 0
if 'optimizer' in checkpoint and 'scheduler' in checkpoint and 'loss_scaler' in checkpoint:
    start_epoch = checkpoint['epoch']+1
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    loss_scaler.load_state_dict(checkpoint['loss_scaler'])


def same_model(model1,model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def train_one_epoch(train_dataloader, model, optimizer, criterion, opt, epoch, loss_scaler):
    loss_collect = []
    start = time.time()
    # train_dataloader.sampler.set_epoch(epoch) # no neee for this because we complete shuffle at dataset
    data_iterator = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))
    optimizer.zero_grad()

    for i,(class_labels, input) in enumerate(data_iterator):

        output = torch.zeros((len(input), opt.embed_dim)).to(opt.device)

        for j in range(0, len(input), opt.bs_base):
            input_x = input[j:j+opt.bs_base,:].to(opt.device)
            with torch.cuda.amp.autocast():
                x = model(input_x)
            output[j:j+opt.bs_base,:] = copy.copy(x.float())
            del x
            torch.cuda.empty_cache()

        # 收集所有 GPU 的 output
        output_no_grad = output.detach()
        gathered_outputs = [torch.zeros_like(output_no_grad, requires_grad=False) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_outputs, output_no_grad)
        
        # 拼接多卡的输出，并为其创建一个新的张量，设置 requires_grad=True
        gathered_out = torch.cat(gathered_outputs, dim=0).clone().detach().requires_grad_(True)

        if criterion.mixup:
            output_mixup = pos_mixup(gathered_out, criterion.num_id)
            num_samples = output_mixup.shape[0]
        else:
            num_samples = gathered_out.shape[0]

        gathered_out.retain_grad()
        loss = 0.

        for q in range(0, num_samples):
            if criterion.mixup: loss += criterion(output_mixup, q)
            else: loss += criterion(gathered_out, q)
        loss_collect.append(loss.item())
        loss.backward()
        print(loss.item())

        # 获取当前 GPU 的索引
        current_gpu_rank = dist.get_rank()

        # 提取与当前 GPU 相关的梯度片段
        start_idx = current_gpu_rank * len(input)
        end_idx = (current_gpu_rank + 1) * len(input)
        output_grad = copy.copy(gathered_out.grad[start_idx:end_idx,:])
        del loss
        del output
        del output_no_grad
        del gathered_outputs
        if criterion.mixup: del output_mixup
        torch.cuda.empty_cache()


        for j in range(0, len(input), opt.bs_base):
            input_x = input[j:j+opt.bs_base,:].to(opt.device)
            with torch.cuda.amp.autocast():
                x = model(input_x)
            loss_scaler.scale(x).backward(output_grad[j:j+opt.bs_base,:])

        loss_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        loss_scaler.step(optimizer)
        loss_scaler.update()

        optimizer.zero_grad()
        if opt.grad_measure:
            grad_measure.include(model.model.last_linear)
        if i==len(train_dataloader)-1:
            data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))
        torch.cuda.synchronize()
    LOG.log('train', LOG.metrics_to_log['train'], [epoch, np.round(time.time()-start,4), np.mean(loss_collect)])
    writer.add_scalar('global/training_loss',np.mean(loss_collect),epoch)
    if opt.grad_measure:
        grad_measure.dump(epoch)

print('\n-----\n')
# if opt.dataset in ['Inaturalist', 'sop', 'cars196', 'cub']:
#     eval_params = {'dataloader': dataloaders['testing'], 'model': model, 'opt': opt, 'epoch': 0}
# elif opt.dataset == 'vehicle_id':
#     eval_params = {
#         'dataloaders': [dataloaders['testing_set1'], dataloaders['testing_set2'], dataloaders['testing_set3']],
#         'model': model, 'opt': opt, 'epoch': 0}
print('epochs -> '+str(opt.n_epochs))

for epoch in range(start_epoch,opt.n_epochs):
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))
    _ = model.train()
    
    train_one_epoch(dataloaders['training'], model, optimizer, criterion, opt, epoch, loss_scaler)
    dataloaders['training'].dataset.reshuffle()
    if torch.distributed.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        aux.set_checkpoint(model, optimizer, LOG.progress_saver,loss_scaler,epoch,scheduler, LOG.prop.save_path + '/checkpoint_'+str(epoch)+'.pth.tar')
    #_ = model.eval()
#     # if opt.dataset in ['Inaturalist', 'sop', 'cars196', 'cub']:
#     #     eval_params = {'dataloader':dataloaders['testing'], 'model':model, 'opt':opt, 'epoch':epoch}
#     # elif opt.dataset=='vehicle_id':
#     #     eval_params = {'dataloaders':[dataloaders['testing_set1'], dataloaders['testing_set2'], dataloaders['testing_set3']], 'model':model, 'opt':opt, 'epoch':epoch}
#     # if opt.infrequent_eval == 1:
#     #     epoch_freq = 5
#     # else:
#     #     epoch_freq = 1
#     # if not opt.dataset == 'vehicle_id':
#     #     if epoch%epoch_freq == 0 or epoch == opt.n_epochs - 1:
#     #         results = eval.evaluate(opt.dataset, LOG, save=True, **eval_params)
#     #         writer.add_scalar('global/recall1',results[0][0],epoch+1)
#     #         writer.add_scalar('global/recall2',results[0][1],epoch+1)
#     #         writer.add_scalar('global/recall3',results[0][2],epoch+1)
#     #         writer.add_scalar('global/recall4',results[0][3],epoch+1)
#     #         writer.add_scalar('global/NMI',results[1],epoch+1)
#     #         writer.add_scalar('global/F1',results[2],epoch+1)
#     #
#     # else:
#     #     if epoch%epoch_freq == 0 or epoch == opt.n_epochs - 1:
#     #         results = eval.evaluate(opt.dataset, LOG, save=True, **eval_params)
#     #         writer.add_scalar('global/recall1',results[2],epoch+1)
#     #         writer.add_scalar('global/recall2',results[3],epoch+1)#writer.add_scalar('global/recall3',results[0][2],0)
#     #         writer.add_scalar('global/recall3',results[6],epoch+1)
#     #         writer.add_scalar('global/recall4',results[7],epoch+1)
#     #         writer.add_scalar('global/recall5',results[10],epoch+1)
#     #         writer.add_scalar('global/recall6',results[11],epoch+1)
#     # if opt.dist_measure:
#     #     distance_measure.measure(model, epoch)
#     # if opt.scheduler != 'none':
#     #     scheduler.step()
    print('\n-----\n')
