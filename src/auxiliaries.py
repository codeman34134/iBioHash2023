import warnings
warnings.filterwarnings("ignore")
import numpy as np, os, csv, datetime, torch, faiss
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
from sklearn import metrics

def args2exp_name(args):
        exp_name = f"{args.dataset}_{args.loss}_{args.lr}_bs{args.bs}_spc{args.samples_per_class}_embed{args.embed_dim}_arch{args.arch}_decay{args.decay}_fclr{args.fc_lr_mul}_anneal{args.sigmoid_temperature}"
        return exp_name

def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str



def f1_score(model_generated_cluster_labels, target_labels, feature_coll, computed_centroids):
    from scipy.special import comb
    d = np.zeros(len(feature_coll))
    for i in range(len(feature_coll)):
        d[i] = np.linalg.norm(feature_coll[i,:] - computed_centroids[model_generated_cluster_labels[i],:])
    labels_pred = np.zeros(len(feature_coll))
    for i in np.unique(model_generated_cluster_labels):
        index = np.where(model_generated_cluster_labels == i)[0]
        ind = np.argmin(d[index])
        cid = index[ind]
        labels_pred[index] = cid
    N = len(target_labels)
    avail_labels = np.unique(target_labels)
    n_labels     = len(avail_labels)
    count_cluster = np.zeros(n_labels)
    for i in range(n_labels):
        count_cluster[i] = len(np.where(target_labels == avail_labels[i])[0])
    keys     = np.unique(labels_pred)
    num_item = len(keys)
    values   = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])
    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[labels_pred[i]]
        count_item[index] = count_item[index] + 1
    tp_fp = 0
    for k in range(n_labels):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)
    tp = 0
    for k in range(n_labels):
        member = np.where(target_labels == avail_labels[k])[0]
        member_ids = labels_pred[member]
        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1
        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)
    fp = tp_fp - tp
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)
    fn = count - tp
    beta = 1
    P  = tp / (tp + fp)
    R  = tp / (tp + fn)
    F1 = (beta*beta + 1) * P * R / (beta*beta * P + R)
    return F1

def eval_metrics_one_dataset(model, test_dataloader, device, k_vals, opt):
    torch.cuda.empty_cache()
    _ = model.eval()
    n_classes = len(test_dataloader.dataset.avail_classes)
    with torch.no_grad():
        target_labels, feature_coll = [],[]
        final_iter = tqdm(test_dataloader, desc='Computing Evaluation Metrics...')
        image_paths= [x[0] for x in test_dataloader.dataset.image_list]
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            feature_coll.extend(out.cpu().detach().numpy().tolist())
        target_labels = np.hstack(target_labels).reshape(-1,1)
        feature_coll  = np.vstack(feature_coll).astype('float32')
        torch.cuda.empty_cache()
        cpu_cluster_index = faiss.IndexFlatL2(feature_coll.shape[-1])
        kmeans            = faiss.Clustering(feature_coll.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000
        kmeans.train(feature_coll, cpu_cluster_index)
        computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, feature_coll.shape[-1])
        faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        faiss_search_index.add(computed_centroids)
        _, model_generated_cluster_labels = faiss_search_index.search(feature_coll, 1)
        NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))
        faiss_search_index  = faiss.IndexFlatL2(feature_coll.shape[-1])
        faiss_search_index.add(feature_coll)
        _, k_closest_points = faiss_search_index.search(feature_coll, int(np.max(k_vals)+1))
        k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]
        print('computing recalls')
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(target_labels)
            recall_all_k.append(recall_at_k)
        print('finished recalls')
        print('computing F1')
        F1 = 0
        print('finished computing f1')
    return F1, NMI, recall_all_k, feature_coll

def eval_metrics_query_and_gallery_dataset(model, query_dataloader, gallery_dataloader, device, k_vals, opt):
    torch.cuda.empty_cache()
    _ = model.eval()
    n_classes = len(query_dataloader.dataset.avail_classes)
    with torch.no_grad():
        query_target_labels, query_feature_coll     = [],[]
        query_image_paths   = [x[0] for x in query_dataloader.dataset.image_list]
        query_iter = tqdm(query_dataloader, desc='Extraction Query Features')
        for idx,inp in enumerate(query_iter):
            input_img,target = inp[-1], inp[0]
            query_target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            query_feature_coll.extend(out.cpu().detach().numpy().tolist())
        gallery_target_labels, gallery_feature_coll = [],[]
        gallery_image_paths = [x[0] for x in gallery_dataloader.dataset.image_list]
        gallery_iter = tqdm(gallery_dataloader, desc='Extraction Gallery Features')
        for idx,inp in enumerate(gallery_iter):
            input_img,target = inp[-1], inp[0]
            gallery_target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            gallery_feature_coll.extend(out.cpu().detach().numpy().tolist())
        query_target_labels, query_feature_coll     = np.hstack(query_target_labels).reshape(-1,1), np.vstack(query_feature_coll).astype('float32')
        gallery_target_labels, gallery_feature_coll = np.hstack(gallery_target_labels).reshape(-1,1), np.vstack(gallery_feature_coll).astype('float32')
        torch.cuda.empty_cache()
        stackset    = np.concatenate([query_feature_coll, gallery_feature_coll],axis=0)
        stacklabels = np.concatenate([query_target_labels, gallery_target_labels],axis=0)
        cpu_cluster_index = faiss.IndexFlatL2(stackset.shape[-1])
        kmeans            = faiss.Clustering(stackset.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000
        kmeans.train(stackset, cpu_cluster_index)
        computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, stackset.shape[-1])
        faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        faiss_search_index.add(computed_centroids)
        _, model_generated_cluster_labels = faiss_search_index.search(stackset, 1)
        NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), stacklabels.reshape(-1))
        faiss_search_index  = faiss.IndexFlatL2(gallery_feature_coll.shape[-1])
        faiss_search_index.add(gallery_feature_coll)
        _, k_closest_points = faiss_search_index.search(query_feature_coll, int(np.max(k_vals)))
        k_closest_classes   = gallery_target_labels.reshape(-1)[k_closest_points]
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(query_target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(query_target_labels)
            recall_all_k.append(recall_at_k)
        recall_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(k_vals, recall_all_k))
        F1 = f1_score(model_generated_cluster_labels, stacklabels, stackset, computed_centroids)
    return F1, NMI, recall_all_k, query_feature_coll, gallery_feature_coll

def recover_closest_one_dataset(feature_matrix_all, image_paths, save_path, n_image_samples=10, n_closest=3):
    image_paths = np.array([x[0] for x in image_paths])
    sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)), n_image_samples)
    faiss_search_index = faiss.IndexFlatL2(feature_matrix_all.shape[-1])
    faiss_search_index.add(feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(feature_matrix_all, n_closest+1)
    sample_paths = image_paths[closest_feature_idxs][sample_idxs]
    f,axes = plt.subplots(n_image_samples, n_closest+1)
    for i,(ax,plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i%(n_closest+1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()

def set_checkpoint(model, optimizer, progress_saver, loss_scaler, epoch,scheduler, savepath):
    torch.save({'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(),
                'progress':progress_saver,'loss_scaler':loss_scaler.state_dict(),'epoch':epoch,'scheduler':scheduler.state_dict()}, savepath)

class CSV_Writer():
    def __init__(self, save_path, columns):
        self.save_path = save_path
        self.columns   = columns

        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(self.columns)

    def log(self, inputs):
        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(inputs)

class InfoPlotter():
    def __init__(self, save_path, title='Training Log', figsize=(20,15)):
        self.save_path = save_path
        self.title     = title
        self.figsize   = figsize
        self.v_colors    = ['r','g','b','y','m','k','c']
        self.t_colors    = ['k','b','r','g']

    def make_plot(self, t_epochs, v_epochs, t_metrics, v_metrics, t_labels, v_labels, appendix=None):
        plt.style.use('ggplot')
        f,axes = plt.subplots(1,2)
        for i in range(len(t_metrics)):
            axes[0].plot(t_epochs, t_metrics[i], '-{}'.format(self.t_colors[i]), linewidth=1, label=t_labels[i])
        axes[0].set_title('Training Performance', fontsize=19)
        axes[0].legend(fontsize=16)
        axes[0].tick_params(axis='both', which='major', labelsize=16)
        axes[0].tick_params(axis='both', which='minor', labelsize=16)
        for i in range(len(v_metrics)):
            axes[1].plot(v_epochs, v_metrics[i], '-{}'.format(self.v_colors[i]), linewidth=1, label=v_labels[i])
        axes[1].set_title(self.title, fontsize=19)
        axes[1].legend(fontsize=16)
        axes[1].tick_params(axis='both', which='major', labelsize=16)
        axes[1].tick_params(axis='both', which='minor', labelsize=16)
        f.set_size_inches(2*self.figsize[0], self.figsize[1])
        savepath = self.save_path
        f.savefig(self.save_path, bbox_inches='tight')
        plt.close()

def set_logging(opt):
    checkfolder = opt.save_path+'/'+opt.savename
    if opt.savename == '':
        date = datetime.datetime.now()
        time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)
        checkfolder = opt.save_path+'/{}_{}_'.format(opt.dataset.upper(), opt.arch.upper())+time_string
    counter     = 1
    while os.path.exists(checkfolder):
        checkfolder = opt.save_path+'/'+opt.savename+'_'+str(counter)
        counter += 1
    os.makedirs(checkfolder)
    opt.save_path = checkfolder
    with open(opt.save_path+'/Parameter_Info.txt','w') as f:
        f.write(gimme_save_string(opt))
    pkl.dump(opt,open(opt.save_path+"/hypa.pkl","wb"))

class LOGGER():
    def __init__(self, opt, metrics_to_log, name='Basic', start_new=True):
        self.prop           = opt
        self.metrics_to_log = metrics_to_log
        if start_new: set_logging(opt)
        if self.prop.dataset != 'vehicle_id':
            self.info_plot = InfoPlotter(opt.save_path+'/InfoPlot_{}.svg'.format(name))
        else:
            self.info_plot = {'Set {}'.format(i): InfoPlotter(opt.save_path+'/InfoPlot_{}_Set{}.svg'.format(name,i+1)) for i in range(3)}
        self.progress_saver = self.provide_progress_saver(metrics_to_log)
        self.csv_loggers= {mode:CSV_Writer(opt.save_path+'/log_'+mode+'_'+name+'.csv', lognames) for mode, lognames in metrics_to_log.items()}

    def provide_progress_saver(self, metrics_to_log):
        Progress_Saver = {key:{sub_key:[] for sub_key in metrics_to_log[key]} for key in metrics_to_log.keys()}
        return Progress_Saver

    def log(self, main_keys, metric_keys, values):
        if not isinstance(main_keys, list):   main_keys = [main_keys]
        if not isinstance(metric_keys, list): metric_keys = [metric_keys]
        if not isinstance(values, list):      values = [values]
        for main_key in main_keys:
            for value, metric_key in zip(values, metric_keys):
                self.progress_saver[main_key][metric_key].append(value)
        self.csv_loggers[main_key].log(values)

    def update_info_plot(self):
        t_epochs         = self.progress_saver['val']['Epochs']
        t_loss_list      = [self.progress_saver['train']['Train Loss']]
        t_legend_handles = ['Train Loss']
        v_epochs         = self.progress_saver['val']['Epochs']
        if self.prop.dataset != 'vehicle_id':
            title = ' | '.join(key+': {0:3.3f}'.format(np.max(item)) for key,item in self.progress_saver['val'].items() if key not in ['Time', 'Epochs'])
            self.info_plot.title = title
            v_metric_list    = [self.progress_saver['val'][key] for key in self.progress_saver['val'].keys() if key not in ['Time', 'Epochs']]
            v_legend_handles = [key for key in self.progress_saver['val'].keys() if key not in ['Time', 'Epochs']]
            self.info_plot.make_plot(t_epochs, v_epochs, t_loss_list, v_metric_list, t_legend_handles, v_legend_handles)
        else:
            for i in range(3):
                title = ' | '.join(key+': {0:3.3f}'.format(np.max(item)) for key,item in self.progress_saver['val'].items() if key not in ['Time', 'Epochs'] and 'Set {}'.format(i) in key)
                self.info_plot['Set {}'.format(i)].title = title
                v_metric_list    = [self.progress_saver['val'][key] for key in self.progress_saver['val'].keys() if key not in ['Time', 'Epochs'] and 'Set {}'.format(i) in key]
                v_legend_handles = [key for key in self.progress_saver['val'].keys() if key not in ['Time', 'Epochs'] and 'Set {}'.format(i) in key]
                self.info_plot['Set {}'.format(i)].make_plot(t_epochs, v_epochs, t_loss_list, v_metric_list, t_legend_handles, v_legend_handles, appendix='set_{}'.format(i))

def metrics_to_examine(dataset, k_vals):
    metric_dict        = {'train':['Epochs','Time','Train Loss']}
    if dataset=='vehicle_id':
        metric_dict['val'] = ['Epochs','Time']
        #Vehicle_ID uses three test sets
        for i in range(3):
            metric_dict['val'] += ['Set {} NMI'.format(i), 'Set {} F1'.format(i)]
            for k in k_vals:
                metric_dict['val'] += ['Set {} Recall @ {}'.format(i,k)]
    else:
        metric_dict['val'] = ['Epochs','Time','NMI', 'F1']
        metric_dict['val'] += ['Recall @ {}'.format(k) for k in k_vals]
    return metric_dict

def run_kmeans(features, n_cluster):
    n_samples, dim = features.shape
    kmeans = faiss.Kmeans(dim, n_cluster)
    kmeans.n_iter, kmeans.min_points_per_centroid, kmeans.max_points_per_centroid = 20,5,1000000000
    kmeans.train(features)
    _, cluster_assignments = kmeans.index.search(features,1)
    return cluster_assignments

def save_graph(opt, model):
    inp = torch.randn((1,3,224,224)).to(opt.device)
    network_output = model(inp)
    if isinstance(network_output, dict): network_output = network_output['Class']
    from graphviz import Digraph
    def make_dot(var, savename, params=None):
        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}
        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='6',
                         ranksep='0.1',
                         height='0.6',
                         width='1')
        dot  = Digraph(node_attr=node_attr, format='svg', graph_attr=dict(size="40,10", rankdir='LR', rank='same'))
        seen = set()
        def size_to_str(size):
            return '('+(', ').join(['%d' % v for v in size])+')'

        def add_nodes(var):
            replacements  = ['Backward', 'Th', 'Cudnn']
            color_assigns = {'Convolution':'orange',
                             'ConvolutionTranspose': 'lightblue',
                             'Add': 'red',
                             'Cat': 'green',
                             'Softmax': 'yellow',
                             'Sigmoid': 'yellow',
                             'Copys':   'yellow'}
            if var not in seen:
                op1 = torch.is_tensor(var)
                op2 = not torch.is_tensor(var) and str(type(var).__name__)!='AccumulateGrad'
                text = str(type(var).__name__)
                for rep in replacements:
                    text = text.replace(rep, '')
                color = color_assigns[text] if text in color_assigns.keys() else 'gray'
                if 'Pool' in text: color = 'lightblue'
                if op1 or op2:
                    if hasattr(var, 'next_functions'):
                        count = 0
                        for i, u in enumerate(var.next_functions):
                            if str(type(u[0]).__name__)=='AccumulateGrad':
                                if count==0: attr_text = '\nParameter Sizes:\n'
                                attr_text += size_to_str(u[0].variable.size())
                                count += 1
                                attr_text += ' '
                        if count>0: text += attr_text
                if op1:
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                if op2:
                    dot.node(str(id(var)), text, fillcolor=color)
                seen.add(var)
                if op1 or op2:
                    if hasattr(var, 'next_functions'):
                        for u in var.next_functions:
                            if u[0] is not None:
                                if str(type(u[0]).__name__)!='AccumulateGrad':
                                    dot.edge(str(id(u[0])), str(id(var)))
                                    add_nodes(u[0])
                    if hasattr(var, 'saved_tensors'):
                        for t in var.saved_tensors:
                            dot.edge(str(id(t)), str(id(var)))
                            add_nodes(t)
        add_nodes(var.grad_fn)
        dot.save(savename)
        return dot
    if not os.path.exists(opt.save_path):
        raise Exception('No save folder {} available!'.format(opt.save_path))
    viz_graph = make_dot(network_output, opt.save_path+"/Network_Graphs"+"/{}_network_graph".format(opt.arch))
    viz_graph.format = 'svg'
    viz_graph.render()
    torch.cuda.empty_cache()
