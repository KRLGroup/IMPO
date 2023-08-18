import math, random, argparse, time, uuid
import os, os.path as osp
from helper import makeDirectory, set_gpu

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_geometric.datasets import TUDataset, MoleculeNet
from torch_geometric.utils import subgraph
from syn_dataset import SynGraphDataset
from impo_model import IMem_Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from logger import Logger
import numpy as np
from sklearn.metrics import f1_score
torch.backends.cudnn.benchmark = False


class Trainer(object):

    def __init__(self, params):
        self.p = params

        # set GPU
        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.p.gpu}')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.p.use_node_attr = True
        self.p.use_edge_attr = True
        self.loadData()

        # build the model
        self.model  = None
        self.optimizer  = None

    # load data
    def loadData(self):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', self.p.dataset)
        if self.p.dataset == 'ba_2motifs':
            print(path)
            dataset = SynGraphDataset(path, 'ba_2motifs')
        elif self.p.dataset == 'MUTAG':
            import mutag_expl
            print(path)
            dataset = mutag_expl.Mutag(path)
        elif self.p.dataset == 'ESOL':
            print(path)
            dataset = MoleculeNet(path, "ESOL")
            print(len(dataset[1]))
        else:
            dataset = TUDataset(path, self.p.dataset, use_node_attr=self.p.use_node_attr, use_edge_attr=self.p.use_edge_attr)
        self.data       = dataset

    # load model
    def addModel(self):
        if self.p.model == 'IMPO':
            model = IMem_Pool(
                dataset=self.data,
                num_layers=self.p.num_layers,
                hidden=self.p.hid_dim,
                num_clusters = self.p.num_clusters,
                dropout_att = self.p.dropout_att,
                heads = self.p.heads,
                explainable = True,
                gcn_normalize = self.p.gcn_normalize,
                lambda1 = self.p.lambda1,
                lambda2 = self.p.lambda2,
                )
        
        else:                   raise NotImplementedError
        model.to(self.device).reset_parameters()
        return model

    def addOptimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        
    # train model for an epoch
    def run_epoch(self, epoch, loader, explainable=False):
        self.model.train()
        self.model.pool.k.requires_grad_ = True

        total_loss = 0
        for data in loader:
            self.optimizer.zero_grad()
            data    = data.to(self.device)
            ground_truth = data.y.clone()
            aux_loss = 0
            if explainable:
                out = self.model.forward_expl(data)
            else:
                out = self.model(data, gumbel_scale=epoch*10)
            if type(out) == tuple:
                out, aux_loss = out
            loss    = F.nll_loss(out, ground_truth.view(-1))+aux_loss
            loss.backward()
            total_loss += loss.item() * self.num_graphs(data)
            self.optimizer.step()

        self.model.project(loader)

        return total_loss / len(loader.dataset)

    # validate or test model
    def predict(self, loader, explainable=False):
        self.model.eval()
        preds = []
        trues = []
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                if explainable:
                    pred = self.model.forward_expl(data)
                else:
                    pred    = self.model(data, gumbel_scale=1e4)
                if type(pred) == tuple:
                    pred = pred[0]
                pred = pred.softmax(-1)
                pred = pred[:,1]
                preds.extend(pred.reshape(-1).detach().cpu().tolist())
                trues.extend(data.y.reshape(-1).detach().cpu().tolist())                
        return accuracy_score(trues, [int(i>=0.5) for i in preds]), roc_auc_score(trues, preds)

    # validate or test model activations
    def predict_activations(self, loader, explainable=False, boolean=False):
        self.model.eval()
        matrix = torch.zeros((self.p.num_clusters*(self.data.num_classes+1), self.data.num_classes))
        tmp = torch.zeros(self.data.num_classes)
        plot_data = []
        data_list = []
        Ss = []
        masks = []
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                preds, S, mask = self.model.forward_activations(data, boolean=boolean, gumbel_scale=1e4)#.detach().cpu()
                S = S.cpu()
                mask = mask.cpu()
                classes = torch.ones_like(mask)*data.y.reshape(-1,1).cpu()
                acts = S[mask]
                classes = classes[mask].reshape(-1,1)
                plot_data.append((acts.numpy(), classes.numpy()))
                Ss.append(S)
                masks.append(mask.numpy())
            trues = data.y.view(-1)

            for p,t in zip(preds, trues):
                tmp[t]+=p.sum()
                matrix[:, t]+=p
                
            data_list.extend(data.to_data_list())
        acts = np.vstack([x[0] for x in plot_data])
        classes = np.vstack([x[1] for x in plot_data])
        plot_data = (acts, classes)
        return matrix/tmp.reshape(1,-1), (data_list, Ss, masks), plot_data
    # save model locally
    def save_model(self, save_path):
        state = {
            'state_dict':   self.model.state_dict(),
            'optimizer' :   self.optimizer.state_dict(),
            'args'      :   vars(self.p)
        }
        torch.save(state, save_path)

    # load model from path
    def load_model(self, load_path):
        state       = torch.load(load_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

    # use 10 fold cross-validation
    def k_fold(self):
        kf = KFold(self.p.folds, shuffle=True, random_state=self.p.seed)

        test_indices, train_indices = [], []
        for _, idx in kf.split(torch.zeros(len(self.data)), self.data.data.y):
            test_indices.append(torch.from_numpy(idx))

        val_indices = [test_indices[i - 1] for i in range(self.p.folds)]

        for i in range(self.p.folds):
            train_mask = torch.ones(len(self.data), dtype=torch.uint8)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero().view(-1))

        return train_indices, test_indices, val_indices

    def train_val_test_split(self):
        indices = list(range(len(self.data)))
        train_indices, test_val_indices = train_test_split(indices, test_size=0.2, shuffle=True, random_state=self.p.seed)
        val_indices = test_val_indices[:len(test_val_indices)//2]
        test_indices = test_val_indices[len(test_val_indices)//2:]

        return [train_indices], [val_indices], [test_indices]

    def num_graphs(self, data):
        if data.batch is not None:  return data.num_graphs
        else:               return data.x.size(0)


    # main function for running the experiments
    def run(self):
        val_accs, test_accs = [], []
        val_acts, test_acts = [], []

        makeDirectory('torch_saved/')
        save_path = 'torch_saved/{}'.format(self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            print('Successfully Loaded previous model')

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        k_fold = self.k_fold
        if self.p.folds == 1:
            k_fold = self.train_val_test_split

        # iterate over folds
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold())):
            # Reinitialise model and optimizer for each fold
            self.model  = self.addModel()
            self.optimizer  = self.addOptimizer()

            train_dataset   = self.data[train_idx]
            test_dataset    = self.data[test_idx]
            val_dataset = self.data[val_idx]

            if 'adj' in train_dataset[0]:
                train_loader    = DenseLoader(train_dataset, self.p.batch_size, shuffle=True, num_workers=1)
                val_loader  = DenseLoader(val_dataset,   self.p.batch_size, shuffle=False, num_workers=1)
                test_loader = DenseLoader(test_dataset,  self.p.batch_size, shuffle=False, num_workers=1)
            else:
                train_loader    = DataLoader(train_dataset, self.p.batch_size, shuffle=True, num_workers=1)
                val_loader  = DataLoader(val_dataset, self.p.batch_size, shuffle=False, num_workers=1)
                test_loader = DataLoader(test_dataset, self.p.batch_size, shuffle=False, num_workers=1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            
            best_val_acc, best_test_acc = 0.0, 0.0
            best_val_auc = 0.0
            prova = 0.0
            best_train_acc, train_acc = 0.0, 0.0
            best_val_act, best_val_loss = 0.0, float('inf')

            max_patience = 10000 # disable early stopping
            patience = 0
            self.model.pool.use_gumbel = False
            for epoch in range(1, self.p.max_epochs + 1):   
                train_loss  = self.run_epoch(epoch, train_loader)
                val_acc, val_auc        = self.predict(val_loader)
                test_acc, test_auc      = self.predict(test_loader)
                train_acc, _        = self.predict(train_loader)
                acts, _, plot_data = self.predict_activations(val_loader, boolean=True)
                
                val_act = (acts[:-1,:].trace()/acts[:-1,:].sum()).item()

                # lr_decay
                if epoch % self.p.lr_decay_step == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.p.lr_decay_factor * param_group['lr']

                patience += 1

                # save model for best val score
                if (val_acc > best_val_acc or (val_act>best_val_act and val_acc >= best_val_acc)):
                    patience = 0
                    best_val_acc    = val_acc
                    best_train_acc  = train_acc
                    best_val_auc    = val_auc
                    best_test_acc   = test_acc
                    best_val_act   = val_act
                    prova = (3*val_acc+val_act)/4
                    self.save_model(save_path)
                    print(acts)
                    logger.set_value(f"train/model_{self.p.seed}_{fold}", save_path)




                logger.log(f"train/loss_{self.p.seed}_{fold}", train_loss)
                logger.log(f"val/acc_{self.p.seed}_{fold}", val_acc)
                logger.log(f"val/act_{self.p.seed}_{fold}", val_act)
                print('---[INFO]---{}/{:02d}/{:03d}: Loss: {:.4f}\tTrain Acc: {:.4f}\tVal Acc: {:.4f}\tTest Acc: {:.4f}\tVal Act: {:.4f}\tBest Val Acc: {:.4f}\tBest Test Acc: {:.4f}'.format(self.p.seed, fold+1, epoch, train_loss, train_acc, val_acc, test_acc, val_act, best_val_acc, best_test_acc))
                if patience >= max_patience:
                    break


            self.load_model(save_path)         
                
            if self.p.eval_expl:
                import matplotlib.pyplot as plt
                import networkx as nx

                acts, classes = plot_data
                classes = classes.reshape(-1)
                num_clusters = acts.shape[-1]
                num_classes = classes.max()+1


                from torch_geometric.utils import to_networkx
                import traceback
                _, (data, Ss, masks), plot_data = self.predict_activations(test_loader)
                
                expl_true = []
                expl_pred = []
                from torch_geometric.data import Data
                i = 0
                fidelity = []
                for S, mask in zip(Ss, masks):
                    for j in range(S.shape[0]):
                        try:
                            expl = (S[j, mask[j], :].argmax(-1) == 0).int()
                            batch = torch.zeros(data[i].x.shape[0]).to(data[i].x.device).long()
                            data_orig = Data(x=data[i].x, y=data[i].y,   edge_index=data[i].edge_index, edge_weight=None, batch=batch)
                            masked_edge_index, _ = subgraph(expl==0, data_orig.edge_index, relabel_nodes=True, num_nodes=data_orig.x.shape[0])
                            data_mask = Data(x=data[i].x[expl==0], y=data[i].y, edge_index=masked_edge_index, edge_weight=None, batch=batch[expl==0]) 
#                             fidelity.append((self.model(data_orig)[0].softmax(-1)[0,0].item()) - (self.model(data_mask)[0].softmax(-1)[0,0].item()))
                            fidelity.append(int(self.model(data_orig)[0].softmax(-1)[0,0].item()>0.5) - int(self.model(data_mask)[0].softmax(-1)[0,0].item()>0.5))
                            # print(self.model(data_orig)[0].softmax(-1)[0,0].item(),  self.model(data_mask)[0].softmax(-1)[0,0].item())
                            for a, b in zip(expl, data[i].node_label):
                                expl_true.append(b.item())
                                expl_pred.append(a.item())
                        except: traceback.print_exc()
                        i+=1
                        
                from sklearn.metrics import accuracy_score
                print('expl acc:',accuracy_score(expl_true, expl_pred))
                print('fidelity:',np.mean(fidelity))
            # load best model for testing
            print(best_test_acc)
            best_test_acc, _    = self.predict(test_loader)
            print(best_test_acc)
            self.load_model(save_path)
            best_test_acc, _    = self.predict(test_loader)
            print(best_test_acc)
            self.load_model(save_path)
            best_test_acc, _    = self.predict(test_loader)
            print(best_test_acc)
            acts, _, _ = self.predict_activations(test_loader, boolean=True)
            best_test_act = (acts[:-1,:].trace()/acts[:-1,:].sum()).item()

            
            logger.set_value(f"test/acc_{self.p.seed}_{fold}", best_test_acc)
            logger.set_value(f"test/act_{self.p.seed}_{fold}", best_test_act)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            val_acts.append(best_val_act)
            val_accs.append(best_val_acc)
            test_acts.append(best_test_act)
            test_accs.append(best_test_acc)

            val_acc_mean            = np.round(np.mean(val_accs), 4)
            test_acc_mean           = np.round(np.mean(test_accs), 4)
            val_act_mean            = np.round(np.mean(val_acts), 4)
            test_act_mean           = np.round(np.mean(test_acts), 4)
            

            logger.log(f"val/acc_mean_{self.p.seed}", val_acc_mean)
            logger.log(f"test/acc_mean_{self.p.seed}", test_acc_mean)
            logger.log(f"val/act_mean_{self.p.seed}", val_act_mean)
            logger.log(f"test/act_mean_{self.p.seed}", test_act_mean)


        print('---[INFO]---Val Acc: {:.4f}, Test Accuracy: {:.3f}, Val Act: {:.4f}, Test Act: {:.3f}'.format(val_acc_mean, test_acc_mean, val_act_mean, test_act_mean))
        

        return val_acc_mean, test_acc_mean

if __name__== '__main__':

    parser = argparse.ArgumentParser(description='Neural Network Trainer Template')

    parser.add_argument('-model',       dest='model',               default='IMPO',                    help='Model to use')
    parser.add_argument('-data',        dest='dataset',             default='PROTEINS', type=str,           help='Dataset to use')
    parser.add_argument('-epoch',       dest='max_epochs',          default=100,        type=int,           help='Max epochs')
    parser.add_argument('-l2',          dest='l2',                  default=5e-4,       type=float,         help='L2 regularization')
    parser.add_argument('-num_layers',  dest='num_layers',          default=3,          type=int,           help='Number of GCN Layers')
    parser.add_argument('-lr_decay_step',   dest='lr_decay_step',   default=50,         type=int,           help='lr decay step')
    parser.add_argument('-lr_decay_factor', dest='lr_decay_factor', default=0.5,        type=float,         help='lr decay factor')
    parser.add_argument('-eval_expl',        dest='eval_expl',                default=False,      type=bool,          help='whether evaluating expl')
    parser.add_argument('-batch',       dest='batch_size',          default=128,        type=int,           help='Batch size')
    parser.add_argument('-hid_dim',     dest='hid_dim',             default=64,         type=int,           help='hidden dims')
    parser.add_argument('-dropout_att', dest='dropout_att',         default=0.1,        type=float,         help='dropout on attention scores')
    parser.add_argument('-lr',          dest='lr',                  default=0.01,       type=float,         help='Learning rate')
    parser.add_argument('-ratio',       dest='ratio',               default=0.5,        type=float,         help='ratio')
    parser.add_argument('-num_clusters',dest='num_clusters',        default=2,          type=int,           help='num_clusters')
    parser.add_argument('-heads',       dest='heads',               default=5,          type=int,           help='heads')
    parser.add_argument('-lambda1',     dest='lambda1',             default=10,         type=float,         help='lambda1')
    parser.add_argument('-lambda2',     dest='lambda2',             default=0.001,      type=float,         help='lambda2')
    parser.add_argument('-gcn_normalize',   dest='gcn_normalize',   default=False,      type=bool,          help='gcn_normalize')
    
    
    parser.add_argument('-folds',       dest='folds',               default=10,         type=int,               help='Cross validation folds (with 1 it only splits train/val/test)')

    parser.add_argument('-name',        dest='name',                default='test_'+str(uuid.uuid4())[:8],  help='Name of the run')
    parser.add_argument('-gpu',         dest='gpu',                 default='0',                            help='GPU to use')
    parser.add_argument('-restore',     dest='restore',             action='store_true',                    help='Model restoring')
    
    args = parser.parse_args()
    if not args.restore:
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    logger = Logger(args.name)

    print('Starting runs...')
    print(args)

    
    # get 20 run average
    seeds = [8971, 85688, 9467, 32830, 28689, 94845, 69840, 50883, 74177, 79585, 1055, 75631, 6825, 93188, 95426, 54514, 31467, 70597, 71149, 81994]
    counter = 0
    args.log_db = args.name
    print("log_db:", args.log_db)

    params = vars(args)
    params['seeds'] = str(seeds)
    logger.set_values(params)

    avg_val = []
    avg_test = []
    for seed in seeds:
        args.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_gpu(args.gpu)

        args.name = '{}_run_{}'.format(args.log_db, counter)

        # start training the model
        model       = Trainer(args)
        val_acc, test_acc   = model.run()
        print('For seed {}\t Val Accuracy: {:.3f} \t Test Accuracy: {:.3f}\n'.format(seed, val_acc, test_acc))
        avg_val.append(val_acc)
        avg_test.append(test_acc)
        counter += 1

    logger.set_value(f"val/acc_mean", np.mean(avg_val))
    logger.set_value(f"val/acc_std", np.std(avg_val))
    logger.set_value(f"test/acc_mean", np.mean(avg_test))
    logger.set_value(f"test/acc_std", np.std(avg_test))
    print('Val Accuracy: {:.3f} ± {:.3f} Test Accuracy: {:.3f} ± {:.3f}'.format(np.mean(avg_val), np.std(avg_val), np.mean(avg_test), np.std(avg_test)))


    logger.save()
