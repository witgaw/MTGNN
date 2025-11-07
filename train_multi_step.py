import argparse
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from net import MTGNNModel, gtnet
from trainer import Trainer
from util import *


@dataclass
class TrainingConfig:
    """Public API configuration for MTGNN training via train_injected()."""
    # Model architecture
    num_nodes: int
    gcn_depth: int = 2
    dropout: float = 0.3
    subgraph_size: int = 20
    node_dim: int = 40
    dilation_exponential: int = 1
    conv_channels: int = 32
    residual_channels: int = 32
    skip_channels: int = 64
    end_channels: int = 128
    in_dim: int = 2
    seq_in_len: int = 12
    seq_out_len: int = 12
    layers: int = 3
    propalpha: float = 0.05
    tanhalpha: float = 3.0

    # Training parameters
    batch_size: int = 64
    val_batch_size: int | None = None  # If None, uses batch_size
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    clip: int = 5
    step_size1: int = 2500
    step_size2: int = 100

    # Behavior flags
    gcn_true: bool = True
    buildA_true: bool = True
    cl: bool = True

    # Device and behavior
    device: str = 'cpu'
    seed: int = 101
    print_every: int = 50
    save: str = './save/'  # Directory for saving model checkpoints during training


@dataclass
class InternalTrainingConfig(TrainingConfig):
    """Internal configuration that extends TrainingConfig with CLI-specific parameters."""
    # Paths (only needed for CLI mode)
    data: str = 'data/METR-LA'
    adj_data: str = 'data/sensor_graph/adj_mx.pkl'
    save: str = './save/'
    
    # Other CLI parameters
    expid: int = 1
    num_split: int = 1


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main(runid):
    """Entry point for CLI-based training."""
    # Convert argparse namespace to InternalTrainingConfig
    config = InternalTrainingConfig(
        num_nodes=args.num_nodes,
        gcn_depth=args.gcn_depth,
        dropout=args.dropout,
        subgraph_size=args.subgraph_size,
        node_dim=args.node_dim,
        dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        in_dim=args.in_dim,
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
        layers=args.layers,
        propalpha=args.propalpha,
        tanhalpha=args.tanhalpha,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clip=args.clip,
        step_size1=args.step_size1,
        step_size2=args.step_size2,
        gcn_true=args.gcn_true,
        buildA_true=args.buildA_true,
        cl=args.cl,
        device=args.device,
        seed=args.seed,
        print_every=args.print_every,
        data=args.data,
        adj_data=args.adj_data,
        save=args.save,
        expid=args.expid,
        num_split=args.num_split
    )
    
    return _train_internal(config, runid, injected_data=None, injected_adj=None, 
                          seed_deterministic=False, return_wrapper=False)


def train_injected(config: TrainingConfig, injected_data, injected_adj):
    """Public API for programmatic training with injected data."""
    if injected_data is None:
        raise ValueError("injected_data is required for train_injected()")
    if injected_adj is None:
        raise ValueError("injected_adj is required for train_injected()")

    # Extend the public config with internal defaults
    internal_config = InternalTrainingConfig(
        **asdict(config),  # Copy all fields from TrainingConfig (includes save path)
        # These are only needed internally and have sensible defaults:
        data='',  # Not used when injected_data is provided
        adj_data='',  # Not used when injected_adj is provided
        expid=1,
        num_split=1
    )

    return _train_internal(internal_config, runid=0, injected_data=injected_data, 
                          injected_adj=injected_adj, seed_deterministic=True, 
                          return_wrapper=True)


def _train_internal(config: InternalTrainingConfig, runid: int, injected_data=None, 
                   injected_adj=None, seed_deterministic=False, return_wrapper=False):
    """
    Internal training logic used by both CLI and programmatic interfaces.
    
    Args:
        config: InternalTrainingConfig with all training parameters
        runid: Run identifier for saving models
        injected_data: Optional pre-loaded data (dict with train/val/test splits)
        injected_adj: Optional pre-loaded adjacency matrix
        seed_deterministic: Whether to set random seeds for reproducibility
        return_wrapper: Whether to return MTGNNModel wrapper or raw results dict
    
    Returns:
        Either MTGNNModel wrapper or dict with results depending on return_wrapper
    """
    if seed_deterministic:
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(config.seed)

    # Create save directory if it doesn't exist
    import os
    if config.save:
        os.makedirs(config.save, exist_ok=True)

    #load data
    device = torch.device(config.device)

    val_batch_size = config.val_batch_size if config.val_batch_size is not None else config.batch_size
    dataloader = load_dataset(config.data, config.batch_size, val_batch_size, val_batch_size, injected_data=injected_data)
    scaler = dataloader['scaler']

    predefined_A = load_adj(config.adj_data, adj_data=injected_adj)
    predefined_A = torch.tensor(predefined_A)-torch.eye(config.num_nodes)
    predefined_A = predefined_A.to(device)

    # if config.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    model = gtnet(config.gcn_true, config.buildA_true, config.gcn_depth, config.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=config.dropout, subgraph_size=config.subgraph_size,
                  node_dim=config.node_dim,
                  dilation_exponential=config.dilation_exponential,
                  conv_channels=config.conv_channels, residual_channels=config.residual_channels,
                  skip_channels=config.skip_channels, end_channels= config.end_channels,
                  seq_length=config.seq_in_len, in_dim=config.in_dim, out_dim=config.seq_out_len,
                  layers=config.layers, propalpha=config.propalpha, tanhalpha=config.tanhalpha, layer_norm_affline=True)

    print(config)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, config.learning_rate, config.weight_decay, config.clip, config.step_size1, config.seq_out_len, scaler, device, config.cl)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    for i in range(1,config.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            if iter%config.step_size2==0:
                perm = np.random.permutation(range(config.num_nodes))
            num_sub = int(config.num_nodes/config.num_split)
            for j in range(config.num_split):
                if j != config.num_split-1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                metrics = engine.train(tx, ty[:,0,:,:],id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
            if iter % config.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)

        if mvalid_loss<minl:
            torch.save(engine.model.state_dict(), config.save + "exp" + str(config.expid) + "_" + str(runid) +".pth")
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # Clear GPU memory after training
    if device.type == 'cuda':
        torch.cuda.empty_cache()


    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(config.save + "exp" + str(config.expid) + "_" + str(runid) +".pth"))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    #valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val'])
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds.squeeze().cpu())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    # Move tensors to device only for metric computation
    yhat = yhat.to(device)
    realy = realy.to(device)
    pred = scaler.inverse_transform(yhat)
    vmae, vmape, vrmse = metric(pred,realy)

    # Clean up validation tensors and clear GPU memory
    del yhat, realy, pred
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test'])
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze().cpu())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(config.seq_out_len):
        # Move to device only for this horizon's computation
        yhat_i = yhat[:, :, i].to(device)
        real_i = realy[:, :, i].to(device)
        pred = scaler.inverse_transform(yhat_i)
        metrics = metric(pred, real_i)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])

    # Get the learned adaptive adjacency matrix if available
    learned_adj = None
    if config.buildA_true:
        with torch.no_grad():
            learned_adj = engine.model.gc(engine.model.idx).cpu().numpy()

    # Store only the public TrainingConfig (not internal paths/settings)
    # Extract just the TrainingConfig fields
    public_config_dict = {
        field.name: getattr(config, field.name) 
        for field in TrainingConfig.__dataclass_fields__.values()
    }

    # Return results including the trained model and learned adjacency matrix
    results = {
        'vmae': vmae,
        'vmape': vmape,
        'vrmse': vrmse,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'model': engine.model,
        'scaler': scaler,
        'learned_adj': learned_adj,
        'config': public_config_dict  # Only save public config
    }

    # Optionally wrap in MTGNNModel for easier saving/loading
    if return_wrapper:
        model_wrapper = MTGNNModel.from_training_results(results)
        # Attach metrics to the wrapper for convenience
        model_wrapper.metrics = {
            'vmae': vmae,
            'vmape': vmape,
            'vrmse': vrmse,
            'mae': mae,
            'mape': mape,
            'rmse': rmse
        }
        return model_wrapper

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--device',type=str,default='cuda:1',help='')
    parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')

    parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
    parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

    parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
    parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--subgraph_size',type=int,default=20,help='k')
    parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
    parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

    parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
    parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
    parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
    parser.add_argument('--end_channels',type=int,default=128,help='end channels')


    parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
    parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
    parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

    parser.add_argument('--layers',type=int,default=3,help='number of layers')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('--clip',type=int,default=5,help='clip')
    parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
    parser.add_argument('--step_size2',type=int,default=100,help='step_size')


    parser.add_argument('--epochs',type=int,default=100,help='')
    parser.add_argument('--print_every',type=int,default=50,help='')
    parser.add_argument('--seed',type=int,default=101,help='random seed')
    parser.add_argument('--save',type=str,default='./save/',help='save path')
    parser.add_argument('--expid',type=int,default=1,help='experiment id')

    parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
    parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

    parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

    parser.add_argument('--runs',type=int,default=10,help='number of runs')

    args = parser.parse_args()
    torch.set_num_threads(3)

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    for i in range(args.runs):
        results = main(i)
        vmae.append(results['vmae'])
        vmape.append(results['vmape'])
        vrmse.append(results['vrmse'])
        mae.append(results['mae'])
        mape.append(results['mape'])
        rmse.append(results['rmse'])

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)

    smae = np.std(mae,0)
    smape = np.std(mape,0)
    srmse = np.std(rmse,0)

    print('\n\nResults for 10 runs\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))





