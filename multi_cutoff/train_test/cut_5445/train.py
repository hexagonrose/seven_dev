import os
from copy import deepcopy

from tqdm import tqdm
from torch_geometric.loader import DataLoader

import sevenn
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
from sevenn.model_build import build_E3_equivariant_model
import sevenn.util as util
from sevenn._const import DEFAULT_TRAINING_CONFIG
from sevenn.train.trainer import Trainer
from sevenn.error_recorder import ErrorRecorder


working_dir = os.getcwd() # save current path
data_path = '/data2_1/haekwan98/sevennet_tutorial/data'
cutoff = 5.0

dataset_prefix = os.path.join(data_path, 'train')
xyz_files = ['1200K.extxyz', '600K.extxyz']
dataset_files = [os.path.join(dataset_prefix, xyz) for xyz in xyz_files]

# Preprocess(build graphs) data before training. It will automatically saves processed graph to {root}/sevenn_data/train.pt, metadata + statistics as train.yaml
dataset = SevenNetGraphDataset(cutoff=cutoff, root=working_dir, files=dataset_files, processed_name='train.pt')

# print(f'# graphs: {len(dataset)}')
# print(f'# atoms (nodes): {dataset.natoms}')
# print(dataset[0])

# split the dataset into train & valid
num_dataset = len(dataset)
num_train = int(num_dataset * 0.95)
num_valid = num_dataset - num_train

dataset = dataset.shuffle()
train_dataset = dataset[:num_train]
valid_dataset = dataset[num_train:]

print(f'# graphs for training: {len(train_dataset)}')
print(f'# graphs for validation: {len(valid_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

train_shift = dataset.per_atom_energy_std
train_scale = dataset.force_rms
train_conv_denominator = dataset.avg_num_neigh

# copy default model configuration.
model_cfg = deepcopy(DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG)

# tune the channel and lmax parameters. You can experiment with different settings.
model_cfg.update({'channel': 16, 'lmax': 2, 'num_convoltution_layer': 5})

# tell models about element in universe
model_cfg.update(util.chemical_species_preprocess([], universal=True))

# tell model about statistics of dataset. kind of data standardization
model_cfg.update({'shift': train_shift, 'scale': train_scale, 'conv_denominator': train_conv_denominator})
model_cfg['multi_cutoff'] = [5, 4, 4, 4, 5]

model = build_E3_equivariant_model(model_cfg)
num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model) # model info
print(f'# model weights: {num_weights}')

#### start training ####
# copy default training configuration
train_cfg = deepcopy(DEFAULT_TRAINING_CONFIG)

# set optimizer and scheduler for training.
train_cfg.update({
  'device': 'cuda',
  'optimizer': 'adam',
  'optim_param': {'lr': 0.01},
  'scheduler': 'linearlr',
  'scheduler_param': {'start_factor': 1.0, 'total_iters': 100, 'end_factor': 0.0001},
  # 'scheduler': 'exponentiallr',
  # 'scheduler_param': {'gamma': 0.99},
  'force_loss_weight': 0.2,
})

# Initialize trainer. It implements common rountines for training.
trainer = Trainer.from_config(model, train_cfg)
print(trainer.loss_functions)  # We have energy, force, stress loss function by defaults. With default 1.0, 0.1, and 1e-6 loss weight
print(trainer.optimizer)
print(trainer.scheduler)

train_cfg.update({
  # List of tuple [Quantity name, metric name]
  # Supporting quantities: Energy, Force, Stress, Stress_GPa
  # Supporting metrics: RMSE, MAE, Loss
  # TotalLoss is special!
    'error_record': [
        ('Energy', 'RMSE'),
        ('Force', 'RMSE'),
        # ('Stress', 'RMSE'),  We skip stress error cause it is too long to print, uncomment it if you want
        ('TotalLoss', 'None'),
    ]
})
train_recorder = ErrorRecorder.from_config(train_cfg)
valid_recorder = deepcopy(train_recorder)
for metric in train_recorder.metrics:
    print(metric)

valid_best = float('inf')
total_epoch = 100    # you can increase this number for better performance.
pbar = tqdm(range(total_epoch))
config = model_cfg  # to save config used in this tutorial.
config.update(train_cfg)

os.makedirs(os.path.join(working_dir, 'checkpoint'), exist_ok=True)
for epoch in pbar:
    # trainer scans whole data from given loader, and updates error recorder with outputs.
    trainer.run_one_epoch(train_loader, is_train=True, error_recorder=train_recorder)
    trainer.run_one_epoch(valid_loader, is_train=False, error_recorder=valid_recorder)
    trainer.scheduler_step(valid_best) 
    train_err = train_recorder.epoch_forward()  # return averaged error over one epoch, then reset.
    valid_err = valid_recorder.epoch_forward()

    # for print. train_err is a dictionary of {metric name with unit: error}
    err_str = 'Train: ' + '    '.join([f'{k}: {v:.3f}' for k, v in train_err.items()])
    err_str += '// Valid: ' + '    '.join([f'{k}: {v:.3f}' for k, v in valid_err.items()])
    pbar.set_description(err_str)

    if valid_err['TotalLoss'] < valid_best:  # saves best checkpoint. by comparing validation set total loss
        valid_best = valid_err['TotalLoss']
        trainer.write_checkpoint(os.path.join(working_dir, 'checkpoint', 'checkpoint_best.pth'), config=config, epoch=epoch)

    if epoch % 10 == 0:  # save checkpoint every 10 epochs
        trainer.write_checkpoint(os.path.join(working_dir, 'checkpoint', f'checkpoint_{epoch}.pth'), config=config, epoch=epoch)