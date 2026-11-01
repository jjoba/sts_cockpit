'''
    Trains model to predict how likely a given deck is to win

    The network has one output (head) per milestone of a run, all sharing the hidden layers (see HEAD_NAMES / process_data.py LABEL_COLS):
    beat the act 1 boss, act 2 boss, first act 3 boss, second act 3 boss (A20 only) and kill the Heart. Heart kill is the last head and the one the cockpit uses and every
    metric below is reported for. The other heads are extra training signal (far more positives than the Heart). A row already past a
    milestone, or for one the run doesn't have (the second act 3 boss below A20), has label -1 for it, which is left out of that head's loss.
    head_weights sets the weight of each head in the loss, a weight of 0 turns a head off, so head_weights=[0,0,0,0,1] is the old single target model.

    Everything needed to compare model / architecture choices is written to model_objects/dev/<model_version>/:
      config.json       settings, data stats, git commit, parameter count
      baselines.json    what simple models score on the same finetuning test rows (is the network moving the needle?)
      metrics.csv       one row per epoch: train vs test gap, ranking (AUC, average precision, lift), calibration, grad / weight norms
      segments.csv      per epoch finetuning test metrics by character and floor range
      calibration.csv   per epoch predicted vs actual win rate by prediction decile
      summary.json      best epochs and final numbers
      learning_curves.png
    and one line per run is appended to model_objects/experiments.csv so runs can be compared side by side.

    Usage (from the project root):
      python scripts/train_model.py
      python scripts/train_model.py --set model_version=fnn_v17_wide hidden_width=1400 dropout=0.3
      python scripts/train_model.py --set model_version=fnn_v17_no_pretrain stages=finetuning
    Any key in DEFAULT_CONFIG can be overridden with --set key=value (values are read as json where possible).
'''
import torch
import os
import sys
import json
import shutil
import subprocess

from time import time

import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# Cards + relics + characters + floor, must match process_data.py and starting_decks.csv
N_FEATURES = 929

# Heads of the network, in the order process_data.py writes the label columns that follow the features in every row
HEAD_NAMES = ['beat_act1_boss', 'beat_act2_boss', 'beat_act3_boss', 'beat_act3_second_boss', 'heart_kill']
N_HEADS = len(HEAD_NAMES)
HEART = N_HEADS - 1
N_COLS = N_FEATURES + N_HEADS

# Last feature columns of a row are floor, then the one hot characters (see process_data.py)
FLOOR_COL = N_FEATURES - 5
CHARACTERS = ['defect', 'ironclad', 'the_silent', 'watcher']
CHARACTER_COLS = list(range(N_FEATURES - 4, N_FEATURES))

# Floor ranges metrics are broken out by
FLOOR_BUCKETS = [('floor_01_10', 1, 10), ('floor_11_25', 11, 25), ('floor_26_40', 26, 40), ('floor_41_plus', 41, 127)]

DEFAULT_CONFIG = {
    'model_version': 'fnn_v20',
    'seed': 0,

    # Architecture: n_hidden layers of hidden_width units (ReLU, dropout) then a sigmoid output
    'n_hidden': 4,
    'hidden_width': N_FEATURES,
    'dropout': 0.5,

    # How upgraded cards enter the network: split = a copy is in either card (plain) or card_u (upgraded), cumulative = an upgraded copy is in BOTH, so card is
    # the total copies and card_u how many of them are upgraded (what Searing Blow's columns already do). The data and the cockpit inputs stay as they are,
    # the model adds the upgraded columns onto their plain ones as its first step
    'upgrade_encoding': 'split',

    # Regularization: l2 is the optimizer weight decay, l1 is a penalty on the absolute value of the weights added to the loss
    'l1': 0.0,
    'l2': 0.0,

    # Optimization
    'optimizer': 'sgd', # sgd or adam
    'lr': 1e-2,
    'momentum': 0.0, # sgd only
    'batch_size': 64,

    # Learning rate over each stage: constant, cosine or linear decay from lr down to lr * lr_final_factor
    'lr_schedule': 'constant',
    'lr_final_factor': 0.01,

    # Start from the weights of a saved TorchScript model (same n_hidden / hidden_width), e.g. a pre training checkpoint to compare finetuning settings
    'init_from': None,

    # Weight of each head (HEAD_NAMES order) in the loss, which is the weighted average of the heads' mean log losses over their labelled rows. 0 turns a head off
    'head_weights': [1.0, 1.0, 1.0, 1.0, 1.0],

    # Optional head weights for individual stages, e.g. {"pre_training_beta": [0, 0, 1, 0, 0]} pretrains on beating the first act 3 boss only. Stages not listed use head_weights
    'stage_head_weights': {},

    # Which data sets to train on in order, and for how many epochs each. Set stages=finetuning to see if pre training helps
    'stages': ['pre_training_alpha', 'pre_training_beta', 'finetuning'],
    'epochs': {'pre_training_alpha': 8, 'pre_training_beta': 20, 'finetuning': 20},

    # Finetuning epoch that goes to production is the best one on this finetuning test metric
    'select_metric': 'log_loss', # log_loss, brier, ece (lower is better) or auc, avg_precision, lift_top_decile (higher is better)

    # Rows used for evaluation each epoch (fixed random subsets of the test and train files)
    'eval_rows': 500000,
    'ft_eval_rows': 2000000,
    'train_eval_rows': 200000,

    # Caps the rows seen per epoch (a fresh random subset each epoch), for quick architecture trials. null = the whole file
    'max_train_rows': None,

    'baselines': True,
    'save_production': True,

    # False when several runs go at once (the experiments file is not safe to append to in parallel)
    'log_experiment': True,
}

LOWER_IS_BETTER = {'log_loss', 'brier', 'ece'}

class BinDataset(Dataset):
    # Rows are fixed width int8 (features then one label per head) written by process_data.py, no header, so the file is memory mapped
    # The OS pages rows in as they are asked for, so shuffling works on files far larger than RAM
    def __init__(self, file_path):
        self.data = np.memmap(file_path, dtype = np.int8, mode = 'r').reshape(-1, N_COLS)

    def __len__(self):
        return self.data.shape[0]

    def take(self, idx):
        # idx should be sorted, which keeps the reads close to sequential
        rows = torch.from_numpy(np.asarray(self.data[idx]))
        return rows[:, :N_FEATURES], rows[:, N_FEATURES:]

class EvalSet:
    # A fixed random subset of a data set, labels and segment columns are read once and the features are read at eval time
    def __init__(self, dataset, max_rows, seed):
        rng = np.random.default_rng(seed)
        n = len(dataset)
        self.dataset = dataset
        self.idx = np.sort(rng.choice(n, min(n, max_rows), replace = False))

        meta = np.asarray(dataset.data[self.idx][:, [FLOOR_COL] + CHARACTER_COLS + list(range(N_FEATURES, N_COLS))])
        self.floor = meta[:, 0].astype(np.int16)
        self.character = meta[:, 1:5].argmax(axis = 1)
        self.labels = meta[:, 5:].astype(np.int8) # every head's label, -1 where the row is past that milestone
        self.y = self.labels[:, HEART] # Heart kill, what all the metrics are about

# Define model. Output is one probability per head (HEAD_NAMES order), the Heart kill is the last
class NeuralNetwork(nn.Module):
    def __init__(self, n_hidden = 4, hidden_width = N_FEATURES, dropout = 0.5):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        width_in = N_FEATURES
        for _ in range(n_hidden):
            layers += [nn.Linear(width_in, hidden_width), nn.ReLU(), nn.Dropout(dropout)]
            width_in = hidden_width

        layers += [nn.Linear(width_in, N_HEADS), nn.Sigmoid()]
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CumulativeUpgradeNetwork(NeuralNetwork):
    # Same network, but each upgraded copy is also counted in its plain column before the first layer (base_idx[i] is the plain column of upgraded column up_idx[i])
    def __init__(self, base_idx, up_idx, n_hidden = 4, hidden_width = N_FEATURES, dropout = 0.5):
        super().__init__(n_hidden, hidden_width, dropout)
        self.register_buffer('base_idx', torch.tensor(base_idx, dtype = torch.long))
        self.register_buffer('up_idx', torch.tensor(up_idx, dtype = torch.long))

    def forward(self, x):
        x = self.flatten(x)
        x = x.index_add(1, self.base_idx, x.index_select(1, self.up_idx))
        return self.linear_relu_stack(x)

def upgrade_column_pairs(project_dir):
    # (plain column, upgraded column) of every card with a card_u column. Searing Blow is left out, its columns are cumulative already
    with open(f'{project_dir}/data/master_lists/master_card_list.txt') as f:
        cards = [line.strip() for line in f]
    with open(f'{project_dir}/data/master_lists/master_relic_list.txt') as f:
        names = cards + [line.strip() for line in f]

    index = {name: i for i, name in enumerate(names)}
    pairs = [(index[name[:-2]], i) for i, name in enumerate(names) if name.endswith('_u') and not name.startswith('searingblow') and name[:-2] in index]
    return [b for b, _ in pairs], [u for _, u in pairs]

############### METRICS ##############

def calibration_bins(y, p, n_bins = 10):
    # Equal frequency bins of predictions: mean predicted vs actual win rate
    order = np.argsort(p)
    rows = []
    for b, chunk in enumerate(np.array_split(order, n_bins)):
        rows.append({'bin': b, 'n': len(chunk), 'mean_pred': p[chunk].mean(), 'actual': y[chunk].mean()})
    return rows

def compute_metrics(y, p, base_rate = None):
    # base_rate = win rate of the training data, gives the log loss of always predicting the base rate to measure skill against
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    out = {'n': len(y), 'win_rate': y.mean(), 'mean_pred': p.mean(), 'pred_std': p.std()}

    out['log_loss'] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    out['brier'] = float(np.mean((p - y) ** 2))

    bins = calibration_bins(y, p)
    out['ece'] = float(sum(b['n'] * abs(b['mean_pred'] - b['actual']) for b in bins) / len(y))

    if base_rate is not None:
        const_loss = -(y.mean() * np.log(base_rate) + (1 - y.mean()) * np.log(1 - base_rate))
        out['log_loss_skill'] = float(1 - out['log_loss'] / const_loss)

    out['acc_at_0.5'] = float(((p > 0.5) == y).mean())

    if 0 < y.sum() < len(y):
        out['auc'] = float(roc_auc_score(y, p))
        out['avg_precision'] = float(average_precision_score(y, p))

        precision, recall, thresholds = precision_recall_curve(y, p)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        out['best_f1'] = float(f1.max())
        out['best_f1_threshold'] = float(thresholds[f1.argmax()])

        # Share of all wins found in the top 10% of predictions, 1 = random, 10 = perfect for a 10% win rate
        top = np.argsort(-p)[:max(1, len(p) // 10)]
        out['lift_top_decile'] = float(y[top].sum() / y.sum() / 0.1)
    else:
        for k in ['auc', 'avg_precision', 'best_f1', 'best_f1_threshold', 'lift_top_decile']:
            out[k] = np.nan

    return out, bins

@torch.no_grad()
def predict(model, dataset, idx, device, chunk = 65536):
    # Probabilities of every head, shape rows x N_HEADS
    model.eval()
    preds = []
    for i in range(0, len(idx), chunk):
        X, _ = dataset.take(idx[i:i + chunk])
        preds.append(model(X.float().to(device)).cpu().numpy())
    return np.concatenate(preds)

def head_metrics(eval_set, p_all, prefix):
    # Log loss and AUC of the milestone heads on the rows still ahead of the milestone (Heart is reported by the main metrics)
    out = {}
    for h, name in enumerate(HEAD_NAMES[:HEART]):
        valid = eval_set.labels[:, h] >= 0
        if not valid.any():
            out[f'{prefix}{name}_log_loss'] = out[f'{prefix}{name}_auc'] = np.nan
            continue

        y, p = eval_set.labels[valid, h], np.clip(p_all[valid, h].astype(np.float64), 1e-7, 1 - 1e-7)
        out[f'{prefix}{name}_log_loss'] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        out[f'{prefix}{name}_auc'] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
    return out

def segment_masks(eval_set):
    masks = {'all': np.ones(len(eval_set.y), dtype = bool)}
    for i, name in enumerate(CHARACTERS):
        masks[f'char_{name}'] = eval_set.character == i
    for name, low, high in FLOOR_BUCKETS:
        masks[name] = (eval_set.floor >= low) & (eval_set.floor <= high)
    return masks

############### BASELINES ##############

def read_rows(dataset, n_rows, seed):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), min(len(dataset), n_rows), replace = False))
    return np.asarray(dataset.data[idx])

def run_baselines(ft_train, ft_eval, base_rate, out_dir):
    # Simple models scored on the same rows as the network. If the network can't beat these it is not learning much about decks
    results = {}
    y = ft_eval.y

    results['constant_base_rate'], _ = compute_metrics(y, np.full(len(y), base_rate), base_rate)

    # Win rate by character and floor range only, no deck information at all
    train_rows = read_rows(ft_train, 1000000, 1)
    def segment_ids(floor, character):
        bucket = np.digitize(floor, [11, 26, 41])
        return character * 4 + bucket

    train_ids = segment_ids(train_rows[:, FLOOR_COL], train_rows[:, CHARACTER_COLS].argmax(axis = 1))
    rate_by_segment = np.array([train_rows[train_ids == s, -1].mean() if (train_ids == s).any() else base_rate for s in range(16)])
    results['character_floor_lookup'], _ = compute_metrics(y, rate_by_segment[segment_ids(ft_eval.floor, ft_eval.character)], base_rate)

    # Logistic regression on the full feature set (floor scaled so it is on a similar scale to card counts)
    fit_rows = train_rows[:500000]
    def features(rows):
        X = rows[:, :N_FEATURES].astype(np.float32)
        X[:, FLOOR_COL] /= 50
        return X

    start = time()
    log_reg = LogisticRegression(max_iter = 300)
    log_reg.fit(features(fit_rows), fit_rows[:, -1])

    X_eval = np.concatenate([features(np.asarray(ft_eval.dataset.data[ft_eval.idx[i:i + 200000]])) for i in range(0, len(ft_eval.idx), 200000)])
    results['logistic_regression'], _ = compute_metrics(y, log_reg.predict_proba(X_eval)[:, 1], base_rate)
    results['logistic_regression']['fit_seconds'] = time() - start

    with open(f'{out_dir}/baselines.json', 'w') as f:
        json.dump(results, f, indent = 2, default = float)

    for name, m in results.items():
        print(f'baseline {name}: log_loss {m["log_loss"]:.4f}, auc {m["auc"]:.4f}, avg_precision {m["avg_precision"]:.4f}')

    return results

############### TRAINING ##############

def lr_at(config, progress):
    # progress = share of the stage's training completed, 0 to 1
    floor = config['lr_final_factor']
    if config['lr_schedule'] == 'cosine':
        scale = floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * progress))
    elif config['lr_schedule'] == 'linear':
        scale = 1 - (1 - floor) * progress
    else:
        scale = 1.0
    return config['lr'] * scale

def train_epoch(model, dataset, optimizer, loss_fn, config, device, rng, epoch, n_epochs, head_weights):
    model.train()
    perm = rng.permutation(len(dataset))[:config['max_train_rows']]
    batch_size = config['batch_size']

    head_weights = torch.tensor(head_weights, dtype = torch.float32, device = device)
    head_bce_sum = torch.zeros(N_HEADS, device = device)
    head_rows = torch.zeros(N_HEADS, device = device)
    loss_sum = torch.zeros((), device = device)
    n_batches = 0
    grad_norms = []

    for i in range(0, len(perm), batch_size):
        if config['lr_schedule'] != 'constant':
            for group in optimizer.param_groups:
                group['lr'] = lr_at(config, (epoch + i / len(perm)) / n_epochs)

        # Sorting inside a batch doesn't change the batch but keeps the disk reads closer together
        X, y = dataset.take(np.sort(perm[i:i + batch_size]))
        X, y = X.to(device), y.to(device)

        pred = model(X.float())

        # Log loss of each head over the rows with a label for it (-1 = already past the milestone), then the weighted average of the heads
        labelled = (y >= 0).float()
        bce = loss_fn(pred, y.clamp(min = 0).float()) * labelled
        head_mean = bce.sum(0) / labelled.sum(0).clamp(min = 1)
        loss = (head_weights * head_mean).sum() / head_weights.sum()

        if config['l1'] > 0:
            loss = loss + config['l1'] * sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)

        loss.backward()

        # Gradient norm is sampled because reading it forces a sync with the gpu
        if n_batches % 200 == 0:
            grad_norms.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))))

        optimizer.step()
        optimizer.zero_grad()

        head_bce_sum += bce.detach().sum(0)
        head_rows += labelled.sum(0)
        loss_sum += loss.detach()
        n_batches += 1

        if n_batches % 20000 == 0:
            print(f'  batch {n_batches}/{len(perm) // batch_size}, running loss {float(loss_sum) / n_batches:.5f}, heart {float(head_bce_sum[HEART] / head_rows[HEART]):.5f}')

    weight_norm = float(torch.sqrt(sum((p ** 2).sum() for p in model.parameters() if p.dim() > 1)))
    # train_loss_running is the Heart head's, comparable with single target runs. train_loss_total is what is optimized
    head_loss = (head_bce_sum / head_rows.clamp(min = 1)).cpu().numpy()
    return {'train_loss_running': float(head_loss[HEART]), 'train_loss_total': float(loss_sum) / n_batches,
            **{f'train_loss_{name}': float(head_loss[h]) for h, name in enumerate(HEAD_NAMES[:HEART])}, 'grad_norm_mean': float(np.mean(grad_norms)), 'weight_norm': weight_norm}

def parse_overrides(argv):
    config = json.loads(json.dumps(DEFAULT_CONFIG))

    if '--set' in argv:
        for item in argv[argv.index('--set') + 1:]:
            key, value = item.split('=', 1)
            if key not in config:
                raise SystemExit(f'Unknown setting {key}, options are {list(config)}')

            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass

            # stages=finetuning or stages=pre_training_beta,finetuning
            if key == 'stages' and isinstance(value, str):
                value = value.split(',')

            config[key] = value

    return config

def git_commit(project_dir):
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd = project_dir, text = True).strip()
    except Exception:
        return 'unknown'

def save_learning_curves(metrics, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize = (13, 8))
    panels = [('ft_log_loss', 'Finetuning test log loss'), ('ft_auc', 'Finetuning test AUC'),
              ('ft_avg_precision', 'Finetuning test average precision'), ('ft_ece', 'Finetuning test calibration error')]

    boundaries = metrics.groupby('stage')['global_epoch'].min().sort_values()

    for ax, (col, title) in zip(axes.flatten(), panels):
        ax.plot(metrics['global_epoch'], metrics[col], marker = 'o', markersize = 3, label = 'test')

        if col == 'ft_log_loss' and 'train_eval_log_loss' in metrics:
            ax.plot(metrics['global_epoch'], metrics['train_eval_log_loss'], marker = 'o', markersize = 3, label = 'train (current stage data)')
            ax.legend()

        for stage, start in boundaries.items():
            ax.axvline(start - 0.5, color = 'gray', linestyle = ':')
            ax.text(start, ax.get_ylim()[1], stage.replace('pre_training_', 'pre_'), fontsize = 7, va = 'top')

        ax.set_title(title)
        ax.set_xlabel('epoch')

    fig.tight_layout()
    fig.savefig(f'{out_dir}/learning_curves.png', dpi = 120)
    plt.close(fig)

############### RUN ##############

if __name__ == '__main__':

    project_dir = os.getcwd()
    config = parse_overrides(sys.argv[1:])
    version = config['model_version']

    for weights in [config['head_weights']] + list(config['stage_head_weights'].values()):
        if len(weights) != N_HEADS or min(weights) < 0 or sum(weights) <= 0:
            raise SystemExit(f'head weights need {N_HEADS} non negative weights (one per head in {HEAD_NAMES}), not all 0, not {weights}')

    # Makes directory for model training if it doesn't exist already
    out_dir = f'{project_dir}/model_objects/dev/{version}'
    os.makedirs(out_dir, exist_ok = True)

    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])

    # Where is training to take place?
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f'Using device: {device}')

    # Instantiate Model
    if config['upgrade_encoding'] == 'cumulative':
        base_idx, up_idx = upgrade_column_pairs(project_dir)
        print(f'Cumulative upgrade encoding: {len(up_idx)} upgraded columns added onto their plain columns')
        model = CumulativeUpgradeNetwork(base_idx, up_idx, config['n_hidden'], config['hidden_width'], config['dropout'])
    elif config['upgrade_encoding'] == 'split':
        model = NeuralNetwork(config['n_hidden'], config['hidden_width'], config['dropout'])
    else:
        raise SystemExit(f"upgrade_encoding is split or cumulative, not {config['upgrade_encoding']}")
    if config['init_from']:
        saved = torch.jit.load(f"{project_dir}/{config['init_from']}", map_location = 'cpu').state_dict()
        state = model.state_dict()

        # A single target checkpoint has a one row output layer: it becomes the Heart head, the other heads start from the fresh initialisation
        for key, value in saved.items():
            if state[key].shape == value.shape:
                state[key] = value
            elif value.shape[0] == 1 and state[key].shape[1:] == value.shape[1:]:
                state[key][HEART] = value[0]
                print(f'  {key}: single target weights copied to the Heart head')
            else:
                raise SystemExit(f'init_from does not fit the model: {key} is {tuple(value.shape)}, expected {tuple(state[key].shape)}')

        model.load_state_dict(state)
        print(f"Initialised from {config['init_from']}")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {n_params:,} parameters')

    # Setup optimization
    loss_fn = nn.BCELoss(reduction = 'none') # per head, averaged in train_epoch
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], weight_decay = config['l2'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = config['lr'], momentum = config['momentum'], weight_decay = config['l2'])

    # Data and fixed evaluation subsets. Finetuning test is what every epoch is judged on, whichever stage is training
    data_dir = f'{project_dir}/data/training_data'
    ft_train = BinDataset(f'{data_dir}/finetuning/train/train.bin')
    ft_test = BinDataset(f'{data_dir}/finetuning/test/test.bin')
    ft_eval = EvalSet(ft_test, config['ft_eval_rows'], 100)

    ft_base_rate = float(read_rows(ft_train, 1000000, 2)[:, -1].mean())

    data_stats = {}
    for d in dict.fromkeys(config['stages'] + ['finetuning']):
        for split in ['train', 'test']:
            ds = BinDataset(f'{data_dir}/{d}/{split}/{split}.bin')
            sample = read_rows(ds, 200000, 3)
            labels = sample[:, N_FEATURES:]
            data_stats[f'{d}_{split}'] = {'rows': len(ds), 'win_rate': float(sample[:, -1].mean()),
                                          'positive_rate_by_head': {name: float(labels[labels[:, h] >= 0, h].mean()) if (labels[:, h] >= 0).any() else None for h, name in enumerate(HEAD_NAMES)},
                                          'share_past_milestone_by_head': {name: float((labels[:, h] < 0).mean()) for h, name in enumerate(HEAD_NAMES)}}

    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump({'config': config, 'n_params': n_params, 'device': device, 'torch': torch.__version__, 'git_commit': git_commit(project_dir),
                   'started': time(), 'data': data_stats, 'ft_eval_rows': len(ft_eval.y)}, f, indent = 2)

    baselines = run_baselines(ft_train, ft_eval, ft_base_rate, out_dir) if config['baselines'] else {}

    metric_rows = []
    segment_rows = []
    calibration_rows = []
    global_epoch = 0
    run_start = time()

    for destination in config['stages']:
        print(f'Training on destination: {destination}')

        training_data = BinDataset(f'{data_dir}/{destination}/train/train.bin')
        testing_data = BinDataset(f'{data_dir}/{destination}/test/test.bin')

        train_eval = EvalSet(training_data, config['train_eval_rows'], 200)
        own_eval = ft_eval if destination == 'finetuning' else EvalSet(testing_data, config['eval_rows'], 300)
        own_base_rate = ft_base_rate if destination == 'finetuning' else float(read_rows(training_data, 500000, 4)[:, -1].mean())

        for t in range(config['epochs'][destination]):
            print(f"{destination} epoch {t + 1}/{config['epochs'][destination]}\n-------------------------------")
            epoch_start = time()

            row = {'stage': destination, 'epoch': t, 'global_epoch': global_epoch}
            row.update(train_epoch(model, training_data, optimizer, loss_fn, config, device, rng, t, config['epochs'][destination],
                                   config['stage_head_weights'].get(destination, config['head_weights'])))
            row['epoch_seconds'] = time() - epoch_start
            row['rows_per_second'] = min(len(training_data), config['max_train_rows'] or len(training_data)) / row['epoch_seconds']

            # Same data the model trained on, in eval mode (no dropout): the gap to test shows over fitting
            m, _ = compute_metrics(train_eval.y, predict(model, training_data, train_eval.idx, device)[:, HEART], own_base_rate)
            row.update({f'train_eval_{k}': v for k, v in m.items()})

            # Test set of the data being trained on
            if destination != 'finetuning':
                p_own = predict(model, testing_data, own_eval.idx, device)
                m, _ = compute_metrics(own_eval.y, p_own[:, HEART], own_base_rate)
                row.update({f'own_{k}': v for k, v in m.items()})
                row.update(head_metrics(own_eval, p_own, 'own_'))

            # Finetuning test: overall, then by character and floor range, and calibration
            p_ft_all = predict(model, ft_test, ft_eval.idx, device)
            p_ft = p_ft_all[:, HEART]
            row.update(head_metrics(ft_eval, p_ft_all, 'ft_'))
            for name, mask in segment_masks(ft_eval).items():
                m, bins = compute_metrics(ft_eval.y[mask], p_ft[mask], ft_base_rate)
                segment_rows.append({'stage': destination, 'epoch': t, 'global_epoch': global_epoch, 'segment': name, **m})

                if name == 'all':
                    row.update({f'ft_{k}': v for k, v in m.items()})
                    calibration_rows += [{'stage': destination, 'epoch': t, 'global_epoch': global_epoch, **b} for b in bins]

            # Skill relative to simple models on the same rows
            for name, base in baselines.items():
                if name != 'constant_base_rate':
                    row[f'ft_auc_vs_{name}'] = row['ft_auc'] - base['auc']
                    row[f'ft_log_loss_vs_{name}'] = row['ft_log_loss'] - base['log_loss']

            metric_rows.append(row)

            # Save out the model
            model_path = f'{out_dir}/{destination}_epoch_{t}.pth'
            model_scripted = torch.jit.script(model) # Export to TorchScript
            model_scripted.save(model_path) # Save

            print(f"  train loss {row['train_loss_running']:.5f} | finetuning test: log_loss {row['ft_log_loss']:.5f}, auc {row['ft_auc']:.4f}, "
                  f"avg_precision {row['ft_avg_precision']:.4f}, ece {row['ft_ece']:.5f} | {row['epoch_seconds'] / 60:.1f} min")

            # Written every epoch so a crashed or stopped run still leaves its curves
            metrics = pd.DataFrame(metric_rows)
            metrics.to_csv(f'{out_dir}/metrics.csv', index = False)
            pd.DataFrame(segment_rows).to_csv(f'{out_dir}/segments.csv', index = False)
            pd.DataFrame(calibration_rows).to_csv(f'{out_dir}/calibration.csv', index = False)

            global_epoch += 1

    save_learning_curves(metrics, out_dir)

    # Best finetuning epoch by the chosen metric (for runs without finetuning, the last stage)
    final_stage = 'finetuning' if 'finetuning' in config['stages'] else config['stages'][-1]
    stage_metrics = metrics[metrics['stage'] == final_stage].reset_index(drop = True)
    select_col = f"ft_{config['select_metric']}"
    best = stage_metrics[select_col].idxmin() if config['select_metric'] in LOWER_IS_BETTER else stage_metrics[select_col].idxmax()
    best_row = stage_metrics.loc[best]

    print(f"Best {final_stage} epoch was {int(best_row['epoch'])} by {config['select_metric']}")
    print(stage_metrics[['epoch', 'ft_log_loss', 'ft_auc', 'ft_avg_precision', 'ft_lift_top_decile', 'ft_ece']].to_string(index = False))

    summary = {
        'model_version': version,
        'best_stage': final_stage,
        'best_epoch': int(best_row['epoch']),
        'select_metric': config['select_metric'],
        'best_metrics': {k: float(v) for k, v in best_row.items() if k.startswith('ft_') or k.startswith('train_eval_')},
        'baselines': {name: {k: v for k, v in m.items() if k in ['log_loss', 'auc', 'avg_precision', 'ece']} for name, m in baselines.items()},
        'train_minutes': (time() - run_start) / 60,
    }

    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent = 2)

    # One line per run so architectures can be compared across runs
    experiment = {'model_version': version, 'git_commit': git_commit(project_dir), 'n_params': n_params,
                  **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in config.items() if k != 'model_version'},
                  'best_stage': final_stage, 'best_epoch': summary['best_epoch'], 'train_minutes': summary['train_minutes']}
    experiment.update({f'best_{k}': v for k, v in summary['best_metrics'].items() if k in
                       [f'ft_{m}' for m in ['log_loss', 'brier', 'ece', 'auc', 'avg_precision', 'lift_top_decile', 'log_loss_skill', 'best_f1'] + [f'{h}_log_loss' for h in HEAD_NAMES[:HEART]]] + ['train_eval_log_loss', 'train_eval_auc']})
    experiment.update({f'baseline_logreg_{k}': v for k, v in summary['baselines'].get('logistic_regression', {}).items()})

    if config['log_experiment']:
        experiments_path = f'{project_dir}/model_objects/experiments.csv'
        experiment_df = pd.DataFrame([experiment])
        if os.path.exists(experiments_path):
            experiment_df = pd.concat([pd.read_csv(experiments_path), experiment_df], ignore_index = True)
        experiment_df.to_csv(experiments_path, index = False)

    # Move the top performing finetuned model to production
    if config['save_production'] and final_stage == 'finetuning':
        shutil.copy(
            f"{out_dir}/finetuning_epoch_{int(best_row['epoch'])}.pth",
            f'{project_dir}/model_objects/production/{version}.pth')
