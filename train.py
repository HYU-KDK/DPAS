#
import argparse
import os
print("DEBUG: Script Started")
import pickle
import pprint

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS
from models.compositional_modules import get_model
from utils import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def train_model(model, optimizer, train_dataset, config, device, start_epoch=0):
    """Function to train the model to predict attributes with cross entropy loss.

    Args:
        model (nn.Module): the model to compute the similarity score with the images.
        optimizer (nn.optim): the optimizer with the learnable parameters.
        train_dataset (CompositionDataset): the train dataset
        config (argparse.ArgumentParser): the config
        device (...): torch device
        start_epoch (int): epoch to start from

    Returns:
        tuple: the trained model (or the best model) and the optimizer
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()

    loss_fn = CrossEntropyLoss()
    #
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).to(device)
    i = 0
    train_losses = []

    torch.autograd.set_detect_anomaly(True)

    for i in range(start_epoch, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            batch_img, batch_target = batch[0], batch[3]
            batch_target = batch_target.to(device)
            batch_img = batch_img.to(device)
            batch_feat = model.encode_image(batch_img)

            logits = model(batch_feat, train_pairs)

            loss = loss_fn(logits, batch_target)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or \
                    (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix(
                {"train loss": np.mean(epoch_train_losses[-50:])}
            )

            progress_bar.update()

        progress_bar.close()
        progress_bar.write(
            f"epoch {i +1} train loss {np.mean(epoch_train_losses)}"
        )
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            save_soft_embeddings(model, config, epoch=i + 1)

    return model, optimizer


def train_dpc(model, optimizer, train_dataset, config, device, start_epoch=0):
    """Function to train the DPC model with DHNO loss (Cross Entropy + Contrastive).
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()

    ce_loss_fn = CrossEntropyLoss()
    
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    
    # [N_pairs, 2]
    train_pairs_tensor = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).to(device)
    
    # Pre-compute masks for Hard Negative Sampling
    # mask_same_obj[i, j] = True if pair i and pair j have same object
    # mask_same_attr[i, j] = True if pair i and pair j have same attribute
    # This might be too large for O(N^2) if N is large. N=1262 for MIT-States. 1262^2 is ~1.6M, which is small.
    
    num_pairs = len(train_pairs_tensor)
    attrs = train_pairs_tensor[:, 0]
    objs = train_pairs_tensor[:, 1]
    
    # Expand dims for broadcasting
    # [N, 1] == [1, N] -> [N, N]
    mask_same_obj = (objs.unsqueeze(1) == objs.unsqueeze(0))
    mask_same_attr = (attrs.unsqueeze(1) == attrs.unsqueeze(0))
    
    # Exclude self (diagonal)
    eye = torch.eye(num_pairs, device=device).bool()
    mask_same_obj = mask_same_obj & (~eye)
    mask_same_attr = mask_same_attr & (~eye)
    
    i = 0
    train_losses = []

    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    
    dhno_lambda = getattr(config, 'dhno_lambda', 0.5) 
    dhno_margin = getattr(config, 'dhno_margin', 1.0)
    entropy_lambda = getattr(config, 'entropy_lambda', 0.0)

    for i in range(start_epoch, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            batch_img, batch_target = batch[0], batch[3]
            batch_target = batch_target.to(device)
            batch_img = batch_img.to(device)
            
            # 1. Forward Pass
            # dpc_next encodes images internally (needs raw images for visual hooks)
            if config.experiment_name == 'dpas':
                logits = model(batch_img, train_pairs_tensor)  # [Batch, N_pairs]
            else:
                batch_feat = model.encode_image(batch_img)
                logits = model(batch_feat, train_pairs_tensor) # [Batch, N_pairs]

            # 2. Cross Entropy Loss
            ce_loss = ce_loss_fn(logits, batch_target)

            # 3. Vectorized DHNO Contrastive Loss
            # For each sample in batch, find hard negatives
            batch_size = len(batch_target)
            
            # Get mask for current batch: [Batch, N_pairs]
            current_mask_obj = mask_same_obj[batch_target]
            current_mask_attr = mask_same_attr[batch_target]
            
            # Get positive scores: [Batch]
            pos_scores = logits[torch.arange(batch_size, device=device), batch_target]
            
            # Hard Negatives (Same Obj): logits[b, mask_same_obj[batch_target[b]]]
            # Mask out non-negatives with a very small value
            obj_logits = logits.clone()
            obj_logits[~current_mask_obj] = -1e9
            hard_neg_obj, _ = obj_logits.max(dim=-1)
            
            # Hard Negatives (Same Attr): logits[b, mask_same_attr[batch_target[b]]]
            attr_logits = logits.clone()
            attr_logits[~current_mask_attr] = -1e9
            hard_neg_attr, _ = attr_logits.max(dim=-1)
            
            # Contrastive Loss: Triplet-like margin loss
            loss_obj = torch.clamp(dhno_margin + hard_neg_obj - pos_scores, min=0.0)
            loss_attr = torch.clamp(dhno_margin + hard_neg_attr - pos_scores, min=0.0)
            
            # Filter out cases where no negatives were found (max was -1e9)
            # (In MIT-States every pair has at least one same-obj and same-attr negative)
            valid_obj = (hard_neg_obj > -1e8)
            valid_attr = (hard_neg_attr > -1e8)
            
            contrastive_loss = (loss_obj[valid_obj].sum() + loss_attr[valid_attr].sum()) / batch_size

            
            # 4. Entropy Regularization (prevent over-confidence)
            entropy_loss = 0.0
            if entropy_lambda > 0:
                logits_f32 = logits.float()
                log_probs = F.log_softmax(logits_f32, dim=-1)
                entropy_loss = -torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=-1))
            
            # Total Loss
            loss = ce_loss + dhno_lambda * contrastive_loss + entropy_lambda * entropy_loss

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or \
                    (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                
                # Clamp gating param if needed? Sigmoid handles it.

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix(
                {"loss": np.mean(epoch_train_losses[-50:]), "ce": ce_loss.item(), "cont": contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0, "ent": entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else 0}
            )

            progress_bar.update()

        progress_bar.close()
        progress_bar.write(
            f"epoch {i +1} train loss {np.mean(epoch_train_losses)}"
        )
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            if config.experiment_name in ['dpc', 'dpc_alpha']:
                save_dpc_embeddings(model, config, epoch=i + 1)
            else:
                save_soft_embeddings(model, config, epoch=i + 1)

    return model, optimizer

def save_dpc_embeddings(model, config, epoch=None):
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    with torch.no_grad():
        if epoch:
            soft_emb_path = os.path.join(
                config.save_path, f"soft_embeddings_epoch_{epoch}.pt"
            )
        else:
            soft_emb_path = os.path.join(
                config.save_path, "soft_embeddings.pt"
            )
            
        # Save both primitives, contextuals, and gating param
        state = {
            "soft_embeddings_primitive": model.soft_embeddings_primitive,
            "soft_embeddings_contextual": model.soft_embeddings_contextual,
            "gating_param": model.gating_param
        }
        
        if config.experiment_name == 'dpc_alpha':
            state["alpha_predictor"] = model.alpha_predictor.state_dict()

        torch.save(state, soft_emb_path)


def save_soft_embeddings(model, config, epoch=None):
    """Function to save soft embeddings.

    Args:
        model (nn.Module): the CSP/COOP module
        config (argparse.ArgumentParser): the config
        epoch (int, optional): epoch number for the soft embedding.
            Defaults to None.
    """
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # save the soft embedding
    with torch.no_grad():
        if epoch:
            soft_emb_path = os.path.join(
                config.save_path, f"soft_embeddings_epoch_{epoch}.pt"
            )
        else:
            soft_emb_path = os.path.join(
                config.save_path, "soft_embeddings.pt"
            )

        torch.save({"soft_embeddings": model.soft_embeddings}, soft_emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
    )
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=5e-05
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--epochs", help="number of epochs", default=20, type=int
    )
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=64, type=int
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=1024, type=int
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )
    parser.add_argument("--save_path", help="save path", type=str)
    parser.add_argument(
        "--save_every_n",
        default=1,
        type=int,
        help="saves the model every n epochs; "
        "this is useful for validation/grid search",
    )
    parser.add_argument(
        "--save_model",
        help="indicate if you want to save the model state dict()",
        action="store_true",
    )
    parser.add_argument("--seed", help="seed value", default=0, type=int)

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,
        type=int
    )
    parser.add_argument(
        "--resume_path",
        help="path to soft embeddings checkpoint to resume from",
        type=str,
        default=None
    )

    parser.add_argument(
        "--dhno_lambda",
        help="weight for DHNO contrastive loss",
        type=float,
        default=0.3
    )
    
    parser.add_argument(
        "--dhno_margin",
        help="margin for DHNO contrastive loss",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--temp",
        help="temperature for logit scaling",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--entropy_lambda",
        help="weight for entropy regularization",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--csp_init_path",
        help="path to CSP soft embeddings for initialization",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--freeze_primitive",
        help="freeze the primitive soft embeddings",
        action="store_true"
    )

    config = parser.parse_args()

    # set the seed value
    set_seed(config.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("training details")
    pprint.pprint(config)

    if os.path.exists(config.save_path) and config.resume_path is None:
        print('file already exists')
        print('exiting!')
        exit(0)

    # This should work for mit-states, ut-zappos, and maybe c-gqa.
    dataset_path = DATASET_PATHS[config.dataset]
    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural')

    model, optimizer = get_model(train_dataset, config, device)

    start_epoch = 0
    if config.resume_path is not None:
        if os.path.exists(config.resume_path):
            print(f"Resuming from checkpoint: {config.resume_path}")
            checkpoint = torch.load(config.resume_path, map_location=device)
            
            if config.experiment_name in ['dpc', 'dpc_alpha']:
                model.soft_embeddings_primitive.data = checkpoint['soft_embeddings_primitive'].data
                model.soft_embeddings_contextual.data = checkpoint['soft_embeddings_contextual'].data
                model.gating_param.data = checkpoint['gating_param'].data
                if config.experiment_name == 'dpc_alpha' and 'alpha_predictor' in checkpoint:
                    model.alpha_predictor.load_state_dict(checkpoint['alpha_predictor'])
            else:
                model.set_soft_embeddings(checkpoint['soft_embeddings'])
            
            # Try to infer start epoch from filename
            try:
                # Expecting format: soft_embeddings_epoch_{epoch}.pt
                filename = os.path.basename(config.resume_path)
                epoch_str = filename.split('_epoch_')[1].split('.pt')[0]
                start_epoch = int(epoch_str)
                print(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                print(f"Could not infer epoch from filename: {e}. Starting from epoch 0 but with loaded weights.")
        else:
            print(f"Resume path {config.resume_path} does not exist!")
            exit(1)

    print("model dtype", model.dtype)
    print("soft embedding dtype", model.soft_embeddings.dtype)

    if not config.evaluate_only:
        if config.experiment_name in ['dpc', 'dpc_alpha']:
            model, optimizer = train_dpc(
                model,
                optimizer,
                train_dataset,
                config,
                device,
                start_epoch=start_epoch
            )
        else:
            model, optimizer = train_model(
                model,
                optimizer,
                train_dataset,
                config,
                device,
                start_epoch=start_epoch
            )

    if config.experiment_name in ['dpc', 'dpc_alpha']:
        save_dpc_embeddings(model, config, epoch=config.epochs)
    else:
        save_soft_embeddings(
            model,
            config,
        )

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)

    if config.save_model:
        torch.save(
            model.dict(),
            os.path.join(
                config.save_path,
                'final_model.pt'))

    print("done!")
