import argparse
import copy
import os
import os.path as osp
import json
import tempfile
import datetime
import shutil

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from colorama import Fore
from mixofshow.data.lora_dataset import LoraDataset
from mixofshow.data.prompt_dataset import PromptDataset
from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline, StableDiffusionPipeline
from mixofshow.pipelines.trainer_edlora import EDLoRATrainer
from mixofshow.utils.convert_edlora_to_diffusers import convert_edlora
from mixofshow.utils.util import MessageLogger, dict2str, reduce_loss_dict, set_path_logger
from test_edlora import visual_validation
from gradient_fusion import save_joint_trained_concepts

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version('0.18.2')


class JointEDLoRATrainer:
    def __init__(self, joint_cfg):
        """Initialize with single joint config instead of multiple configs"""
        self.trainers = []
        self.optimizers = []
        self.train_dataloaders = []
        self.schedulers = []  # Keep only if you need schedulers
        self.joint_dataloader = None  # Initialize joint_dataloader attribute
        self.logger = get_logger('mixofshow', log_level='INFO')  # Initialize logger
        self.joint_cfg = joint_cfg  # Store config
        
        # Get number of concepts from config
        self.num_concepts = len([k for k in joint_cfg['datasets'].keys() if k.startswith('concept_')])
        
        if self.num_concepts == 0:
            raise ValueError("No concept datasets found in configuration!")
        
        # Initialize trainers for each concept
        for i in range(1, self.num_concepts + 1):
            concept_key = f'concept_{i}'
            trainset_cfg = joint_cfg['datasets'][concept_key]
            
            # Load concept list from JSON file
            if 'concept_list' in trainset_cfg:
                with open(trainset_cfg['concept_list'], 'r') as f:
                    concept_data = json.load(f)
                    if isinstance(concept_data, list) and len(concept_data) > 0:
                        trainset_cfg.update(concept_data[0])
            
            # Create dataset
            train_dataset = LoraDataset(trainset_cfg)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=trainset_cfg['batch_size_per_gpu'],
                shuffle=True,
                drop_last=True
            )
            
            # Add to lists
            self.train_dataloaders.append(train_dataloader)
            
            # Create trainer config by copying the models config
            trainer_cfg = copy.deepcopy(joint_cfg['models'])
            
            # Remove all concept-specific configurations
            for j in range(1, self.num_concepts + 1):
                trainer_cfg.pop(f'concept_{j}', None)
            
            # Add only the current concept's settings
            trainer_cfg['new_concept_token'] = joint_cfg['models'][concept_key]['new_concept_token']
            trainer_cfg['initializer_token'] = joint_cfg['models'][concept_key]['initializer_token']
            
            # Add replace_mapping from dataset config if it exists
            if 'replace_mapping' in trainset_cfg:
                trainer_cfg['replace_mapping'] = trainset_cfg['replace_mapping']
            
            trainer = EDLoRATrainer(**trainer_cfg)
            
            # Initialize concept tokens
            trainer.new_concept_cfg = trainer.init_new_concept(
                trainer_cfg['new_concept_token'],
                trainer_cfg['initializer_token'],
                enable_edlora=trainer_cfg['enable_edlora']
            )
            
            self.trainers.append(trainer)
            
            # Set optimizer
            train_opt = copy.deepcopy(joint_cfg['train'])
            optim_type = train_opt['optim_g'].pop('type')
            assert optim_type == 'AdamW', 'only support AdamW now'
            optimizer = torch.optim.AdamW(trainer.get_params_to_optimize(), **train_opt['optim_g'])
            self.optimizers.append(optimizer)
            
            # Set scheduler
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=joint_cfg.get('total_iter', 1000)
            )
            self.schedulers.append(scheduler)
        
        # Load joint dataset if configured
        if 'joint' in joint_cfg['datasets']:
            joint_dataset_cfg = joint_cfg['datasets']['joint']
            print(Fore.GREEN + "Loading joint dataset..." + Fore.RESET)
            
            # Load concept list from JSON file
            if 'concept_list' in joint_dataset_cfg:
                with open(joint_dataset_cfg['concept_list'], 'r') as f:
                    concept_data = json.load(f)
                    if isinstance(concept_data, list) and len(concept_data) > 0:
                        joint_dataset_cfg.update(concept_data[0])
            
            # Create joint dataset
            joint_dataset = LoraDataset(joint_dataset_cfg)
            self.joint_dataloader = torch.utils.data.DataLoader(
                joint_dataset,
                batch_size=joint_dataset_cfg['batch_size_per_gpu'],
                shuffle=True,
                drop_last=True
            )
            print(Fore.GREEN + f"Joint dataset loaded with {len(joint_dataset)} samples" + Fore.RESET)
    
    def prepare(self, accelerator):
        """Prepare trainers, optimizers, and dataloaders with accelerator"""
        for i in range(self.num_concepts):
            # Only prepare trainer, optimizer and train dataloader (skip validation)
            self.trainers[i], self.optimizers[i], self.train_dataloaders[i] = accelerator.prepare(
                self.trainers[i], 
                self.optimizers[i], 
                self.train_dataloaders[i]
            )
            
            # Scheduler
            self.schedulers.append(
                get_scheduler(
                    'linear',
                    optimizer=self.optimizers[i],
                    num_warmup_steps=0,
                    num_training_steps=len(self.train_dataloaders[i]) * accelerator.num_processes
                )
            )
        
        # Prepare joint dataloader if it exists
        if self.joint_dataloader is not None:
            self.joint_dataloader = accelerator.prepare(self.joint_dataloader)
    
    def process_joint_batch(self, batch):
        """Special processing for joint training batches with independent token replacement"""
        processed_batch = batch.copy()
        
        # Process masks if they exist
        if 'masks' in batch and 'img_masks' in batch:
            image_filenames = [os.path.basename(path) for path in batch['image_paths']]
            processed_masks = {}
            
            for token in ['<TOK1>', '<TOK2>']:
                token_masks = []
                for img_filename in image_filenames:
                    # Get the mask mapping for this image
                    if img_filename in batch['mask_mapping']:
                        mask_path = os.path.join(
                            batch['mask_dir'],
                            batch['mask_mapping'][img_filename][token]
                        )
                        if os.path.exists(mask_path):
                            mask = batch['masks'][token][batch['image_paths'].index(img_filename)]
                            token_masks.append(mask)
                        else:
                            raise FileNotFoundError(f"Mask file not found: {mask_path}")
                    else:
                        raise KeyError(f"No mask mapping found for image: {img_filename}")
                
                processed_masks[token] = torch.stack(token_masks) if token_masks else None
            
            processed_batch['masks'] = processed_masks
        
        # Get the replacement mapping from the dataset config
        if hasattr(batch, 'replace_mapping'):
            replace_mapping = batch['replace_mapping']
        else:
            # Fallback to the stored concept tokens if no mapping in batch
            replace_mapping = {}
            for concept_id, tokens in self.concept_tokens.items():
                placeholder = f"<{concept_id.upper()}_TOK>"
                replacement = ' '.join(tokens)
                replace_mapping[placeholder] = replacement
        
        # Replace tokens in prompts using the mapping
        new_prompts = []
        for prompt in batch['prompts']:
            new_prompt = prompt
            for token, replacement in replace_mapping.items():
                new_prompt = new_prompt.replace(token, replacement)
            new_prompts.append(new_prompt)
        processed_batch['prompts'] = new_prompts
        
        return processed_batch

    def train_step(self, accelerator):
        """Train step that handles single concept and joint concept data differently"""
        total_loss = 0
        
        # Get joint batch if available
        if self.joint_dataloader is not None:
            try:
                joint_batch = next(iter(self.joint_dataloader))
                # Process joint batch - update all concepts together
                for i, trainer in enumerate(self.trainers):
                    if 'masks' in joint_batch:
                        mask_dict = joint_batch['masks']
                        token = f"<TOK{i+1}>"
                        mask = mask_dict.get(token, None)
                    else:
                        mask = joint_batch.get('img_masks', None)
                    
                    joint_loss = trainer(
                        joint_batch['images'],
                        joint_batch['prompts'],
                        mask,
                        joint_batch.get('img_masks', None)
                    )
                    if torch.isnan(joint_loss):
                        joint_loss = torch.tensor(0.0, device=accelerator.device)
                    
                    # Scale loss and update
                    scaled_loss = joint_loss / len(self.trainers)
                    accelerator.backward(scaled_loss)
                    total_loss += joint_loss.item()
                    
                    # Update trainer
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
                    self.optimizers[i].step()
                    self.schedulers[i].step()
                    self.optimizers[i].zero_grad()
            except StopIteration:
                joint_batch = None
        
        # Process single concept batches - update each concept separately
        for i, (trainer, dataloader) in enumerate(zip(self.trainers, self.train_dataloaders)):
            try:
                batch = next(iter(dataloader))
                # Get masks
                if 'masks' in batch:
                    masks = batch['masks']
                else:
                    masks = batch.get('img_masks', None)
                
                # Compute individual loss
                individual_loss = trainer(batch['images'], batch['prompts'], masks, batch.get('img_masks', None))
                if torch.isnan(individual_loss):
                    individual_loss = torch.tensor(0.0, device=accelerator.device)
                
                # Update for this concept only
                accelerator.backward(individual_loss)
                total_loss += individual_loss.item()
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
                self.optimizers[i].step()
                self.schedulers[i].step()
                self.optimizers[i].zero_grad()
            except StopIteration:
                continue

        return total_loss, False

    def save_checkpoint(self, accelerator, global_step, trainer_idx):
        """Save checkpoint for a specific trainer in a format compatible with compose_concepts"""
        enable_edlora = self.joint_cfg['models']['enable_edlora']
        lora_type = 'edlora' if enable_edlora else 'lora'
        
        # Save to fixed path
        save_dir = self.joint_cfg['path']['models']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{lora_type}_model-{global_step}.pth')
        latest_path = os.path.join(save_dir, f'{lora_type}_model-latest.pth')

        if accelerator.is_main_process:
            trainer = accelerator.unwrap_model(self.trainers[trainer_idx])
            
            # Get the correct concept names from the configuration
            concept_key = f'concept_{trainer_idx + 1}'
            concept_tokens = self.joint_cfg['models'][concept_key]['new_concept_token'].split('+')
            
            # Create a concept config that matches save_joint_trained_concepts format
            concept_list = [{
                "lora_path": save_path,
                "concept_name": token
            } for token in concept_tokens]
            
            # Save the trainer state first
            accelerator.save({'params': trainer.delta_state_dict()}, save_path)
            self.logger.info(f'Save state to {save_path}')
            
            # Also save as latest
            shutil.copy2(save_path, latest_path)
            self.logger.info(f'Save latest state to {latest_path}')

            # Call save_joint_trained_concepts with the concept list
            save_joint_trained_concepts(
                concept_list=concept_list,
                pretrained_model_path=self.joint_cfg['models']['pretrained_path'],
                save_path=os.path.join(save_dir, f'checkpoint-{global_step}'),
                suffix=f'concept_{trainer_idx + 1}',
                device=accelerator.device
            )

        accelerator.wait_for_everyone()

        # Run validation if configured
        if self.joint_cfg['val'].get('val_during_save', False):
            self.logger.info(f'Start validation {save_path}:')
            for lora_alpha in self.joint_cfg['val']['alpha_list']:
                pipeclass = EDLoRAPipeline if enable_edlora else StableDiffusionPipeline
                pipe = pipeclass.from_pretrained(
                    self.joint_cfg['models']['pretrained_path'],
                    scheduler=DPMSolverMultistepScheduler.from_pretrained(
                        self.joint_cfg['models']['pretrained_path'], 
                        subfolder='scheduler'
                    ),
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to('cuda')
                
                pipe, new_concept_cfg = convert_edlora(
                    pipe, 
                    torch.load(save_path), 
                    enable_edlora=enable_edlora, 
                    alpha=lora_alpha
                )
                pipe.set_new_concept_cfg(new_concept_cfg)
                pipe.set_progress_bar_config(disable=True)
                
                # Create a validation dataloader if needed
                if 'val_vis' in self.joint_cfg['datasets']:
                    val_dataset = PromptDataset(self.joint_cfg['datasets']['val_vis'])
                    val_dataloader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=self.joint_cfg['datasets']['val_vis']['batch_size_per_gpu'],
                        shuffle=False
                    )
                    val_dataloader = accelerator.prepare(val_dataloader)
                    visual_validation(
                        accelerator, 
                        pipe, 
                        val_dataloader, 
                        f'Iters-{global_step}_Alpha-{lora_alpha}', 
                        self.joint_cfg
                    )
                
                del pipe
                torch.cuda.empty_cache()

    def save_joint_checkpoint(self, accelerator, global_step):
        """
        Save checkpoint for all trainers in a joint manner.
        
        Args:
            accelerator: Accelerator instance
            global_step: Current training step
        """
        enable_edlora = self.joint_cfg['models']['enable_edlora']
        lora_type = 'edlora' if enable_edlora else 'lora'
        
        # Save to fixed path
        save_dir = self.joint_cfg['path']['models']
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{lora_type}_model-{global_step}.pth')
        latest_path = os.path.join(save_dir, f'{lora_type}_model-latest.pth')

        if accelerator.is_main_process:
            # Collect concepts from all trainers
            concept_list = []
            for i in range(len(self.trainers)):
                trainer = accelerator.unwrap_model(self.trainers[i])
                
                # Get the correct concept names from the configuration
                concept_key = f'concept_{i + 1}'
                concept_tokens = self.joint_cfg['models'][concept_key]['new_concept_token'].split('+')
                
                # Save the trainer state
                trainer_save_path = os.path.join(save_dir, f'{lora_type}_model-{global_step}_trainer{i+1}.pth')
                accelerator.save({'params': trainer.delta_state_dict()}, trainer_save_path)
                
                # Add concepts to the list
                for token in concept_tokens:
                    concept_list.append({
                        "lora_path": trainer_save_path,
                        "concept_name": token
                    })
            
            # Save the latest checkpoint
            shutil.copy2(concept_list[0]['lora_path'], latest_path)
            
            # Call save_joint_trained_concepts with combined concept list
            save_joint_trained_concepts(
                concept_list=concept_list,
                pretrained_model_path=self.joint_cfg['models']['pretrained_path'],
                save_path=os.path.join(save_dir, f'checkpoint-{global_step}'),
                suffix='joint_concepts',
                device=accelerator.device
            )

        accelerator.wait_for_everyone()


def train(root_path, joint_cfg, opt_path):
    # Set accelerator
    accelerator = Accelerator(
        mixed_precision=joint_cfg['mixed_precision'],
        gradient_accumulation_steps=joint_cfg['gradient_accumulation_steps']
    )
    
    # Set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, root_path, joint_cfg['name'], joint_cfg, opt_path, is_train=True)
    
    # Get logger
    logger = get_logger('mixofshow', log_level='INFO')
    logger.info(accelerator.state, main_process_only=True)
    logger.info(f"Training {len(joint_cfg['datasets'])} concepts jointly")
    
    # Set seed
    if joint_cfg.get('manual_seed') is not None:
        set_seed(joint_cfg['manual_seed'])
    
    # Initialize joint trainer
    joint_trainer = JointEDLoRATrainer(joint_cfg)
    joint_trainer.prepare(accelerator)
    
    global_step = 0
    msg_logger = MessageLogger(joint_cfg, global_step)
    
    # Calculate total steps from train config
    total_steps = joint_cfg['train'].get('total_iter', 1000)  # Default to 1000 if not specified
    
    # Initialize progress bar
    if accelerator.is_main_process:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_steps, desc="Training", position=0)
    
    while True:
        loss, _ = joint_trainer.train_step(accelerator)
        
        # Check for stop signal from train_step
        if loss == -1:
            logger.info("Training completed - learning rate reached zero")
            break
        
        if accelerator.sync_gradients:
            global_step += 1
            
            # Collect learning rates first
            current_lrs = []
            for optimizer in joint_trainer.optimizers:
                for param_group in optimizer.param_groups:
                    current_lrs.append(param_group['lr'])
            
            # Update progress bar
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}' if not torch.isnan(torch.tensor(loss)) else 'NaN',
                    'step': global_step,
                    'lr': f'{current_lrs[0]:.6f}'
                })
            
            # Log message
            if global_step % joint_cfg['logger']['print_freq'] == 0:
                log_dict = {
                    'iter': global_step,
                    'avg_loss': loss,
                    'lrs': current_lrs
                }
                msg_logger(log_dict)
            
            # Save models
            if global_step % joint_cfg['logger']['save_checkpoint_freq'] == 0:
                joint_trainer.save_joint_checkpoint(accelerator, global_step)
            
            # Check if we've reached total steps
            if global_step >= total_steps:
                logger.info("Training completed - reached total steps")
                break
    
    # Close progress bar
    if accelerator.is_main_process:
        progress_bar.close()
    
    # Save final models
    joint_trainer.save_joint_checkpoint(accelerator, global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True,
                       help='Path to joint configuration file')
    parser.add_argument('--total_steps', type=int, default=None,
                       help='Override total training steps from config')
    parser.add_argument('--use_joint_dataset', action='store_true',
                       help='Whether to use the joint dataset for training. If not set, only single concept datasets will be used.')
    args = parser.parse_args()
    
    # Load config
    joint_cfg = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)
    
    # Override total steps if provided
    if args.total_steps is not None:
        joint_cfg['train']['total_iter'] = args.total_steps
        # Update paths to include step count
        base_path = f"experiments/EDLoRA_cat2_dog6_joint_{args.total_steps}"
        joint_cfg['path'] = {
            'models': f"{base_path}/models",
            'training_states': f"{base_path}/training_states",
            'visualization': f"{base_path}/visualization",
            'log': base_path
        }

    # Control joint dataset usage based on argument
    if not args.use_joint_dataset and 'joint' in joint_cfg['datasets']:
        del joint_cfg['datasets']['joint']
        print(Fore.YELLOW + "Joint dataset disabled by command line argument" + Fore.RESET)

    print(joint_cfg['path'])
    
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train(root_path, joint_cfg, args.opt) 