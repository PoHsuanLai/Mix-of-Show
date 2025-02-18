import os

import groundingdino.datasets.transforms as T
import numpy as np
import torch
# from GroundingDINO.groundingdino.models import build_model
# from GroundingDINO.groundingdino.util import box_ops
# from GroundingDINO.groundingdino.util.inference import predict
# from GroundingDINO.groundingdino.util.slconfig import SLConfig
# from GroundingDINO.groundingdino.util.utils import clean_state_dict
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


class LangSAM():

    def __init__(self, sam_type="vit_h", ckpt_path=None):
        self.sam_type = sam_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set default cache path
        self.cache_path = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))
        os.makedirs(self.cache_path, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Cache path: {self.cache_path}")
        
        self.build_groundingdino()
        self.build_sam(ckpt_path)

    def build_sam(self, ckpt_path):
        """Build SAM model - first try local file, then try downloading."""
        if ckpt_path is None:
            # Try to find model in cache
            if self.sam_type is None:
                print("No sam type indicated. Using vit_h by default.")
                self.sam_type = "vit_h"
            
            checkpoint_url = SAM_MODELS[self.sam_type]
            model_name = os.path.basename(checkpoint_url)
            cache_path = os.path.join(CACHE_PATH, model_name)
            
            # Check if model exists in cache
            if os.path.exists(cache_path):
                print(f"Loading SAM model from cache: {cache_path}")
                state_dict = torch.load(cache_path, map_location='cpu')
            else:
                print(f"Downloading SAM model from {checkpoint_url}")
                try:
                    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
                except Exception as e:
                    raise ValueError(f"Failed to download or load SAM model. Error: {str(e)}")
            
            try:
                sam = sam_model_registry[self.sam_type]()
                sam.load_state_dict(state_dict, strict=True)
            except Exception as e:
                raise ValueError(f"Failed to initialize SAM model. Error: {str(e)}")
        else:
            if not os.path.exists(ckpt_path):
                raise ValueError(f"Checkpoint path does not exist: {ckpt_path}")
            try:
                sam = sam_model_registry[self.sam_type](ckpt_path)
            except Exception as e:
                raise ValueError(f"Failed to load SAM from checkpoint. Error: {str(e)}")
        
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases
    
    def predict_dino_class(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_trans = transform_image(image_pil)
        
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases
    def predict_sam(self, image_pil, boxes):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()
    def predict_sam_mask(self, image_pil, mask_pil):
        W, H = image_pil.size
        image_array = np.asarray(image_pil)
        mask_array=np.asarray(mask_pil)
        point = torch.tensor([H//2,W//2]).unsqueeze(0).unsqueeze(0)
        lab = torch.tensor([1]).unsqueeze(0)
        # print(lab.shape)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        # point = 
        masks = torch.tensor([])
        # torch.tensor(mask_array).to(self.sam.device)
        masks, _, _ = self.sam.predict_torch(
            point_coords=point.to(self.sam.device),
            point_labels=lab.to(self.sam.device),
            boxes=None,
            multimask_output=False,
            mask_input=torch.tensor(mask_array).to(self.sam.device)
        )
        masks = masks.squeeze(1)
        return masks.cpu()
    def predict(self, image_pil, text_prompt, box_threshold=0.20, text_threshold=0.20):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits
