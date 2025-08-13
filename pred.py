# pred.py

import os
import cv2
import json
from PIL import Image
import numpy as np
from tqdm import tqdm


import torch
from networks.transunet import TransUnetTM
from datasets.dataset import ResizePadTM


organ_to_position_map = {
    'Breast': 'breast',
    'Cardiac': 'cardiac',
    'Thyroid': 'thyroid',
    'Fetal Head': 'head',
    'Kidney': 'kidney',
    'Appendix': 'appendix',
    'Liver': 'liver',
}


position_prompt_one_hot_dict = {
    "breast":   [1, 0, 0, 0, 0, 0, 0, 0],
    "cardiac":  [0, 1, 0, 0, 0, 0, 0, 0],
    "thyroid":  [0, 0, 1, 0, 0, 0, 0, 0],
    "head":     [0, 0, 0, 1, 0, 0, 0, 0],
    "kidney":   [0, 0, 0, 0, 1, 0, 0, 0],
    "appendix": [0, 0, 0, 0, 0, 1, 0, 0],
    "liver":    [0, 0, 0, 0, 0, 0, 1, 0],
    "indis":    [0, 0, 0, 0, 0, 0, 0, 1]
}


task_prompt_one_hot_dict = {
    "segmentation": [1, 0],
    "classification": [0, 1]
}

organ_to_nature_map = {
    'Breast': 'tumor',
    'Cardiac': 'organ',
    'Thyroid': 'tumor',
    'Fetal Head': 'organ',
    'Kidney': 'organ',
    'Appendix': 'organ',
    'Liver': 'organ',
}

nature_prompt_one_hot_dict = {
    "tumor": [1, 0],
    "organ": [0, 1],
}

type_prompt_one_hot_dict = {
    "whole": [1, 0, 0],
    "local": [0, 1, 0],
    "location": [0, 0, 1],
}


def _compute_resize_pad_params(orig_h, orig_w, out_h, out_w):
    scale = min(out_h / float(orig_h), out_w / float(orig_w))
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    off_y = (out_h - new_h) // 2
    off_x = (out_w - new_w) // 2
    return scale, new_h, new_w, off_y, off_x


class Model:
    def __init__(self):
        print("Initializing model...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class Args:
            img_size = 224
            prompt = True
            film_scale = float(os.environ.get('FILM_SCALE', '0.7'))
            prior_lambda = float(os.environ.get('PRIOR_LAMBDA', '0.5'))
            cls_head_variant = os.environ.get('CLS_HEAD_VARIANT', 'linear')
            cls_dropout = float(os.environ.get('CLS_DROPOUT', '0.3'))


        args = Args()
        self.args = args

        # Build TransUnetTM matching training-time architecture (grayscale input, 2-class seg)
        self.network = TransUnetTM(
            img_size=args.img_size,
            in_chans=1,
            seg_out_ch=2,
            lora_rank=int(os.environ.get('LORA_RANK', '8')),
            lora_alpha=float(os.environ.get('LORA_ALPHA', '16.0')),
            lora_dropout=float(os.environ.get('LORA_DROPOUT', '0.0')),
            lora_only=bool(int(os.environ.get('LORA_ONLY', '0'))) if os.environ.get('LORA_ONLY') is not None else False,
            max_lora_scale=float(os.environ.get('MAX_LORA_SCALE', '0.5')),
            scale_mode=str(os.environ.get('SCALE_MODE', 'sigmoid')),
            film_scale=args.film_scale,
            prior_lambda=args.prior_lambda,
            cls_head_variant=args.cls_head_variant,
            cls_dropout=args.cls_dropout,
        ).to(self.device)

        # Load checkpoint: prefer CKPT env, else common default under exp_out/ or exp_tm/
        ckpt = os.environ.get('CKPT')
        if ckpt is None or not os.path.isfile(ckpt):
            # try latest tm run path
            fallback_paths = [
                'exp_tm/latest/best_model.pth',
                'exp_out/trial_1/best_model.pth',
            ]
            for p in fallback_paths:
                if os.path.isfile(p):
                    ckpt = p
                    break
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError("Checkpoint not found. Set CKPT env or place best_model.pth under exp_tm/latest or exp_out/trial_1.")
        pretrained = torch.load(ckpt, map_location=self.device)
        # Choose the correct state dict container
        if isinstance(pretrained, dict) and 'state_dict' in pretrained:
            state_dict = pretrained['state_dict']
        elif isinstance(pretrained, dict) and 'model' in pretrained:
            state_dict = pretrained['model']
        else:
            state_dict = pretrained

        # Strip any DistributedDataParallel 'module.' prefixes
        cleaned = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith('module.') else k
            cleaned[nk] = v

        # Load non-strict to be resilient to minor head/config diffs
        missing_unexpected = self.network.load_state_dict(cleaned, strict=False)
        if getattr(missing_unexpected, 'missing_keys', None) or getattr(missing_unexpected, 'unexpected_keys', None):
            print(f"[Info] Loaded with non-strict. Missing: {len(getattr(missing_unexpected,'missing_keys',[]))}, Unexpected: {len(getattr(missing_unexpected,'unexpected_keys',[]))}")

        self.network.eval()
        
        self.transform = ResizePadTM(output_size=[args.img_size, args.img_size])

        print("Model initialized.")

    def predict_segmentation_and_classification(self, data_list, input_dir, output_dir):
        class_predictions = {}

        for data_dict in tqdm(data_list, desc="Processing images"):
            img_path = os.path.join(input_dir, data_dict['img_path_relative'])
            task = data_dict['task']
            dataset_name = data_dict['dataset_name']
            organ_name = data_dict['organ']

            img = Image.open(img_path).convert('RGB')
            original_size = img.size # (width, height)
            img_np = np.array(img)
            
            sample = {'image': img_np / 255.0, 'label': np.zeros(img_np.shape[:2])}
            processed_sample = self.transform(sample)
            # processed_sample['image']: (1,H,W) float
            image_tensor = processed_sample['image'].unsqueeze(0).to(self.device) # [B=1,1,H,W]

            with torch.no_grad():
                if self.args.prompt:
                    task_p_vec = task_prompt_one_hot_dict[task]
                    task_prompt = torch.tensor(task_p_vec, dtype=torch.float).unsqueeze(0).to(self.device)

                    position_key = organ_to_position_map.get(organ_name, 'indis')
                    position_p_vec = position_prompt_one_hot_dict[position_key]
                    position_prompt = torch.tensor(position_p_vec, dtype=torch.float).unsqueeze(0).to(self.device)

                    nature_key = organ_to_nature_map.get(organ_name, 'organ')
                    nature_p_vec = nature_prompt_one_hot_dict[nature_key]
                    nature_prompt = torch.tensor(nature_p_vec, dtype=torch.float).unsqueeze(0).to(self.device)
                    
                    type_p_vec = type_prompt_one_hot_dict["whole"]
                    type_prompt = torch.tensor(type_p_vec, dtype=torch.float).unsqueeze(0).to(self.device)

                    model_input = (image_tensor, position_prompt, task_prompt, type_prompt, nature_prompt)
                    outputs_tuple = self.network(model_input)
                else:
                    outputs_tuple = self.network(image_tensor)

            if task == 'classification':
                if dataset_name == 'Breast_luminal':
                    num_classes = 4
                    logits = outputs_tuple[2]
                else:
                    num_classes = 2
                    logits = outputs_tuple[1]

                probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                prediction = int(np.argmax(probabilities))
                
                class_predictions[data_dict['img_path_relative']] = {
                    'probability': probabilities.tolist(),
                    'prediction': prediction
                }

            elif task == 'segmentation':
                seg_logits = outputs_tuple[0]  # [B,2,224,224]
                seg_pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1).squeeze(0)  # [224,224]
                pred_mask_224 = seg_pred.cpu().numpy().astype(np.uint8)  # {0,1}

                # Inverse ResizePad: crop unpadded region then resize back to original size
                orig_w, orig_h = original_size
                out_h = out_w = self.args.img_size
                _, new_h, new_w, off_y, off_x = _compute_resize_pad_params(orig_h, orig_w, out_h, out_w)
                content = pred_mask_224[off_y:off_y+new_h, off_x:off_x+new_w]
                restored = cv2.resize(content, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                restored = (restored > 0).astype(np.uint8) * 255
                mask_img = Image.fromarray(restored)

                save_path = os.path.join(output_dir, data_dict['img_path_relative'].replace('img', 'mask'))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                mask_img.save(save_path)

        with open(os.path.join(output_dir, 'classification.json'), 'w') as f:
            json.dump(class_predictions, f, indent=4)


if __name__ == '__main__':
    input_dir = 'data/Val/'
    data_list_path = 'data/Val/private_val_for_participants.json'

    output_dir = os.environ['OUTPUT_DIR']
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(data_list_path, 'r') as f:
        data_list = json.load(f)

    model = Model()
    model.predict_segmentation_and_classification(data_list, input_dir, output_dir)
    print("Inference completed.")