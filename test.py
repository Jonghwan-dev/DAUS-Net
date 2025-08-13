import argparse
import logging
import os
import random
import sys
import time
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from configs.config import get_config

from datasets.dataset import ResizePadTM
from datasets.dataset import USdatasetCls, USdatasetSeg

from util.utils import omni_seg_test
from sklearn.metrics import accuracy_score

from networks.transunet import TransUnetTM


def _compute_resize_pad_params(orig_h, orig_w, out_h, out_w):
    scale = min(out_h / orig_h, out_w / orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    off_y = (out_h - new_h) // 2
    off_x = (out_w - new_w) // 2
    return scale, new_h, new_w, off_y, off_x


def restore_mask_from_padded(pred_mask_hw, orig_h, orig_w, out_h, out_w):
    _, new_h, new_w, off_y, off_x = _compute_resize_pad_params(orig_h, orig_w, out_h, out_w)
    inner = pred_mask_hw[off_y:off_y+new_h, off_x:off_x+new_w]
    restored = cv2.resize(inner.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return restored


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/', help='root dir for data')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_saveout', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml",
                    metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--report_model_stats', action='store_true', help='Print model parameter counts and approximate model size (MB)')
parser.add_argument('--report_flops', action='store_true', help='Estimate GFLOPs with a single dummy forward (requires thop)')
parser.add_argument('--report_only', action='store_true', help='Only report stats (params/GFLOPs) and exit')
parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint .pth to evaluate or report')

parser.add_argument('--prompt', action='store_true', help='using prompt')
parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank (0 disables LoRA)')
parser.add_argument('--lora_alpha', type=float, default=16.0, help='LoRA scaling alpha')
parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout probability')
parser.add_argument('--lora_only', action='store_true', help='train only LoRA adapters (freeze base linear weights)')
parser.add_argument('--max_lora_scale', type=float, default=1.0, help='cap for prompt-controlled LoRA runtime scale')
parser.add_argument('--scale_mode', type=str, default='sigmoid', choices=['sigmoid','softplus','tanh'], help='activation for LoRA scale head')

parser.add_argument('--film_scale', type=float, default=0.7, help='film scale')
parser.add_argument('--prior_lambda', type=float, default=0.5, help='prior lambda')
parser.add_argument('--cls_head_variant', type=str, default='linear', choices=['linear','per_head_mlp'], help='classification head variant')
parser.add_argument('--cls_dropout', type=float, default=0.3, help='classification dropout probability')

args = parser.parse_args()
config = get_config(args)


def inference(args, model, test_save_path=None):

    if not os.path.exists("exp_out/result_tm.csv"):
        with open("exp_out/result_tm.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'task', 'metric', 'time'])

    seg_test_set = [
        "BUS-BRA",
        "BUSIS",
        "BUSI",
        "CAMUS",
        "DDTI",
        "Fetal_HC",
        "KidneyUS",
        "private_Thyroid",
        "private_Kidney",
        "private_Fetal_Head",
        "private_Cardiac",
        "private_Breast_luminal",
        "private_Breast",
    ]

    for dataset_name in seg_test_set:
        num_classes = 2
        db_test = USdatasetSeg(
            base_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            split="test",
            list_dir=os.path.join(args.root_path, "segmentation", dataset_name),
            transform=ResizePadTM(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )
        logging.info("Length of {} test set is: {}".format(dataset_name, len(db_test)))
        testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=16)
        logging.info("Testing %s" % dataset_name)

        metric_list = 0.0
        count_matrix = np.ones((len(db_test), num_classes-1))
        
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label = sampled_batch["image"], sampled_batch["label"]
            if args.prompt:
                position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                task_prompt = torch.tensor(np.array([[1]*position_prompt.shape[0], [0]*position_prompt.shape[0]])).permute([1, 0]).float()
                type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                metric_i = omni_seg_test(
                    image, label, model, classes=num_classes, prompt=args.prompt,
                    type_prompt=type_prompt, nature_prompt=nature_prompt,
                    position_prompt=position_prompt, task_prompt=task_prompt
                )
            else:
                metric_i = omni_seg_test(image, label, model, classes=num_classes)

            for sample_index in range(len(metric_i)):
                if not metric_i[sample_index][1]:
                    count_matrix[i_batch*args.batch_size+sample_index, 0] = 0

            metric_i = [element[0] for element in metric_i]
            metric_list += np.array(metric_i).sum()

            # Optional: save outputs restored to original size if requested
            if args.is_saveout and test_save_path is not None:
                # We need original sizes; USdatasetSeg stores case_name, enable reading images again
                for b in range(image.shape[0]):
                    case_name = sampled_batch['case_name'][b]
                    img_path = os.path.join(args.root_path, 'segmentation', dataset_name, 'imgs', case_name)
                    orig = cv2.imread(img_path)
                    orig_h, orig_w = orig.shape[:2]

                    with torch.no_grad():
                        if args.prompt:
                            out = model((image[b].unsqueeze(0).cuda(), position_prompt[b].unsqueeze(0).cuda(), task_prompt[b].unsqueeze(0).cuda(), type_prompt[b].unsqueeze(0).cuda(), nature_prompt[b].unsqueeze(0).cuda()))
                        else:
                            out = model(image[b].unsqueeze(0).cuda())
                    seg_logits = out[0]
                    seg_pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    restored = restore_mask_from_padded(seg_pred, orig_h, orig_w, args.img_size, args.img_size)

                    save_dir = os.path.join(test_save_path, dataset_name, 'masks')
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(save_dir, case_name), (restored>0).astype(np.uint8)*255)

        metric_list = metric_list / (count_matrix.sum(axis=0) + 1e-6)
        performance = np.mean(metric_list, axis=0)
        logging.info('Testing performance (TM) in best val model: DSC : %f' % (performance))

        with open("exp_out/result_tm.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if args.prompt:
                writer.writerow([dataset_name, 'omni_seg_prompt_tm@'+args.output_dir, performance,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])
            else:
                writer.writerow([dataset_name, 'omni_seg_tm@'+args.output_dir, performance,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])

    cls_test_set = [
        "Appendix",
        "BUS-BRA",
        "BUSI",
        "Fatty-Liver",
        "private_Liver",
        "private_Breast_luminal",
        "private_Breast",
        "private_Appendix",
    ]

    for dataset_name in cls_test_set:
        if dataset_name == "private_Breast_luminal":
            num_classes = 4
        else:
            num_classes = 2
        db_test = USdatasetCls(
            base_dir=os.path.join(args.root_path, "classification", dataset_name),
            split="test",
            list_dir=os.path.join(args.root_path, "classification", dataset_name),
            transform=ResizePadTM(output_size=[args.img_size, args.img_size]),
            prompt=args.prompt
        )
        logging.info("Length of {} test set is: {}".format(dataset_name, len(db_test)))
        testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=16)
        logging.info("Testing %s" % dataset_name)

        label_list = []
        prediction_list = []
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label = sampled_batch["image"], sampled_batch["label"]
            case_name = sampled_batch['case_name'][0]
            if args.prompt:
                position_prompt = torch.tensor(np.array(sampled_batch['position_prompt'])).permute([1, 0]).float()
                task_prompt = torch.tensor(np.array([[0]*position_prompt.shape[0], [1]*position_prompt.shape[0]])).permute([1, 0]).float()
                type_prompt = torch.tensor(np.array(sampled_batch['type_prompt'])).permute([1, 0]).float()
                nature_prompt = torch.tensor(np.array(sampled_batch['nature_prompt'])).permute([1, 0]).float()
                with torch.no_grad():
                    output = model((image.cuda(), position_prompt.cuda(), task_prompt.cuda(),
                                   type_prompt.cuda(), nature_prompt.cuda()))
            else:
                with torch.no_grad():
                    output = model(image.cuda())

            if num_classes == 4:
                logits = output[2]
            else:
                logits = output[1]

            prediction = np.argmax(torch.softmax(logits, dim=1).data.cpu().numpy())
            logging.info('idx %d case %s label: %d predict: %d' % (i_batch, case_name, label, prediction))

            label_list.append(label.numpy())
            prediction_list.append(prediction)

        label_list = np.array(label_list)
        prediction_list = np.array(prediction_list)
        for i in range(num_classes):
            logging.info('class %d acc %f' % (i, accuracy_score(
                (label_list == i).astype(int), (prediction_list == i).astype(int))))
        performance = accuracy_score(label_list, prediction_list)
        logging.info('Testing performance (TM) in best val model: acc : %f' % (performance))

        with open("exp_out/result_tm.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if args.prompt:
                writer.writerow([dataset_name, 'omni_cls_prompt_tm@'+args.output_dir, performance,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])
            else:
                writer.writerow([dataset_name, 'omni_cls_tm@'+args.output_dir, performance,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Strengthen determinism across libraries
    try:
        os.environ.setdefault('PYTHONHASHSEED', str(args.seed))
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(args.deterministic == 1)
    except Exception:
        pass

    net = TransUnetTM(
        img_size=args.img_size,
        in_chans=1,
        seg_out_ch=2,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_only=args.lora_only,
        max_lora_scale=args.max_lora_scale,
        scale_mode=args.scale_mode,
    ).cuda()

    # ---- Model stats (no dummy needed) ----
    if args.report_model_stats or args.report_flops:
        try:
            total_params = sum(p.numel() for p in net.parameters())
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            size_mb = sum(p.numel() * p.element_size() for p in net.parameters()) / 1024 / 1024
            print(f"[Model] Params: total={total_params/1e6:.2f} M, trainable={trainable_params/1e6:.2f} M, size≈{size_mb:.1f} MB")
        except Exception as e:
            print(f"[Model] Param/size report failed: {e}")

    # ---- GFLOPs report (requires a dummy forward) ----
    if args.report_flops:
        try:
            from thop import profile
            net.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dummy_img = torch.randn(1, 1, args.img_size, args.img_size, device=device)
            if args.prompt:
                # Use minimal one-hot sizes consistent with pred.py: pos(8), task(2), type(3), nature(2)
                position_prompt = torch.zeros(1, 8, device=device); position_prompt[0,0] = 1.0
                task_prompt = torch.zeros(1, 2, device=device); task_prompt[0,0] = 1.0
                type_prompt = torch.zeros(1, 3, device=device); type_prompt[0,0] = 1.0
                nature_prompt = torch.zeros(1, 2, device=device); nature_prompt[0,1] = 1.0
                # TransUnetTM.forward expects a SINGLE positional argument; when using prompts,
                # the code passes a tuple as that single argument. So wrap the composite in another tuple.
                composite = (dummy_img, position_prompt, task_prompt, type_prompt, nature_prompt)
                inputs = (composite,)
            else:
                inputs = (dummy_img,)
            macs, _ = profile(net, inputs=inputs, verbose=False)
            print(f"[Model] FLOPs: {macs/1e9:.3f} GFLOPs (THOP)")
        except Exception as e:
            print(f"[Model] FLOPs report skipped ({e}). You may install THOP: pip install thop")

    if args.report_only:
        # Skip loading checkpoint and inference
        sys.exit(0)

    # Resolve checkpoint
    if args.ckpt is not None and os.path.exists(args.ckpt):
        snapshot = args.ckpt
    else:
        snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))

    device = torch.device("cuda")
    model = net.to(device=device)
    # Load checkpoint (strip 'module.' if present)
    ckpt = torch.load(snapshot, map_location=device)
    if any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', '', 1): v for k, v in ckpt.items()}
    msg = model.load_state_dict(ckpt, strict=False)

    print("self trained swin unet", msg)
    snapshot_name = snapshot.split('/')[-1]

    logging.basicConfig(filename=args.output_dir+"/"+"test_result_tm.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_saveout:
        args.test_save_dir = os.path.join(args.output_dir, "predictions_tm")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
