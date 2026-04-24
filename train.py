"""Unified training script for IRSTD benchmark.

Directory layout per experiment:
    log/{dataset}/{model}/seed{N}/
        config.json              # 完整超参数快照
        metrics.csv              # 每 epoch 指标
        best_iou.pth.tar         # 最佳 IoU checkpoint
        best_iou.json            # 最佳 IoU epoch + 指标
        best_niou.pth.tar        # 最佳 nIoU checkpoint
        best_niou.json           # 最佳 nIoU epoch + 指标
        checkpoints/             # 周期性 checkpoint (--save_interval > 0)
            checkpoint_ep{N}.pth.tar
        final_eval.json          # 两个 best checkpoint 的全指标评测
    log/{dataset}/{model}/summary.txt   # 多 seed 聚合结果

Default settings (unified across all models):
    Loss:       BCELogits + SoftIoU (combined)
    Optimizer:  AdamW (lr=3e-4, weight_decay=1e-4)
    Scheduler:  WarmupCosine (5 epochs warmup + cosine, eta_min=1e-6)
    Epochs:     300
    Batch size: 16
    Patch size: 256
    Grad clip:  1.0
"""

import argparse
import csv
import json
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TrainSetLoader, TestSetLoader
from metrics import IoU, nIoU, PD_FA
from net import Net
from utils import seed_pytorch, get_optimizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description="PyTorch IRSTD Train")
parser.add_argument("--model_names", default=["LightUNet"], nargs="+",
                    help="model name: LightUNet, ACM")
parser.add_argument("--dataset_names", default=["IRSTD-1K"], nargs="+",
                    help="dataset name")
parser.add_argument("--dataset_dir", default="./datasets", type=str)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--n_epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--warmup", type=int, default=5,
                    help="linear warmup epochs (0 = no warmup)")
parser.add_argument("--loss", default="BCE+SoftIoU", type=str,
                    choices=["BCE+SoftIoU", "Focal"],
                    help="loss function")
parser.add_argument("--grad_clip", type=float, default=1.0,
                    help="gradient clip max norm (0 = disabled)")
parser.add_argument("--save", default="./log", type=str)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--seeds", default=[42], nargs="+", type=int,
                    help="random seeds to sweep (e.g. 42 123 456)")
parser.add_argument("--save_interval", type=int, default=0,
                    help="save checkpoint every N epochs (0 = only best_iou + best_niou)")
parser.add_argument("--resume", default=None, type=str,
                    help="resume from a run directory (e.g. log/IRSTD-1K/LightUNet/seed42)")

opt = parser.parse_args()


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def save_config(run_dir, model_name, dataset_name, seed, params, flops):
    config = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "epochs": opt.n_epochs,
        "batch_size": opt.batch_size,
        "patch_size": opt.patch_size,
        "optimizer": "AdamW",
        "lr": opt.lr,
        "weight_decay": opt.weight_decay,
        "scheduler": "WarmupCosine" if opt.warmup > 0 else "CosineAnnealingLR",
        "warmup": opt.warmup,
        "min_lr": opt.min_lr,
        "loss": opt.loss,
        "grad_clip": opt.grad_clip,
        "threshold": opt.threshold,
        "params": params,
        "flops": flops,
        "git_commit": get_git_commit(),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return config


def init_metrics_csv(run_dir):
    path = os.path.join(run_dir, "metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "IoU", "nIoU"])
    return path


def append_metrics(csv_path, epoch, lr, loss, iou=None, niou=None):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{lr:.8f}",
            f"{loss:.6f}" if loss is not None else "",
            f"{iou:.6f}" if iou is not None else "",
            f"{niou:.6f}" if niou is not None else "",
        ])


def make_test_loader(dataset_name):
    """Create test DataLoader once, reuse across epochs."""
    test_set = TestSetLoader(opt.dataset_dir, dataset_name, dataset_name)
    return DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)


def make_ckpt(epoch, state_dict, model_name, dataset_name, loss_name, optimizer=None, scheduler=None):
    """Build a checkpoint dict with all state needed for resume / test."""
    ckpt = {
        "epoch": epoch,
        "state_dict": state_dict,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "loss_name": loss_name,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    return ckpt


def eval_fast(net, test_loader, threshold):
    """Fast evaluation: IoU + nIoU only, pure GPU."""
    iou_metric = IoU()
    niou_metric = nIoU()

    with torch.no_grad():
        for img, gt_mask, size, _ in test_loader:
            img = img.cuda()
            gt_mask = gt_mask.cuda()
            logits = net(img)
            prob = logits.sigmoid()
            prob = prob[:, :, : size[0], : size[1]]
            gt_mask = gt_mask[:, :, : size[0], : size[1]]
            binary = (prob > threshold).float()
            iou_metric.update(binary, gt_mask)
            niou_metric.update(binary, gt_mask)

    return iou_metric.get(), niou_metric.get()


def eval_full(net, test_loader, threshold):
    """Full evaluation: IoU + nIoU + PD/FA."""
    iou_metric = IoU()
    niou_metric = nIoU()
    pd_fa_metric = PD_FA()

    with torch.no_grad():
        for img, gt_mask, size, _ in test_loader:
            img = img.cuda()
            gt_mask = gt_mask.cuda()
            logits = net(img)
            prob = logits.sigmoid()
            prob = prob[:, :, : size[0], : size[1]]
            gt_mask = gt_mask[:, :, : size[0], : size[1]]
            binary = (prob > threshold).float()
            iou_metric.update(binary, gt_mask)
            niou_metric.update(binary, gt_mask)
            pd_fa_metric.update(binary[0, 0].cpu(), gt_mask[0, 0].cpu(), size)

    pd, fa = pd_fa_metric.get()
    return {
        "IoU": iou_metric.get(),
        "nIoU": niou_metric.get(),
        "PD": pd,
        "FA": fa,
    }


def run_single(model_name, dataset_name, seed):
    """Train one model on one dataset with one seed."""
    seed_pytorch(seed)
    run_dir = os.path.join(opt.save, dataset_name, model_name, f"seed{seed}")

    # avoid overwriting existing results — append short timestamp
    if os.path.exists(os.path.join(run_dir, "config.json")):
        run_dir = os.path.join(opt.save, dataset_name, model_name,
                               f"seed{seed}_{int(time.time())}")

    os.makedirs(run_dir, exist_ok=True)

    # count params & flops
    tmp_net = Net(model_name=model_name, loss_name=opt.loss).cuda()
    params = sum(p.numel() for p in tmp_net.parameters())
    try:
        from thop import profile
        dummy = torch.randn(1, 1, opt.patch_size, opt.patch_size).cuda()
        flops, _ = profile(tmp_net, inputs=(dummy,), verbose=False)
    except Exception:
        flops = 0
    del tmp_net

    config = save_config(run_dir, model_name, dataset_name, seed, params, flops)

    # build dataloader
    train_set = TrainSetLoader(opt.dataset_dir, dataset_name, opt.patch_size)
    train_loader = DataLoader(train_set, num_workers=opt.threads,
                              batch_size=opt.batch_size, shuffle=True)

    net = Net(model_name=model_name, loss_name=opt.loss).cuda()

    # create test loader once, reuse every epoch
    test_loader = make_test_loader(dataset_name)

    optimizer, scheduler = get_optimizer(
        net, "AdamW", "CosineAnnealingLR",
        {"lr": opt.lr, "weight_decay": opt.weight_decay},
        {"epochs": opt.n_epochs, "min_lr": opt.min_lr, "warmup": opt.warmup},
    )

    # resume support
    start_epoch = 1
    best_iou = 0.0
    best_iou_epoch = 0
    best_niou = 0.0
    best_niou_epoch = 0
    csv_path = None  # set below depending on resume

    print(f"  [seed={seed}] params={params:,}  FLOPs={flops / 1e6:.1f}M")

    if opt.resume:
        resume_dir = opt.resume
        # find latest checkpoint to resume from
        resume_ckpt = None
        ckpt_dir = os.path.join(resume_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_ep")],
                           key=lambda x: int(x.split("ep")[1].split(".")[0]))
            if ckpts:
                resume_ckpt = os.path.join(ckpt_dir, ckpts[-1])

        # fallback to best_niou checkpoint
        if not resume_ckpt:
            best_path = os.path.join(resume_dir, "best_niou.pth.tar")
            if os.path.exists(best_path):
                resume_ckpt = best_path

        if resume_ckpt:
            run_dir = resume_dir  # continue writing to same directory
            ckpt = torch.load(resume_ckpt, weights_only=False)
            ckpt_model_name = ckpt.get("model_name")
            ckpt_dataset_name = ckpt.get("dataset_name")
            if ckpt_model_name is not None and ckpt_model_name != model_name:
                raise ValueError(
                    f"Resume checkpoint model mismatch: expected {model_name}, got {ckpt_model_name}"
                )
            if ckpt_dataset_name is not None and ckpt_dataset_name != dataset_name:
                raise ValueError(
                    f"Resume checkpoint dataset mismatch: expected {dataset_name}, got {ckpt_dataset_name}"
                )
            net.load_state_dict(ckpt["state_dict"])
            start_epoch = ckpt["epoch"] + 1

            # restore optimizer & scheduler state if available
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            else:
                # fallback: step scheduler to correct LR position
                for _ in range(ckpt["epoch"]):
                    scheduler.step()

            # restore best tracking
            for tag in ["best_iou", "best_niou"]:
                jpath = os.path.join(resume_dir, f"{tag}.json")
                if os.path.exists(jpath):
                    with open(jpath) as f:
                        info = json.load(f)
                    if tag == "best_iou":
                        best_iou = info["IoU"]
                        best_iou_epoch = info["epoch"]
                    else:
                        best_niou = info["nIoU"]
                        best_niou_epoch = info["epoch"]

            # keep existing metrics.csv (append, do NOT overwrite)
            csv_path = os.path.join(resume_dir, "metrics.csv")

            print(f"  [seed={seed}] Resuming from ep{ckpt['epoch']}  "
                  f"best_IoU={best_iou:.4f}@{best_iou_epoch}  "
                  f"best_nIoU={best_niou:.4f}@{best_niou_epoch}")
        else:
            print(f"  [seed={seed}] No checkpoint found in {resume_dir}, starting fresh")
            csv_path = init_metrics_csv(run_dir)
    else:
        csv_path = init_metrics_csv(run_dir)
    t_start = time.time()

    for epoch in range(start_epoch, opt.n_epochs + 1):
        # --- train one epoch ---
        net.train()
        losses = []
        for img, gt_mask in train_loader:
            img, gt_mask = img.cuda(), gt_mask.cuda()
            logits = net(img)
            loss = net.loss(logits, gt_mask)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), opt.grad_clip)
            optimizer.step()

        epoch_loss = float(np.mean(losses)) if losses else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- fast eval every epoch (IoU + nIoU, GPU) ---
        net.eval()
        iou_val, niou_val = eval_fast(net, test_loader, opt.threshold)
        append_metrics(csv_path, epoch, current_lr, epoch_loss, iou_val, niou_val)

        # track best IoU
        if iou_val > best_iou:
            best_iou = iou_val
            best_iou_epoch = epoch
            torch.save(make_ckpt(epoch, net.state_dict(), model_name, dataset_name,
                                  opt.loss, optimizer, scheduler),
                       os.path.join(run_dir, "best_iou.pth.tar"))
            with open(os.path.join(run_dir, "best_iou.json"), "w") as f:
                json.dump({"epoch": epoch, "IoU": iou_val, "nIoU": niou_val}, f, indent=2)

        # track best nIoU
        if niou_val > best_niou:
            best_niou = niou_val
            best_niou_epoch = epoch
            torch.save(make_ckpt(epoch, net.state_dict(), model_name, dataset_name,
                                  opt.loss, optimizer, scheduler),
                       os.path.join(run_dir, "best_niou.pth.tar"))
            with open(os.path.join(run_dir, "best_niou.json"), "w") as f:
                json.dump({"epoch": epoch, "IoU": iou_val, "nIoU": niou_val}, f, indent=2)

        # periodic checkpoint (opt-in via --save_interval)
        if opt.save_interval > 0 and epoch % opt.save_interval == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(make_ckpt(epoch, net.state_dict(), model_name, dataset_name,
                                  opt.loss, optimizer, scheduler),
                       os.path.join(ckpt_dir, f"checkpoint_ep{epoch}.pth.tar"))

        # print every 10 epochs or at key points
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  [seed={seed}] ep{epoch:3d}  loss={epoch_loss:.4f}  "
                  f"IoU={iou_val:.4f}  nIoU={niou_val:.4f}  "
                  f"best_IoU={best_iou:.4f}@{best_iou_epoch}  "
                  f"best_nIoU={best_niou:.4f}@{best_niou_epoch}  "
                  f"[{elapsed:.0f}s]")

    # --- full evaluation: run on best_iou and best_niou ---
    print(f"  [seed={seed}] Running full evaluation on best IoU & best nIoU checkpoints...")

    results = {}
    for tag in ["best_iou", "best_niou"]:
        ckpt_path = os.path.join(run_dir, f"{tag}.pth.tar")
        ckpt = torch.load(ckpt_path, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        full = eval_full(net, test_loader, opt.threshold)
        full["checkpoint_epoch"] = ckpt["epoch"]
        results[tag] = full

    # save final eval results
    total_time = time.time() - t_start
    final_output = {"best_iou": results["best_iou"], "best_niou": results["best_niou"],
                    "total_seconds": total_time}
    with open(os.path.join(run_dir, "final_eval.json"), "w") as f:
        json.dump(final_output, f, indent=2)

    for tag in ["best_iou", "best_niou"]:
        r = results[tag]
        print(f"  [seed={seed}] {tag} @ep{r['checkpoint_epoch']}:  "
              f"IoU={r['IoU']:.4f}  nIoU={r['nIoU']:.4f}  "
              f"PD={r['PD']:.4f}  FA={r['FA']:.8f}")

    # update config
    config["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config["total_seconds"] = total_time
    config["best_iou"] = best_iou
    config["best_iou_epoch"] = best_iou_epoch
    config["best_niou"] = best_niou
    config["best_niou_epoch"] = best_niou_epoch
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return results


def summarize(model_name, dataset_name):
    """Aggregate results across seeds and write summary.txt."""
    model_dir = os.path.join(opt.save, dataset_name, model_name)
    seed_results = []

    for seed_dir in sorted(os.listdir(model_dir)):
        if not seed_dir.startswith("seed"):
            continue
        eval_path = os.path.join(model_dir, seed_dir, "final_eval.json")
        config_path = os.path.join(model_dir, seed_dir, "config.json")
        if not os.path.exists(eval_path):
            continue
        with open(eval_path) as f:
            ev = json.load(f)
        with open(config_path) as f:
            cfg = json.load(f)
        seed_results.append({
            "seed": cfg["seed"],
            "params": cfg["params"],
            "flops": cfg["flops"],
            "best_iou": ev["best_iou"],
            "best_niou": ev["best_niou"],
        })

    if not seed_results:
        return

    summary_path = os.path.join(model_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  {model_name} on {dataset_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Params:  {seed_results[0]['params']:,}\n")
        f.write(f"FLOPs:   {seed_results[0]['flops'] / 1e6:.1f}M\n")
        f.write(f"Epochs:  {opt.n_epochs}\n")
        seeds = [s['seed'] for s in seed_results]
        f.write(f"Seeds:   {seeds}\n\n")

        for tag in ["best_iou", "best_niou"]:
            f.write(f"\n--- {tag} ---\n")
            header = f"{'Seed':<8} {'Epoch':<8} {'IoU':<10} {'nIoU':<10} {'PD':<10} {'FA':<14}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for s in seed_results:
                r = s[tag]
                f.write(f"{s['seed']:<8} {r['checkpoint_epoch']:<8} "
                        f"{r['IoU']:<10.4f} {r['nIoU']:<10.4f} "
                        f"{r['PD']:<10.4f} {r['FA']:<14.8f}\n")

            if len(seed_results) >= 2:
                f.write("-" * len(header) + "\n")
                parts = ["Mean±Std  ", " " * 8]
                for k in ["IoU", "nIoU", "PD", "FA"]:
                    vals = [s[tag][k] for s in seed_results]
                    fmt = ".6f" if k == "FA" else ".4f"
                    parts.append(f"{np.mean(vals):{fmt}}±{np.std(vals):{fmt}}  ")
                f.write("".join(parts).strip() + "\n")

        f.write(f"\n{'='*60}\n")

    with open(summary_path) as f:
        print(f.read())


if __name__ == "__main__":
    for dataset_name in opt.dataset_names:
        for model_name in opt.model_names:
            print(f"\n{'='*60}")
            print(f"  {model_name} on {dataset_name}")
            print(f"{'='*60}")

            for seed in opt.seeds:
                run_single(model_name, dataset_name, seed)

            summarize(model_name, dataset_name)
