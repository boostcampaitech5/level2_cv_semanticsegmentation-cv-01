import wandb
import datetime
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from utils import save_model, dice_coef
from time import time

debug_test = False
debug_interval = 2

def validation(epoch, model, classes, data_loader, criterion, thr=0.5):
    print(f"Start validation #{epoch:2d}")
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step,(images, masks) in  tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images, masks = torch.from_numpy(images).cuda(), torch.from_numpy(masks).cuda()
            # images. masks = images.cuda(), masks.cuda()
            outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)
            if debug_test:
                break

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [f"{c:<12}: {d.item():.4f}" for c, d in zip(classes, dices_per_class)]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()
    log_data = {f"class_accuracy/{k}": v.item() for k, v in zip(classes, dices_per_class) }
    log_data.update({"train/val_avg_dice": avg_dice,"epoch":epoch})

    wandb.log(
        log_data
        # step = epoch
    )
    return avg_dice


def train(model, args, data_loader, val_loader, criterion, optimizer, order,epoch,accum_step=1):

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.train()
    for step, (images, masks) in enumerate(data_loader):
        images, masks = torch.from_numpy(images).cuda(), torch.from_numpy(masks).cuda()
        # images, masks = images.cuda(), masks.cuda()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            # inference
            outputs = model(images)
            # loss 계산
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        if (step+1)%accum_step == 0 or step+1 == len(data_loader):
            # torch.nn.utils.clip_grad.clip_grad_norm(model.parameters(),2.0)
            scaler.step(optimizer)
            scaler.update()

        wandb.log({"train/LR": optimizer.param_groups[0]['lr'], "train/loss": loss})
    
        if (step + 1) % 20 == 0:
            print(
                f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                f"Epoch [{epoch+1}/{args.num_epoch}], "
                f"Step [{step+1}/{len(data_loader)}], "
                f"Loss: {round(loss.item(),4)}"
            )
        if debug_test:
            break
        
            
def run(model, args, data_loader, val_loader, criterion, optimizer,scheduler, order,accum_step=1):
    print(f"Start training..")

    best_dice = 0.0
    model = model.cuda()
    
    for epoch in range(args.num_epoch):
        train(model, args, data_loader, val_loader, criterion, optimizer, order,epoch,accum_step=4)
        if scheduler is not None:
            scheduler.step(epoch+1)
        if (epoch + 1) % args.val_every == 0:
            dice = validation(epoch + 1, model, args.classes, val_loader, criterion)
            if best_dice < dice:
                print(
                    f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}"
                )
                print(f"Save model in {args.saved_dir}")
                best_dice = dice
                save_model(
                    model,
                    save_path="/opt/ml/level2_cv_semanticsegmentation-cv-01/pretrain",
                    file_name=f"{args.model_name}_best{order}.pth",
                )
            if epoch+1 == args.num_epoch:
                save_model(
                    model,
                    save_path="/opt/ml/level2_cv_semanticsegmentation-cv-01/pretrain",
                    file_name=f"{args.model_name}_epoch{epoch}.pth"
                )