# Training loop cell — URGENT fix (autocast/BCE crash)

**What to do:** In `training/train_colab.ipynb` open in Colab, **DELETE** the
contents of the **training loop cell** (Phase (g) — "Training Loop"),
**paste** the ENTIRE code block below into that cell as-is, and re-run
**only this cell**.

**DO NOT RE-RUN THE EARLIER CELLS** — the model, optimizer, data loaders and
checkpoint/resume state are already in memory; re-running the cells from
scratch may break the resume state / triggers needless reloading.

**Why it crashed:** `torch.nn.BCELoss()` (the gdt loss) was being called
inside the autocast(bf16) block — PyTorch considers this unsafe and raises
an error. The fix below computes the gdt loss AND the `pix_loss` call
(because the `bce` component inside the official BiRefNet `PixLoss` uses the
same raw `BCELoss` pattern) with inputs upcast to fp32 via
`torch.autocast(..., enabled=False)`. The math is THE SAME — only these two
loss calls now run in fp32; the model forward (the real heavy compute) still
stays under bf16 autocast.

```python
import time
import traceback

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
TRAIN_LOG_PATH = STATUS_DIR / "train_log.txt"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

UNITS_PER_HOUR_A100 = 13  # approximate (Colab A100 ~11-13 units/hour); verify the exact value in Colab's "Resources" panel.


def log_epoch_row(epoch: int, loss: float, lr_now: float, elapsed_sec: float, eval_mae: float | None) -> None:
    row = f"epoch={epoch}\tloss={loss:.6f}\tlr={lr_now:.8f}\ttime_sec={elapsed_sec:.1f}"
    if eval_mae is not None:
        row += f"\teval_mae={eval_mae:.6f}"
    print(row)
    # Writing the log to Drive is best-effort: a transient Drive I/O error must NOT KILL
    # training (the row was already printed to the console; the next epoch's row will be tried again).
    try:
        with open(TRAIN_LOG_PATH, "a") as f:
            f.write(row + "\n")
    except OSError as e:
        print(f"  WARNING: could not write to train_log.txt ({e}) — training continues, will retry next epoch.")


def save_and_sync_checkpoint(epoch: int) -> None:
    raw_state = model.state_dict()  # with torch.compile it may carry the '_orig_mod.' prefix -- SAME behavior as the official train.py
                                     # (saves WITHOUT removing the prefix); check_state_dict is always applied at load time.
    payload_out = {
        "model": raw_state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    # 1) Local disk FIRST — this must always succeed (if it fails there is a real problem, the error propagates).
    local_path = local_ckpt_dir / f"epoch_{epoch}.pth"
    torch.save(payload_out, local_path)
    tcl.prune_old_checkpoints(local_ckpt_dir, KEEP_LAST_N_CHECKPOINTS)

    # 2) THEN Drive — best-effort: a transient Drive I/O error (quota/sync stall)
    # does not kill training; the local copy is safe, and the NEXT epoch will try a
    # new Drive copy (in the worst case Drive lags 1 epoch behind).
    try:
        drive_path = drive_ckpt_dir / f"epoch_{epoch}.pth"
        shutil.copy2(local_path, drive_path)
        tcl.prune_old_checkpoints(drive_ckpt_dir, KEEP_LAST_N_CHECKPOINTS)
        print(f"  checkpoint saved + copied to Drive: {drive_path}")
    except OSError as e:
        print(f"  WARNING: checkpoint could not be copied to Drive ({e}) — LOCAL copy is safe: {local_path}; "
              f"will retry next epoch.")


def train_one_epoch(epoch: int) -> float:
    model.train()

    # --- Fine-tune trick: recompute FROM THE BASE according to the absolute epoch number (resume-safe, see the note above) ---
    pix_loss.lambdas_pix_last = dict(BASE_LAMBDAS_PIX_LAST)
    if tcl.should_apply_finetune_reweight(epoch, EPOCHS, config.finetune_last_epochs):
        n = epoch - (EPOCHS + config.finetune_last_epochs)
        if config.task == "Matting":
            pix_loss.lambdas_pix_last["mse"] = BASE_LAMBDAS_PIX_LAST["mse"] * (0.9 ** n)
            pix_loss.lambdas_pix_last["ssim"] = BASE_LAMBDAS_PIX_LAST["ssim"] * (0.9 ** n)
        else:
            pix_loss.lambdas_pix_last["bce"] = BASE_LAMBDAS_PIX_LAST["bce"] * 0
            pix_loss.lambdas_pix_last["iou"] = BASE_LAMBDAS_PIX_LAST["iou"] * (0.5 ** n)
            pix_loss.lambdas_pix_last["mae"] = BASE_LAMBDAS_PIX_LAST["mae"] * (0.9 ** n)

    running_sum, running_n = 0.0, 0
    n_batches = len(train_loader)
    optimizer.zero_grad()
    for micro_step, batch in enumerate(train_loader):
        inputs = batch[0].to(device, non_blocking=True)
        gts = batch[1].to(device, non_blocking=True)
        class_labels = batch[2].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            scaled_preds, class_preds_lst = model(inputs)
            if config.out_ref:
                (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds

        # why: BCE (BCELoss / binary_cross_entropy) is forbidden under autocast; SAME math as the official
        # BiRefNet train.py/loss.py, only this block (gdt + pix_loss) is computed in fp32.
        with torch.autocast(device_type="cuda", enabled=False):
            if config.out_ref:
                loss_gdt = None
                for gi, (gp, gl) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                    gp_i = torch.nn.functional.interpolate(gp.float(), size=gl.shape[2:], mode="bilinear", align_corners=True).sigmoid()
                    gl_i = gl.float().sigmoid()
                    li = criterion_gdt(gp_i, gl_i)
                    loss_gdt = li if gi == 0 else loss_gdt + li
            loss_cls = 0.0 if None in class_preds_lst else cls_loss(class_preds_lst, class_labels)
            # pix_loss (the official PixLoss) also uses raw BCELoss in its 'bce' component -- upcast to fp32 for the same reason.
            scaled_preds_f = [sp.float() for sp in scaled_preds]
            loss_pix, _loss_dict_pix = pix_loss(scaled_preds_f, torch.clamp(gts, 0, 1).float(), pix_loss_lambda=1.0)
            loss = loss_pix + loss_cls
            if config.out_ref:
                loss = loss + loss_gdt * 1.0

        (loss / ACCUM).backward()
        if (micro_step + 1) % ACCUM == 0 or (micro_step + 1) == n_batches:
            optimizer.step()
            optimizer.zero_grad()

        running_sum += loss.item() * inputs.size(0)
        running_n += inputs.size(0)
        if micro_step % 200 == 0:
            print(f"  epoch {epoch} iter {micro_step}/{n_batches} loss={loss.item():.5g}")

    lr_scheduler.step()
    return running_sum / max(running_n, 1)


def main() -> None:
    for epoch in range(epoch_st, EPOCHS + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(epoch)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        eval_mae = None
        if epoch % N_EVAL_EVERY == 0 or epoch == EPOCHS:
            eval_mae = run_quick_eval(model, EVAL_STEMS, local_val_im, local_val_gt, device)

        log_epoch_row(epoch, avg_loss, current_lr, elapsed, eval_mae)
        save_and_sync_checkpoint(epoch)

        # MEASURED cost report (compare with the theoretical table in the parameter cell):
        hours = elapsed / 3600
        est_units = hours * UNITS_PER_HOUR_A100
        remaining = EPOCHS - epoch
        print(f"  COST: this epoch {hours:.2f} hours ≈ {est_units:.0f} units "
              f"(assuming A100 ~{UNITS_PER_HOUR_A100} units/hour); "
              f"remaining {remaining} epochs ≈ {remaining * hours:.1f} hours ≈ {remaining * est_units:.0f} units. "
              f"If it will exceed your budget, stop now — RESUME='auto' resumes from where it left off.")

    print("TRAINING COMPLETE.")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    print(tb)
    try:  # the FATAL record is also best-effort — if Drive cannot be written, it must not overshadow the real error.
        with open(TRAIN_LOG_PATH, "a") as f:
            f.write(f"epoch=FATAL\ttraceback={tb!r}\n")
    except OSError as log_err:
        print(f"WARNING: FATAL record could not be written to train_log.txt ({log_err}).")
    raise
```
