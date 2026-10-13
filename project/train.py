"""Train one method on one outer fold (4 training / 1 internal-validation / 1 test patient).

The internal-validation loss drives ReduceLROnPlateau and checkpoint selection; the
outer test patient is loaded only after the best checkpoint has been fixed.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .models.losses import LOSS_DEFINITION, objective
from .models.network import (CT_INPUTS, DEFAULT_CT_INPUT, build_model, deterministic, load_checkpoint,
                             make_optimizer, patient_order, predict, settings, tensors)
from .dataloader.protocol import METHODS, folds, load_cases, save_json, validate_fold, verify_inputs
from .utils.config import load_config


class BestSelection:
    def __init__(self, validation_patient):
        self.validation_patient = validation_patient
        self.loss = math.inf
        self.epoch = None

    def consider(self, epoch, loss, patient):
        if patient != self.validation_patient:
            raise ValueError("checkpoint selection must use the internal-validation patient")
        if not math.isfinite(loss):
            raise ValueError("nonfinite validation loss")
        if loss < self.loss:
            self.loss, self.epoch = float(loss), int(epoch)
            return True
        return False


def evaluate(model, case, method, device, ct_input):
    model.eval()
    with torch.inference_mode():
        ct, surrogate, rpm, target = tensors(case, device, method, ct_input=ct_input)
        prediction, flow = predict(model, method, ct, surrogate, rpm)
        return [value.item() for value in objective(prediction, target, flow)]


def run(method, fold, data, output, epochs=None, device="cuda:0", ct_input=DEFAULT_CT_INPUT):
    split = folds()[fold - 1]
    validate_fold(split)
    verify_inputs(data, split["train"] + [split["validation"]])
    out = Path(output) / method / f"fold_{fold:02d}"
    out.mkdir(parents=True, exist_ok=False)
    config = settings(method, ct_input)
    epochs = epochs or config["epochs"]
    config.update(method=method, split=split, epochs=epochs, precision="FP32", recompute=True,
                  loss=LOSS_DEFINITION, torch=str(torch.__version__), cuda=torch.version.cuda)
    deterministic(config["seed"])
    device = torch.device(device)
    model = build_model(method, channels=config["channels"]).to(device)
    config["parameters"] = sum(p.numel() for p in model.parameters())
    optimizer, scheduler = make_optimizer(model, method)
    selector = BestSelection(split["validation"])
    cases = load_cases(data, split["train"] + [split["validation"]], forbidden={split["test"]})
    initial = evaluate(model, cases[split["validation"]], method, device, ct_input)[0]
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError("invalid initial internal-validation loss")
    config["initial_validation_loss"] = initial
    save_json(out / "config.json", config)
    for epoch in range(1, epochs + 1):
        tick = time.monotonic()
        model.train()
        order = patient_order(split["train"], method)
        totals = []
        for patient in order:
            ct, surrogate, rpm, target = tensors(cases[patient], device, method, training=True,
                                                 ct_input=ct_input)
            optimizer.zero_grad(set_to_none=True)
            prediction, flow = predict(model, method, ct, surrogate, rpm)
            loss, mse, smooth = objective(prediction, target, flow)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            totals.append([loss.item(), mse.item(), smooth.item()])
            del ct, surrogate, rpm, target, prediction, flow, loss, mse, smooth
        validation, val_mse, val_smooth = evaluate(model, cases[split["validation"]], method, device,
                                                   ct_input)
        if selector.consider(epoch, validation, split["validation"]):
            torch.save(dict(model=model.state_dict(), epoch=epoch, validation_loss=validation,
                            split=split, config=config), out / "best.pt")
        scheduler.step(validation / initial)
        means = np.mean(totals, axis=0)
        row = dict(epoch=epoch, train_loss=float(means[0]), train_mse=float(means[1]),
                   train_smoothness=float(means[2]), val_loss=validation, val_mse=val_mse,
                   val_smoothness=val_smooth, relative_val_loss=validation / initial,
                   lr=optimizer.param_groups[0]["lr"], best_epoch=selector.epoch,
                   seconds=time.monotonic() - tick, patient_order=order)
        with (out / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(dict(method=method, fold=fold, **row)), flush=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), epoch=epochs, split=split, config=config),
               out / "last.pt")
    save_json(out / "selection.json", dict(epoch=selector.epoch, val_loss=selector.loss,
                                           validation_patient=split["validation"], frozen=True))
    # Outer test patient: loaded only now, evaluated once with the frozen best checkpoint.
    verify_inputs(data, [split["test"]])
    model.load_state_dict(load_checkpoint(out / "best.pt")["model"], strict=True)
    model.eval()
    case = load_cases(data, [split["test"]])[split["test"]]
    with torch.inference_mode():
        ct, surrogate, rpm, _ = tensors(case, device, method, ct_input=ct_input)
        prediction, flow = predict(model, method, ct, surrogate, rpm)
    if not torch.isfinite(prediction).all() or not torch.isfinite(flow).all():
        raise ValueError("nonfinite test output")
    np.savez_compressed(out / "test_prediction.npz", patient=split["test"],
                        prediction=prediction[0, 0].cpu().numpy(), dvf=flow[0].cpu().numpy(),
                        best_epoch=selector.epoch)
    save_json(out / "status.json", dict(state="complete", best_epoch=selector.epoch,
                                        best_validation_loss=selector.loss))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=load_config()["epochs"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ct-input", choices=CT_INPUTS, default=DEFAULT_CT_INPUT,
                        help="reference (default): the encoder reads only the 50%% CT; "
                             "sequence: it reads the 50/70/90%% CT, as in the reported results")
    args = parser.parse_args()
    run(args.method, args.fold, args.data, args.output, args.epochs, args.device, args.ct_input)


if __name__ == "__main__":
    main()
