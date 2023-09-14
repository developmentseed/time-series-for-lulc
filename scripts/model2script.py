from model import PixelLM
import torch
import numpy as np

def convert():   
    CKPT = "/home/tam/Documents/devseed/labs-lulc/checkpoints/epoch_18-step_2204-loss_0.939-f1score_0.813.ckpt"
    model = PixelLM.load_from_checkpoint(CKPT)

    script = model.to_torchscript()
    torch.jit.save(script, "ts-model.pt")

def predict():
    # Load a chip
    chip_id = "/home/tam/Documents/repos/time-series-for-lulc/data/cubesxy/3435-7280-14.npz"
    data = np.load(chip_id, allow_pickle=True)
    # data["X"] = w, h, time, band; data["y"] = w, h
    X = data["X"]
    shape = X.shape
    X = X.reshape(-1, 10, 13).astype(np.float32)

    new_model = torch.jit.load("ts-model.pt")
    X = torch.from_numpy(X).to("cuda")
    logits = new_model(X)
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    pred = pred.reshape(shape[:2])
    print(pred.shape)

if __name__ == "__main__":
    predict()