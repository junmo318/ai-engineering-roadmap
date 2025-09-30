import torch
import sys

def load_model(path='../models/linear_model.pth'):
    state = torch.load(path, map_location='cpu')
    return state['w'], state['b']

def predict(x_value, w, b):
    x = torch.tensor([[float(x_value)]])
    y = x.mm(w) + b
    return float(y.item())

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <x_value>")
        sys.exit(1)
    w, b = load_model()
    x_val = sys.argv[1]
    y_pred = predict(x_val, w, b)
    print(f"Input {x_val} -> Predicted y = {y_pred:.4f}")