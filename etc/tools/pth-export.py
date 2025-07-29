import torch
import json
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Export PIPNet model to JSON")
    parser.add_argument('--model_path', type=str, help='Path to .pth file')
    parser.add_argument('--output_json', type=str, help='Output JSON path')
    parser.add_argument('--compact', action='store_true', help='Export counts for large arrays (>20 elements)')
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path
    output_json = args.output_json
    compact = args.compact

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint

    # Metadata (defaults for 300W)
    num_landmarks = checkpoint.get('num_landmarks', 68)
    num_neighbors = checkpoint.get('num_neighbors', 10)
    neighbor_indices = checkpoint.get('neighbor_indices', None)

    # Generate placeholder neighbor indices if missing
    if neighbor_indices is None:
        neighbor_indices = np.zeros((num_landmarks, num_neighbors), dtype=int)
        for i in range(num_landmarks):
            neighbor_indices[i] = [(i + j + 1) % num_landmarks for j in range(num_neighbors)]

    weights = []
    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            shape = list(tensor.shape)
            if compact and tensor.numel() > 20:
                weights.append({"key": key, "shape": shape, "count": tensor.numel()})
            else:
                data = tensor.cpu().numpy().flatten().astype(np.float32).tolist()
                weights.append({"key": key, "shape": shape, "data": data})

    output = {
        "metadata": {
            "num_landmarks": num_landmarks,
            "num_neighbors": num_neighbors,
            "neighbor_indices": neighbor_indices.tolist()
        },
        "weights": weights
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Exported to {output_json}")

if __name__ == "__main__":
    main()
