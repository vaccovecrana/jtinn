import torch
import torchvision.models as models

# Create the model without pretrained weights
model = models.resnet18(pretrained=False)

# Set all convolutional weights (4D tensors) to 0.1
# Set batch norm gamma (1D 'weight') to 1.0, beta/bias to 0.0
for name, param in model.named_parameters():
    if param.dim() == 4:  # Convolutional weights
        param.data.fill_(0.1)
    if 'bias' in name:  # All biases, including batch norm beta
        param.data.fill_(0.0)
    elif param.dim() == 1:  # Batch norm gamma (skips if it's a bias)
        param.data.fill_(1.0)

# Set running means to 0.0 and running vars to 1.0
for name, buffer in model.named_buffers():
    if 'running_mean' in name:
        buffer.data.fill_(0.0)
    elif 'running_var' in name:
        buffer.data.fill_(1.0)

# Create input tensor of shape (1, 3, 32, 32) filled with 1.0
input_tensor = torch.ones(1, 3, 32, 32)

# Run forward pass to get the feature map after layer4
model.eval()
with torch.no_grad():
    x = model.conv1(input_tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    feature_map = model.layer4(x)

# Flatten the feature map (should be 512 values) and print as Java float array
flat = feature_map.view(-1).tolist()
print('new float[] {')
print(', '.join(f'{val:.8f}f' for val in flat))
print('};')