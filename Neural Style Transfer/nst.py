import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models as models
from torchsummary import summary
import numpy as np


class DeepConvNet(nn.Module):
    def __init__(self, model):
        super(DeepConvNet, self).__init__()
        # we need the conv layer output at above positions (mostly after pooling layers)
        # we just focus on output of chosen_layers
        self.chosen_layers = ["0", "5", "10", "19", "28"]
        self.model = model

    def forward(self, x):
        layer_outputs = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_layers:
                layer_outputs.append(x)

        return layer_outputs


def load_image(img_pth, transform):
    image = Image.open(img_pth)
    image_dims_h_w = image.height, image.width
    image = transform(image).unsqueeze(0)
    return image.to(DEVICE), image_dims_h_w


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512

transformations = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize(  # values from VGG paper
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # ),
    ]
)

content_img, original_dims = load_image("harbor.jpg", transformations)
style_img, _ = load_image("style.jpg", transformations)
# generated_img = torch.randn(content_img.data.shape, device=DEVICE, requires_grad=True)
generated_img = content_img.clone().requires_grad_(True).to(DEVICE)

base_model = models.vgg19(weights="IMAGENET1K_V1").features[:29]
model = DeepConvNet(base_model).to(DEVICE).eval()

# Hyperparameters
NUM_STEPS = 300
LEARNING_RATE = 0.001
ALPHA = 1  # weight for content loss
BETA = 0.01  # weight for style loss (increase to impart more style to image)

# we wish to optimise the generated image
optimizer = optim.Adam([generated_img], lr=LEARNING_RATE)

for step in range(1, NUM_STEPS+1):
    # get feature maps from chosen conv layers
    content_img_maps = model(content_img)
    style_img_maps = model(style_img)
    generated_img_maps = model(generated_img)

    style_loss = 0.
    content_loss = 0.

    for content_img_map, style_img_map, generated_img_map in zip(content_img_maps, style_img_maps, generated_img_maps):
        # batch size of 1
        batch_size, channel, height, width = generated_img_map.shape

        # Calculating style loss
        # Gram Matrix calculation for generated image and style image
        G = generated_img_map.reshape(channel, height*width).mm(generated_img_map.reshape(channel, height*width).t())
        A = style_img_map.reshape(channel, height*width).mm(style_img_map.reshape(channel, height*width).t())
        style_loss += torch.mean((G - A) ** 2)

        # Calculating content loss
        content_loss += torch.mean((generated_img_map - content_img_map) ** 2)

    total_loss = ALPHA*content_loss + BETA*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"{step}: {total_loss}")
        generated_img = transforms.Resize(original_dims, antialias=True)(generated_img)
        # if normalised pixel values, de-normalise it before saving the image
        save_image(generated_img, f"generated/gen{step}.jpg")



