import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from data_loader import get_loader
from model import CaptionModel


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="flickr8k/Images/",
        caption_file="flickr8k/captions.txt",
        transform=transform,
        num_workers=2
    )

    # hyperparameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOAD_MODEL = False
    SAVE_MODEL = False
    TRAIN_CNN = False

    EMBEDDINGS_SIZE = 256
    HIDDEN_SIZE = 256
    VOCAB_SIZE = len(dataset.vocab)
    NUM_LAYERS = 2
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 3

    # tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialise model, loss function, optimizer
    model = CaptionModel(
        embed_size=EMBEDDINGS_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.str_to_idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # fine-tuning the CNN
    for name, param in model.encoderCNN.resnet.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True  # unfreeze the last layer
        else:
            param.requires_grad = TRAIN_CNN  # freeze the rest layers

    if LOAD_MODEL:
        step = load_checkpoint(
            checkpoint=torch.load("checkpoint10.tar"),
            model=model,
            optimizer=optimizer
        )
    model.train()  # set model to train mode after loading checkpoint

    for epoch in range(NUM_EPOCHS):
        # print_examples(model, DEVICE, dataset)

        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader)):
            imgs = imgs.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = model(imgs, captions[:-1])
            loss = loss_fn(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        if SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, epoch)


if __name__ == "__main__":
    train()
