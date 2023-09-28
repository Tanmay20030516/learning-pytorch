import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

# 1. Class weighting -> can be done too
# bombay_cat (10), persian_cat (50)
loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor([5, 1]),  # relatively giving more weight to bombay cat
)


# 2. Random sampling (weighted) -> preferred
def get_loader(root_dir, batch_size):
    # transformations
    transformations = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
        ]
    )

    # creating dataset
    dataset = datasets.ImageFolder(root=root_dir, transform=transformations)
    # print(dataset.class_to_idx)
    subdirectories = dataset.classes  # get the names of sub folders (class names) in root dir

    class_weights = []  # list containing class weights

    for subdirectory in subdirectories:
        images_path = os.listdir(os.path.join(root_dir, subdirectory))  # entered the subdirectory
        class_weights.append(1/len(images_path))  # more class examples => less class weight
    # just an alternate logic for above code
    # for root, subdir, files in os.walk(root_dir):
    #     if len(files) > 0:
    #         class_weights.append(1 / len(files))

    # setting up sample weights (weight given to each example in the dataset)
    sample_weights = list([0] * len(dataset))
    for index, (data, label) in enumerate(iter(dataset)):
        class_weight = class_weights[label]
        sample_weights[index] = class_weight  # giving each sample a weight corr to its class

    # now we will do oversampling in accordance with the sample weights
    random_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,  # else we have example occurring only once
    )

    # creating our data loader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=random_sampler,
    )
    print("Dataset loaded...")
    return loader


def main():
    data_loader = get_loader(root_dir="data", batch_size=8)
    num_bombay_cats = 0
    num_persian_cats = 0
    for epoch in range(15):
        for data, labels in data_loader:
            num_bombay_cats += torch.sum(labels == 0)
            num_persian_cats += torch.sum(labels == 1)

    print("After weighted random sampling")
    print(f"No of Bombay cats: {num_bombay_cats}")
    print(f"No of Persian cats: {num_persian_cats}")


if __name__ == "__main__":
    main()
