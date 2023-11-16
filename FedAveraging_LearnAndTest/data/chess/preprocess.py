import numpy as np
import os
import pickle
import torch
from path import Path
from argparse import ArgumentParser
from fedlab.utils.dataset.slicing import noniid_slicing
# from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

current_dir = Path(__file__).parent.abspath()
dir_path = os.path.dirname(os.path.abspath(__file__))
# Specify the name of your CSV file
train_data_dir_path = "Chess_Train/"
test_data_dir_path = "Chess_Test/"
data_dir_path = "Chess/"
# Construct the full file path
train_data_dir = os.path.join(dir_path, train_data_dir_path)
test_data_dir = os.path.join(dir_path, train_data_dir_path)
data_dir = os.path.join(dir_path, data_dir_path)

transform=transforms.Compose([
    transforms.RandomRotation(10),      # rotate +/- 10 degrees
    transforms.RandomHorizontalFlip(),  # reverse 50% of images
    transforms.Resize(224),             # resize shortest side to 224 pixels
    transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
])


class ChessDataset(Dataset):
    def __init__(self, subset) -> None:
        self.data = torch.stack(list(map(lambda tup: tup[0], subset)))
        self.targets = torch.stack(list(map(lambda tup: torch.tensor(tup[1]), subset)))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

def preprocess(args):
    if os.path.isdir(current_dir / "pickles"):
        os.system("rm -rf {}/pickles".format(current_dir))
    chess_train = datasets.ImageFolder(root=train_data_dir, transform=transform)
    chess_test = datasets.ImageFolder(root=test_data_dir, transform=transform)
    
    np.random.seed(args.seed)
    train_idxs = noniid_slicing(
        chess_train, args.client_num_in_total, args.classes * args.client_num_in_total,
    )

    # Set random seed again is for making sure numpy split trainset and testset in the same way.
    np.random.seed(args.seed)
    test_idxs = noniid_slicing(
        chess_test, args.client_num_in_total, args.classes * args.client_num_in_total,
    )
    # Now train_idxs[i] and test_idxs[i] have the same classes.

    all_trainsets = []
    all_testsets = []

    for train_indices, test_indices in zip(train_idxs.values(), test_idxs.values()):
        all_trainsets.append(ChessDataset([chess_train[i] for i in train_indices]))
        all_testsets.append(ChessDataset([chess_test[i] for i in test_indices]))
    os.mkdir(current_dir / "pickles")
    # Store clients local trainset and testset as pickles.
    for i in range(args.client_num_in_total):
        with open("{}/pickles/client_{}.pkl".format(current_dir, i), "wb") as file:
            pickle.dump((all_trainsets[i], all_testsets[i]), file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=4)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    preprocess(args)
