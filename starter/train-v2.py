#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from PIL import Image, ImageFile
import os

import argparse

def test(model, test_dataset, batch_size, criterion):

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    test_loss_accumulator = 0
    test_correct_accumulator= 0
    model.eval()
    
    for batch_data, batch_target in test_loader:
        batch_pred = model(batch_data) # forward pass
        batch_loss = criterion(batch_pred, batch_target) # calculate loss
        test_loss_accumulator += batch_loss
        batch_pred = batch_pred.argmax(dim=1, keepdim=True) # pred is a 2-d tensor of shape (batch_size, 1) which contains the number of the predicted class
        test_correct_accumulator += batch_pred.eq(batch_target.view_as(batch_pred)).sum().item() # add 1 for every correct predicted image
    
    print(f"Test: Accuracy: {100* (test_correct_accumulator/len(test_loader.dataset))}%")
    print(f"Test: Average loss: {(test_loss_accumulator/len(test_loader.dataset))}")

    pass

def train(model, train_dataset, k_folds, batch_size, criterion, optimizer, epochs):

    targets = np.array(train_dataset.targets)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = {}

    # StratifiedKFold is just splitting on the indices, it is not splitting the actual data
    for fold, (train_ids, validation_ids) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"Processing fold number {fold+1}")

        # With the ids we can then get the actual data
        train_subsampler = SubsetRandomSampler(train_ids) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)

        validation_subsampler = SubsetRandomSampler(validation_ids) 
        validation_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=validation_subsampler)
        
        for epoch in range(epochs):
            
            model.train()
    
            for batch_data, batch_target in train_loader: 
                optimizer.zero_grad() # reset gradients
                batch_pred = model(batch_data) # forward pass
                batch_loss = criterion(batch_pred, batch_target) # calculate loss 
                batch_loss.backward() # calculate gradients
                optimizer.step() # update weights
    
            model.eval()

            # Validation
            epoch_validation_loss_accumulator = 0
            epoch_validation_correct_accumulator = 0
        
            with torch.no_grad():
                for batch_data, batch_target in validation_loader:
                    batch_pred = model(batch_data) # forward pass
                    batch_loss = criterion(batch_pred, batch_target) # calculate loss
                    epoch_validation_loss_accumulator += batch_loss
                    
                    # batch_pred is a 2-d tensor of shape (batch_size, 1) which contains the number of the predicted class
                    batch_pred = batch_pred.argmax(dim=1, keepdim=True)
                    
                    # add 1 for every correct predicted image
                    epoch_validation_correct_accumulator += batch_pred.eq(batch_target.view_as(batch_pred)).sum().item() 

            epoch_validation_accuracy = 100 * (epoch_validation_correct_accumulator/len(validation_loader.dataset))
            epoch_validation_average_loss = (epoch_validation_loss_accumulator/len(validation_loader.dataset))
            
            print(f"Epoch #{epoch}: Validation accuracy: {epoch_validation_accuracy}%, Validation average loss: {epoch_validation_average_loss}")
        
        fold_results[fold] = {'epoch_validation_average_loss': epoch_validation_average_loss, 'epoch_validation_accuracy': epoch_validation_accuracy}

    avg_val_loss = np.mean([fold_results[f]['epoch_validation_average_loss'] for f in fold_results])
    avg_val_acc = np.mean([fold_results[f]['epoch_validation_accuracy'] for f in fold_results])

    print(f"All folds completed: Validation average loss: {avg_val_loss}, Validation average accuracy : {avg_val_acc}")
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    #model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = models.resnet50(pretrained=True)

    # freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # last layer needs 5 output neuros for the 5 classes we have
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 5)

    # to be explicit about what will be trained
    for param in model.fc.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    
    return model

def is_file_an_image(path):
  try:
    im = Image.open(path)
    return True
  except:
    print(f"File: {path} is not an image. Ignoring")
    return False

def create_data_sets():
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    train_data_path = os.environ['SM_CHANNEL_TRAIN']
    test_data_path = os.environ['SM_CHANNEL_TEST']

    # From https://pytorch.org/vision/0.19/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    # ResNet50_Weights.IMAGENET1K_V2:
    # The images are resized to resize_size=[256] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224]. 
    # Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(), # ToTensor scales to range [0.0, 1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(), # ToTensor scales to range [0.0, 1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Every directory in the directory root is considered a class, and every file in a 'class' directory is an observation/image.
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform, is_valid_file=is_file_an_image)
    #train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=eval_transform, is_valid_file=is_file_an_image)
    #test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataset, test_dataset

def main(args):

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    print("Start model training")
    train_dataset, test_dataset = create_data_sets()
    model=train(model, train_dataset, args.k_folds, args.batch_size, loss_criterion, optimizer, args.epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    print("Start model testing")
    test(model, test_dataset, args.batch_size, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'resnet50-classify.pth'))

if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--k-folds', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    
    main(args)