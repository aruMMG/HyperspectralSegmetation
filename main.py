import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


from utils.dataset import createPixelData, makeDataset
from utils.model import DeepSpectra

#Pixel Level Classification
def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)
        
def training_cycle(model, dataloader, optimizer, loss_fn):
    print("training started")
    model.train()
    counter = 0
    train_running_loss = 0.0
    train_running_acc=0.0
    for i, data in tqdm(enumerate(dataloader)):
        counter += 1
        # print(f"class type of input data {type(data)}")
        # extract the features and labels
        features = data['features'].view(-1,1,200).to(device)
        # print(features.size())
        labels = data['labels'].to(device)
        
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        outputs = model(features)
        #outputs = model(features, labels, infer = False)
        loss = loss_fn(outputs, labels)
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs,labels)]
        acc = matches.count(True)/len(matches)
        train_running_loss += loss.item()
        train_running_acc += acc
        
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_acc = train_running_acc / counter   
    print(train_acc) 
    return train_loss, train_acc
def eval(dataloader, model, loss_fn):
    model.eval()
    counter = 0
    running_loss = 0.0
    running_acc=0.0
    matches = []
    mislabel =[]
    with torch.no_grad():
        for data in tqdm(dataloader):
            counter += 1
            # extract the features and labels
            features = data['features'].view(-1,1,200).to(device)
            labels = data['labels'].to(device)
            #index = data['index'].to(device)
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs,labels)]
            acc = matches.count(True)/len(matches)
            running_acc += acc
            
    loss = running_loss / counter
    accuracy = running_acc / counter  
    return accuracy, loss

def test(dataloader, model):
    model.eval()

    correct = 0
    total = 0
    counter = 0
    pred = []
    actual = []
    mislabel =[]
    with torch.no_grad():
        for data in tqdm(dataloader):
            counter+=1
            # extract the features and labels       
            features = data['features'].view(-1,1,200).to(device)
            labels = data['labels'].to(device)
            #index = data['index'].to(device)
            outputs = model(features)
            #print(outputs.shape)
            for i in range(features.shape[0]):            
                real_class = torch.argmax(labels[i])
                predicted_class = torch.argmax(outputs[i])
                actual.append(int(real_class))
                pred.append(int(predicted_class))
            
                #if pred != actual:
                #    mislabel.append(index[i].cpu())

    
    cm = confusion_matrix(actual, pred, labels=[0,1,2,3,4,5,6,7,8])
    return actual, pred, cm#, mislabel

def train(args):
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for run in range(args.folds):
        best_eval_accuracy = 0

        # Assuming you have X and Y arrays
        X_train, X_eval, Y_train, Y_eval = train_test_split(hsi_pixel_data, hsi_pixel_label, test_size=0.2, random_state=42)

        # Creating datasets and dataloader
        train_dataset = makeDataset(X_train, Y_train)
        eval_dataset = makeDataset(X_eval, Y_eval)
        train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size = 128, shuffle=True)

        model = DeepSpectra(args.num_class).to(device)
        optimizer = optim.Adam(params=model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()   
        model.apply(initialize_weights)
        for epoch in range(args.epochs):
            train_loss, train_acc = training_cycle(model, train_dataloader, optimizer, loss_fn)
            print(train_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            if epoch % 1  == 0:
                model_val_acc, model_val_loss = eval(eval_dataloader, model, loss_fn)
                test_loss_list.append(model_val_loss)
                test_acc_list.append(model_val_acc)

                if best_eval_accuracy<=model_val_acc:
                    best_eval_accuracy = model_val_acc
                    best_epoch = epoch+1
                    print(f'saving weight for epoch {epoch+1}')
                    torch.save(
                                {
                                    'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                },
                                f'logFile/{args.log_name}/weights/best_{run}.pt',
                                )
        print(f'saving weight for last')
        torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                    f'logFile/{args.log_name}/weights/last_{run}.pt',
        )


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('--log_name', type=str, default='check', help='Save log file name')
    parser.add_argument('--mask_dir', type=str, default='./FX17/Masks/HSI1_Test', help='input mask directoy')
    parser.add_argument('--data_dir', type=str, default='./FX17/Data/HSI1_Test', help='input data directoy')
    parser.add_argument('--num_class', default=9, type=int, help='number of classes')
    parser.add_argument('--weighted_loss', action='store_true', help='Set the weighted loss to true.')
# Hyp
    parser.add_argument('--weight_folder', default="default", type=str, help='model')

    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--folds', default=5, type=int, help='number of epochs')

    args = parser.parse_args()

    hsi_pixel_data, hsi_pixel_label = createPixelData(args.data_dir, args.mask_dir)
    class_counts = np.sum(hsi_pixel_label, axis=0)
    print(class_counts)


    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on a GPU")
    else:
        device = torch.device("cpu")
        print("Running on a CPU")

    labels = ["PE", "PET", "PP", "PS", "PVC", "PMMA", "ABS", "PC", "Others"]
    train(args)