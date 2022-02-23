import torch
import torch.nn as nn
from torchvision import datasets, models, transforms 
from torch.utils.data import Dataset, DataLoader
from pascal_dataset import pascal_dataset
from sklearn.metrics import classification_report  
import matplotlib.pyplot as plt 
from focul_loss import FocalLoss 

device = 'cuda:0'

# Class for cnn fine-tuning, we will use resnet model
class custom_fcn(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model = models.resnet18(pretrained=True)  
        # self.model = models.vgg16(pretrained=True).to(device) 
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.model_backbone = torch.nn.Sequential(*(list(self.model.children())[:-2])) 

        self.last_conv = nn.Conv2d(512, n_class, 1)
        self.upsample1 = nn.ConvTranspose2d(n_class, n_class, 9, stride=4, padding=3, bias=False) 
        self.upsample2 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=0, dilation=1, bias=False) 

        # stride 32, padding 16, filter size 64 
 

    def forward(self, x):
        y = self.model_backbone(x)
        # print("backbone output: ", y.shape)
        y = self.last_conv(y)
        # print("last conv output: ", y.shape)
        y = nn.ReLU()(self.upsample1(y))
        # print("upsample output: ", y.shape)
        y = self.upsample2(y)
        # print("upsample2 output: ", y.shape) 
        return y  

# This module has solution for question 2b and 2c - Resnet model trained for only the last layer 
def train_fcn(train_set, test_set):
    n_classes = 21
    model = custom_fcn(n_classes).to(device) # custom model cnn

    # train hyper-parameters
    n_epochs = 50
    batch_size_train = 32 
    batch_size_test = 32
    learning_rate = 0.0002 
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss() 
    focal_criterion = FocalLoss(gamma = 3.0) 

    train_dl = DataLoader(train_set, batch_size = batch_size_train, shuffle = True)  
    test_dl = DataLoader(test_set, batch_size = batch_size_test, shuffle = True)  

    print("train set size: ", len(train_set)) 
    print("test set size: ", len(test_set))     

    overall_losses_train = []
    overall_losses_test = [] 
    overall_true_labels = []
    overall_pred_labels = []

    best_acc = 0

    # Checking the weights status
    # for nm, m in model.named_parameters():
    #     print(nm, m.requires_grad)

    for e in range(n_epochs):
        model.train()
        train_batch_loss = 0
        train_batch_acc = 0 
        test_batch_loss = 0
        test_batch_acc = 0
        
        # Training loop
        for id, batch in enumerate(train_dl):
            x = batch['X'].to(device)
            y = batch['Y'].to(device)


            y_ = model(x) 
        
            optim.zero_grad()
            loss = criterion(y_, y) # Computation of loss
            # loss = focal_criterion(y_, y)  # Focal loss computation to be used for training 
            loss.backward()         # Backward prop of loss 
            optim.step()            # Step taken by the optimizer to compute 

            loss_val = loss.item()
            y_label = torch.argmax(y_, dim=1)

            acc_val = (y==y_label).sum() * 1.0/ (y.shape[0] * y.shape[1] * y.shape[2])

            train_batch_loss += loss_val
            train_batch_acc += acc_val.item() 

            # print("[Ep:{}] [{}/{}] - train loss: {}, train acc: {}".format(e, id, len(train_dl), loss_val, acc_val.item()))
        
        # Test loop
        model.eval()
        for id, batch in enumerate(test_dl):
            x = batch['X'].to(device)
            y = batch['Y'].to(device)

            y_ = model(x)

            loss = criterion(y_, y) # Computation of loss
            # loss = focal_criterion(y_, y)  # Focal loss computation to be used for training 
            loss_val = loss.item()

            y_label = torch.argmax(y_, dim=1)

            acc_val = (y==y_label).sum() * 1.0/ (y.shape[0] * y.shape[1] * y.shape[2]) 

            if (e==n_epochs-1): # If last epoch then compute the stats 
                overall_true_labels += list(y.detach().cpu().numpy())
                overall_pred_labels += list(y_label.detach().cpu().numpy())

            test_batch_loss += loss_val
            test_batch_acc += acc_val.item()

        train_avg_loss = train_batch_loss / len(train_dl)
        train_avg_acc = train_batch_acc / len(train_dl)
        test_avg_loss = test_batch_loss / len(test_dl)
        test_avg_acc = test_batch_acc / len(test_dl)  

        if (test_avg_acc > best_acc):
            torch.save(model.state_dict(), './weights/custom_model_best_ce.pth')
            best_acc = test_avg_acc 

        overall_losses_train.append(train_avg_loss)
        overall_losses_test.append(test_avg_loss) 

        print("[Ep:{}] - train loss: {}, test loss: {}, train acc: {}, test acc: {}".format(e, round(train_avg_loss,3), round(test_avg_loss,3),
                                    round(train_avg_acc,3), round(test_avg_acc, 3))) 

    # print("Classification Stats after epoch {}: ".format(n_epochs))
    # print(classification_report(overall_true_labels, overall_pred_labels)) 

    # Plotting the train losses over iterations 
    plt.plot(overall_losses_train, label='train-loss')
    plt.plot(overall_losses_test, label='test-loss')
    plt.title('Training and Testing loss over training iterations')
    plt.xlabel('iterations')
    plt.ylabel('loss_val')
    plt.legend()
    plt.savefig('./figures/ce_loss_custom_model.png') 




def run_main():
    model = custom_fcn(21).to(device)

    x = torch.rand(4, 3, 512, 771).cuda()
    y = model(x)
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)

    img_tforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop((480, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    seg_tforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop((480, 320)),
        transforms.ToTensor()
    ]) 

    img_tforms_test = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    seg_tforms_test = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ]) 

    root_folder = './data_pascal/VOCdevkit/VOC2012'
    train_dataset = pascal_dataset(root_folder, 'train', img_tforms, seg_tforms)
    test_dataset = pascal_dataset(root_folder, 'val', img_tforms, seg_tforms)  

    train_fcn(train_dataset, test_dataset)


if __name__ == "__main__":
    run_main() 
