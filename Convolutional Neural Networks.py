## Imports
import os
import time
import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy.testing as npt
#from torchsummary import summary
# from tqdm import trange

# Checks for the availability of GPU 
is_cuda = torch.cuda.is_available()
if torch.cuda.is_available():
    print("working on gpu!")
else:
    print("No gpu! only cpu ;)")
    
## The following random seeds are just for deterministic behaviour of the code and evaluation

##############################################################################
################### DO NOT MODIFY THE CODE BELOW #############################    
##############################################################################

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = '0'

###############################################################################

import torchvision
import torchvision.transforms as transforms
import os

if not os.path.isdir('./data'):
    os.mkdir('./data')
root = './data/'

# List of transformation on the data - here we will normalize the image data to (-1,1)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5)),])
# Geta  handle to Load the data
training_data = torchvision.datasets.FashionMNIST(root, train=True, transform=transform,download=True)
testing_data = torchvision.datasets.FashionMNIST(root, train=False, transform=transform,download=True)

num_train = len(training_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_bs = 60
test_bs = 50

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Create Data loaders which we will use to extract minibatches of data to input to the network for training
train_loader = torch.utils.data.DataLoader(training_data, batch_size=train_bs,
    sampler=train_sampler, drop_last=False)
valid_loader = torch.utils.data.DataLoader(training_data, batch_size=train_bs, 
    sampler=valid_sampler, drop_last=False)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=test_bs, 
    drop_last=False)


# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

## get a batch of data
images, labels = iter(train_loader).next()


image_dict = {0:'T-shirt/Top', 1:'Trouser', 2:'Pullover', 3:'Dress',
              4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker',
              8:'Bag', 9:'Ankle Boot'}

fig = plt.figure(figsize=(8,8))

print(images.size())

for i in np.arange(1, 13):
    ax = fig.add_subplot(3,4,i, frameon=False)
    img = images[i][0]
    ax.set_title(image_dict[labels[i].item()])
    plt.imshow(img, cmap='gray')

import torch.nn as nn

class Model(nn.Module):
    ## init function is the constructor and we define all the layers used in our model. 
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        
        self.conv1=nn.Conv2d(1,16, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.relu1=nn.ReLU(inplace=True)
        self.mp1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2=nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(32)
        self.relu2=nn.ReLU(inplace=True)
        self.mp2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3=nn.Conv2d(32,64, kernel_size=5, stride=1, padding=2)
        self.bn3=nn.BatchNorm2d(64)
        self.relu3=nn.ReLU(inplace=True)
        self.mp3=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(576,num_classes)
        
    def forward(self, x):
        # We will start with feeding the data to the first layer. 
        # We take the output x and feed it back to the next layer 
        x = self.conv1(x)
        x = self.bn1(x)
        # Continue in ths manner to get the output of the final layer. 
        
        x = self.relu1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
 
    def flatten(self, x):
        N, C, H, W = x.size()
        #reshape x to (N, C*H*W) 
        
        
        x=x.view(N, C*H*W)
        return x

model = Model(num_classes=10)
test_input1 = torch.randn(16,1,28,28)
out1 = model(test_input1)
test_input2 = torch.rand(20,1,28,28)
out2 = model(test_input2)


learning_rate = 1e-2
decayRate = 0.999
epochs = 5
number_of_classes = 10


## First we will define an instance of the model to train
model = Model(num_classes=number_of_classes)
print(model)

#Move the model to the gpu if is_cuda
if is_cuda:
  model = model.cuda()

# define the loss 'criterion' as nn.CrossEntropyLoss() object
# criterion = 

criterion=nn.CrossEntropyLoss()

# Initialize the Adam optimizer for the model.parameters() using the learning_rate
# optimizer = 

optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)


# This is the learning rate scheduler. It decreases the learning rate as we approach convergence
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# optimizer = None

out = torch.FloatTensor([[0.1,0.8,0.05,0.05]])
true = torch.LongTensor([1])
assert criterion(out, true), 0.8925

def train_model(epochs=25, validate=True):
  
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        # Iterate through the batches in the data
        training_loss = 0.0
        validation_loss = 0.0
      
        model.train()
        itr = 0
        for (images,labels)  in train_loader:
            
            if is_cuda:
                device = 'cuda'
                images = images.to(device) #checking images in GPU
                labels = labels.to(device)
                outputs = model(images) #Extraction
                loss = criterion(outputs, labels)
                optimizer.zero_grad() #clar 0 grad gradients
                loss.backward()
                optimizer.step()
                my_lr_scheduler.step()
                training_loss=training_loss+loss #add losses
                
            if itr%100 == 0:
                print('Epoch %d/%d, itr = %d, Train Loss = %.3f, LR = %.3E'\
#                       %(epoch, epochs, itr, loss.item(),optimizer.param_groups[0]['lr']))
            itr += 1
        train_loss.append(training_loss/len(train_loader))
        print('------------------------------------------------')
     
        if validate:
            model.eval()
            with torch.no_grad():
                itr = 0
                for (images,labels)  in valid_loader:
                 
                    if is_cuda:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        validation_loss=validation_loss+loss

                    if itr%100 == 0:
                        print('Epoch %d/%d, itr = %d, Val Loss = %.3f, LR = %.3E'\
#                               %(epoch, epochs, itr, loss.item(),optimizer.param_groups[0]['lr']))
                    itr += 1
                val_loss.append(validation_loss/len(valid_loader))
                print('################################################')
                
    return model, train_loss, val_loss

start = time.time()
trained_model, train_loss, val_loss = train_model(epochs, validate=True)
end = time.time()
print('Time to train in seconds ',(end - start))

it = np.arange(epochs)
plt.plot(it, train_loss, label='training loss')
plt.plot(it, val_loss, label='validation loss')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.legend(loc='upper right')
plt.show()

## Testing Loop

def evaluate_model(model, loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total_samples = 0
        for images, labels in loader:
         
            if is_cuda:
                device = 'cuda'
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                pred=outputs.argmax(axis=1)
            
            total_samples += labels.size(0)
        
        accuracy = correct/total_samples*100
        print("Total Accuracy on the Input set: {} %".format(accuracy))
        return accuracy

# With these settings, obtained 95% train and 91% test accuracy
tr_acc = evaluate_model(model, train_loader)
ts_acc = evaluate_model(model, test_loader)
print('Train Accuracy = %.3f'%(tr_acc))
print('Test Accuracy = %.3f'%(ts_acc))

# test cases for test accuracy > 90%
# hidden tests follow

## Visualize the test samples with predicted output and true output
images, labels = iter(test_loader).next()
# images = images.numpy()
if is_cuda:
  images = images.cuda()
  labels = labels.cuda()

out = model(images)
_, preds = torch.max(out, dim=1)

images = images.to('cpu').numpy()

fig = plt.figure(figsize=(15,15))
for i in np.arange(1, 13):
    ax = fig.add_subplot(4, 3, i)
    plt.imshow(images[i][0])
    ax.set_title("Predicted: {}/ Actual: {}".format(image_dict[preds[i].item()], image_dict[labels[i].item()]), 
                color=('green' if preds[i] == labels[i] else 'red'))

