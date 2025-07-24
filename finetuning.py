# -*- coding: utf-8 -*-
import datasets, models, train, config
from torchvision import transforms
from torch import nn
from torch.optim import Adam

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

flags = config.flags
train_loader,val_loader,test_loader = datasets.get_dataloader(flags['data_root'],transform,batch_size=flags['batch_size'])
model = models.mobilenet(num_classes=100)

from torchinfo import summary
print(summary(model,input_size=(16,3,224,224)))

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(),lr=flags['learning_rate'],weight_decay=1e-4)

train_loss,train_acc,val_loss,val_acc = train.train(model,train_loader,val_loader,criterion,optimizer,'cpu',flags)
test_loss, test_acc = train.test(model,test_loader,criterion,'cpu')

print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc} %')