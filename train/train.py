# -*- coding: utf-8 -*-
import torch
import os
def train(model,train_loader,val_loader,criterion,optimizer,device,flags):
    model.to(device)
    train_loss, train_acc = [],[]
    val_loss, val_acc = [],[]
    for epoch in range(flags['num_epochs']):
        model.train()
        batch_loss, batch_acc = 0.0, 0.0
        count = 0
        for i, (images,labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.cpu().item()
            pred = torch.argmax(output, dim=1)
            batch_acc += (pred==labels).sum().cpu().item()
            count += len(images)
        batch_loss = batch_loss / len(train_loader)
        batch_acc = batch_acc / count * 100.0
        print(f'Epoch: {epoch+1}, Train Loss: {batch_loss}, Train Accuracy: {batch_acc} %')
        train_loss.append(batch_loss)
        train_acc.append(batch_acc)
        
        with torch.no_grad():
            model.eval()
            batch_loss, batch_acc = 0.0, 0.0
            count=0
            for j, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                
                batch_loss += loss.cpu().item()
                pred = torch.argmax(output, dim=1)
                batch_acc += (pred==labels).sum().cpu().item()
                count +=len(images)
            batch_loss /= len(val_loader)
            batch_acc = batch_acc / count * 100.0
            val_loss.append(batch_loss)
            val_acc.append(batch_acc)
            if (epoch%5 == 4):
                print(f'Validation Loss: {batch_loss}, Validation Accuracy: {batch_acc}')
    
    torch.save(model.state_dict(),os.path.join(flags['model_root'],'model_state_dict.pt'))
    return train_loss,train_acc,val_loss,val_acc
    
def test(model,test_loader,criterion,device):
    model.to(device)
    test_loss, test_acc = 0.0,0.0
    count = 0
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        pred = torch.argmax(output, dim=1)
        loss = criterion(output, labels)
        test_loss += loss.cpu().item()
        test_acc += (pred==labels).sum().cpu().item()
        count += len(images)
    test_loss /= len(test_loader)
    test_acc = test_acc / count * 100.0
    return test_loss, test_acc