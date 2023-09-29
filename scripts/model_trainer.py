import torch 

def train_model(model, criterion, optimizer, num_epochs, dataloaders, datasets):
    """Model train function

    Args:
        model (pytorch): Pytorch model
        criterion (Torch loss): Pytorch loss function
        optimizer (Torch): Pytorch optimizer
        num_epochs (int): Number of epochs
        dataloaders (dict): Dictionary of dataloaders
        datasets (dict): Dictionary of datasets

    Returns:
        pytorch: model
    """
    dev =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels,_ in dataloaders[phase]:
                inputs = inputs.float().to(dev)
                labels = labels.to(dev)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return model