import torch
import torch.nn.functional as F

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
    model.train()
    
    total_loss = []
    
    for batch_idx, (x, label) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        x = x.to(device)
        label = label.to(device)
        
        output = model(x)
        
        loss = loss_fn(output, label)
        
        total_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.6f}", end="")
    print()
    
    return sum(total_loss)/len(total_loss)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    
    total_mse_loss = []
    total_mae_loss = []
    for batch_idx, (x, label) in enumerate(dataloader, start=1):
        x = x.to(device)
        label = label.to(device)
        
        output = model(x)
        
        mse_loss = F.mse_loss(output, label)
        mae_loss = F.l1_loss(output, label)
        
        total_mse_loss.append(mse_loss.item())
        total_mae_loss.append(mae_loss.item())
        
        print(f"\rEvaluate: {100*batch_idx/len(dataloader):.2f}%", end="")
    print()
    
    result = {
        'MSE Loss': sum(total_mse_loss)/len(total_mse_loss),
        'MAE Loss': sum(total_mae_loss)/len(total_mae_loss)
    }
    
    return result