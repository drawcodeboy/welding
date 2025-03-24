import torch
from torch import nn

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb = nn.Embedding(num_embeddings=8, embedding_dim=2)
        
        self.li1 = nn.Linear(9, 32)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.li2 = nn.Linear(32, 18)
        
        self.acti = nn.ReLU()
    
    def forward(self, x):
        # Number(Sheet) Embedding
        int_encoded = x[:, 0].unsqueeze(1).to(dtype=torch.int64)
        x_rest = x[:, 1:]
        
        int_embedded = self.emb(int_encoded).squeeze()
            
        x = torch.cat((int_embedded, x_rest), dim=1)
        
        # Forward
        x = self.acti(self.bn1(self.li1(x)))
        x = self.li2(x)
        
        # Reshape (B, 9, 2)
        x = x.reshape(-1, 9, 2)
        return x
        

if __name__ == '__main__':
    model = DNN()
    tensor = torch.tensor([[3., 0.73880597, 0.8880597, 0., 0.6641791, 0.55223881, 0.05970149, 0.01492537],
                           [5., 0.81343284, 0.8880597, 0.01492537, 0.32835821, 0.62686567, 0.05223881, 0.01492537]])
    output = model(tensor)
    print(output.shape)