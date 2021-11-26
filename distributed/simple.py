import torch
import torch.nn as nn
from ray.train import Trainer
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import ray
from ray import train

num_samples = 20
input_size = 10
layer_size = 15
output_size = 5

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, layer_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(layer_size, output_size)

    def forward(self, input):
        return self.layer2(self.relu(self.layer1(input)))

# In this example we use a randomly generated dataset.
input = torch.randn(num_samples, input_size)
labels = torch.randn(num_samples, output_size)

# this is the single worker
def train_func():

    input = torch.randn(num_samples, input_size)
    labels = torch.randn(num_samples, output_size)


    num_epochs = 10
    model = NeuralNetwork()
    #model = DistributedDataParallel(model)
    
    device = torch.device(f"cuda:{train.local_rank()}" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)

    model = model.to(device)
    loss_fn = nn.MSELoss()
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    #optimizer = optimizer.to(device)

    for epoch in range(num_epochs):
        input, labels = input.to(device), labels.to(device)
        output = model(input)
        
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")



def train_func_distributed():
    num_epochs = 3
    model = NeuralNetwork()
    model = DistributedDataParallel(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        output = model(input)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")


ray.init(address="auto")
trainer = Trainer(backend="torch", num_workers=2, use_gpu=True)
trainer.start()
results = trainer.run(train_func)
trainer.shutdown()
