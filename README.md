# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

<img width="840" height="186" alt="image" src="https://github.com/user-attachments/assets/22ad9b3d-76c6-4840-a322-49f6f8ecbf61" />


## DESIGN STEPS
### STEP 1: 


### STEP 1: 

Create a class RNNModel using PyTorch that contains an RNN layer and a fully connected layer to process sequential data and produce the final output.

### STEP 2: 
Set the input size, hidden size, number of layers, and output size. Then create the model, loss function (criterion), and optimizer for training.



### STEP 3: 

Use a train_loader to provide input sequences (x_batch) and target values (y_batch) in batches during training.


### STEP 4: 
For each epoch, pass the input data through the model, calculate the loss using the criterion, perform backpropagation (loss.backward()), and update the weights using the optimizer.
.



### STEP 5: 
Record the training loss for each epoch and plot it using Matplotlib to visualize how the model improves over time




## PROGRAM

### Name:Franklin raj g

### Register Number:212223230058

```python
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out




# Train the Model

def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')
    # Plot training loss
    print('Name:franklin raj G')
    print('Register Number:212223230058')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
train_model = train_model(model, train_loader, criterion, optimizer)


```

### OUTPUT

## Training Loss Over Epochs Plot


<img width="1190" height="645" alt="image" src="https://github.com/user-attachments/assets/80a27f7e-e305-403f-a8f6-6b86fbe229b6" />

## True Stock Price, Predicted Stock Price vs time

<img width="1283" height="638" alt="image" src="https://github.com/user-attachments/assets/4425f4d2-8fea-45d9-ab82-3c50dcb9289f" />


### Predictions
<img width="375" height="57" alt="image" src="https://github.com/user-attachments/assets/028ce9da-6b47-404a-8395-1ffba98176a4" />


## RESULT
thus,To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data, has been done by pytorch.
