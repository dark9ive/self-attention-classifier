from utils.dataloader import *
from utils.model import *
from constant import *

def train():
    device = torch.device(DEVICE)
    model = SelfAttentionClassifier(EMBED_DIM, HEADS, CLASS_NUM, device).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.NLLLoss()

    for epoch in range(EPOCHS):
        for i, (inputs, targets, lens) in enumerate(MyTrainDataLoader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs, lens)
            loss = loss_fn(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss for every 100th batch
            if (i+1) % 100 == 0:
                total_test = 0
                total_correct = 0
                for _, (inputs, targets, lens) in enumerate(MyTrainDataLoader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs, lens)
                    loss_test = loss_fn(outputs, targets)

                    total_test += len(lens)
                    for y_pred, y_true in zip(outputs, targets):
                        if max(y_pred) == y_pred[y_true]:
                            total_correct += 1
                
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(MyTrainDataLoader)}], Loss: {round(loss.item(), 4)}, Loss@Test: {round(loss_test.item(), 4)}, Acc@Test: {round(total_correct / total_test, 4)}')

if __name__ == '__main__':
    train()
