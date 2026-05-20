import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"Training on: {device}")

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        total_mae  = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(prediction, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_mae  += torch.mean(torch.abs(batch.y - prediction)).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_mae  = total_mae  / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_mae)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                val_loss += loss_fn(prediction, batch.y).item()
                val_mae  += torch.mean(torch.abs(batch.y - prediction)).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae  = val_mae  / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_mae)

        print(f"Epoch {epoch+1:2d}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies
