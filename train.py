# Import necessary libraries
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from datasets.atmospheric_dataset import AtmosphericDataset
import matplotlib.pyplot as plt
from utils.mesh_creation import create_k_nearest_neighboors_edges
from models import gcn
import os
import csv

def train(model, device, epochs, training_dataloader, validation_dataloader, optimizer, scheduler, criterion, training_name):
    saving_path = "trained_models/" + training_name
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    
    print(f"Training on: {device}")
    model.to(device)
    best_val_loss = float("inf")

    # Initialize lists to store training and validation losses
    training_losses = []
    validation_losses = []

    # Save training parameters to a CSV file
    with open(f"{saving_path}/training_parameters.csv", mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["parameter", "value"])
        writer.writerow(["training_name", training_name])
        writer.writerow(["batch_size", BATCH_SIZE])
        writer.writerow(["epochs", epochs])
        writer.writerow(["variables", VARIABLES])
        writer.writerow(["num_variables", NUM_VARIABLES])
        writer.writerow(["hidden_channels", HIDDEN_CHANNELS])
        writer.writerow(["learning_rate", LR])
        writer.writerow(["gamma", GAMMA])
        writer.writerow(["start_year", START_YEAR])
        writer.writerow(["end_year", END_YEAR])

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        print("-" * 50)
        print(f"Epoch: {epoch+1:03d}/{epochs}")
        training_loss = 0
        model.train()
        for i, data in enumerate(training_dataloader):
            data.to(device)
            optimizer.zero_grad()

            # Make prediction and calculate loss
            y_pred = model(data.x, data.edge_index)
            y = data.y
            loss = criterion(y_pred, y)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(
                    f"Batch: {i+1}/{len(training_dataloader)}, Loss: {training_loss/(i+1):.4f}"
                )

        # Step the scheduler
        scheduler.step()

        # Evaluate the model on the validation set
        model.eval()
        loss = 0
        for i, data in enumerate(validation_dataloader):
            data.to(device)
            # Make prediction and calculate loss
            y_pred = model(data.x, data.edge_index)
            y = data.y
            loss += criterion(y_pred, y).item()
        val_loss = loss / len(validation_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Trace model
            traced_model = torch.jit.trace(model, (data.x, data.edge_index))
            torch.jit.save(traced_model, f"{saving_path}/traced_model.pt")
            print("Model saved!")

        # Save training and validation losses to a CSV file
        training_losses.append(training_loss / len(training_dataloader))
        validation_losses.append(val_loss)
        with open(f"{saving_path}/losses.csv", mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "training_loss", "validation_loss"])
            for i in range(epoch + 1):
                writer.writerow([i + 1, training_losses[i], validation_losses[i]])

        # Plot training and validation losses and save the plot
        plt.clf()
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
        plt.savefig(f"{saving_path}/training_losses_plot.png")

if __name__=="__main__":
    # Define constants
    TRAINING_NAME = "gcn_1"
    BATCH_SIZE = 32
    EPOCHS = 15
    VARIABLES = ["geopotential_500", "u_500", "v_500"]
    NUM_VARIABLES = len(VARIABLES)
    HIDDEN_CHANNELS = 16
    LR = 0.01
    GAMMA = 0.7
    START_YEAR = 1950
    END_YEAR = 1970

    # Define the model
    model = gcn.GCN(
        in_channels=NUM_VARIABLES,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=NUM_VARIABLES,
    )

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LR,
        )
    
    criterion = torch.nn.MSELoss()

    # Create edges and points
    edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=24)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    # Load the training and validation datasets
    training_dataset = AtmosphericDataset(
        edge_index=edge_index,
        atmosphere_variables=VARIABLES,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )
    validation_dataset = AtmosphericDataset(
        edge_index=edge_index,
        atmosphere_variables=VARIABLES,
        start_year=2003,
        end_year=2006,
    )
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train(
        model=model,
        device=device,
        epochs=EPOCHS,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        training_name=TRAINING_NAME,
    )