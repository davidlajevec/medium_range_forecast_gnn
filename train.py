# Import necessary libraries
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

import matplotlib.pyplot as plt
import os
import csv
import json

def train(
    model,
    device,
    epochs,
    training_dataset,
    validation_dataset,
    batch_size,
    optimizer,
    scheduler,
    criterion,
    training_name,
    patience,
    input_graph_attributes=["x", "edge_index", "edge_attr"],
):
    """
    Trains a given PyTorch Geometric model on a training dataset and validates it on a validation dataset.

    Args:
        model (torch.nn.Module): The PyTorch Geometric model to train.
        device (torch.device): The device to train the model on.
        epochs (int): The number of epochs to train the model for.
        training_dataset (torch_geometric.data.Dataset): The training dataset.
        validation_dataset (torch_geometric.data.Dataset): The validation dataset.
        batch_size (int): The batch size to use for training and validation.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to use for training.
        criterion (torch.nn.Module): The loss function to use for training and validation.
        training_name (str): The name of the training run. Used to save the trained model and training statistics.
        patience (int): The number of epochs to wait for validation loss improvement before early stopping.
        input_graph_attributes (list, optional): The list of input graph attributes to use for the model. Defaults to ["x", "edge_index"].

    Returns:
        None
    """
    saving_path = "trained_models/" + training_name
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    print(f"Training on: {device}")
    print(f"Model name: {training_name}")
    model.to(device)
    best_val_loss = float("inf")
    early_stop_counter = 0

    # Initialize lists to store training and validation losses
    training_losses = []
    validation_losses = []

    training_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True
    )

    # Save the edge index and edge attributes to a file
    with open(f"{saving_path}/edge_index.txt", mode="w") as file:
        file.write(str(training_dataset[0].edge_index.tolist()))
    try:
        with open(f"{saving_path}/edge_attr.txt", mode="w") as file:
            file.write(str(training_dataset[0].edge_attr.tolist()))
    except:
        pass

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
            data_mapping = {
                attr: getattr(data, attr) for attr in input_graph_attributes
            }
            y_pred = model(**data_mapping)
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
            data_mapping = {
                attr: getattr(data, attr) for attr in input_graph_attributes
            }
            y_pred = model(**data_mapping)
            y = data.y
            loss += criterion(y_pred, y).item()
        val_loss = loss / len(validation_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save model
            torch.save(model, f"{saving_path}/model.pth")
            print("Model saved!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(
                    f"Validation loss did not improve for {patience} epochs. Stopping early."
                )
                break

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


if __name__ == "__main__":
    from datasets.atmospheric_dataset import AtmosphericDataset
    from models.LGCNLearnedWeightsLayered import GNN
    from utils.mesh_creation import create_neighbooring_edges
    # Define constants
    TRAINING_NAME = "train_test"
    BATCH_SIZE = 16
    EPOCHS = 1
    VARIABLES = ["geopotential_500", "u_500", "v_500"]
    STATIC_FIELDS = ["land_sea_mask", "surface_topography"]
    NUM_ATMOSPHERIC_VARIABLES = len(VARIABLES) 
    NUM_STATIC_FIELDS = len(STATIC_FIELDS)
    HIDDEN_CHANNELS = 32
    LR = 0.0005
    GAMMA = 0.9
    START_YEAR_TRAINING = 1950
    END_YEAR_TRAINING = 1950
    PATIENCE = 2

    # Define the model
    model = GNN(
        node_in_features=NUM_ATMOSPHERIC_VARIABLES+NUM_STATIC_FIELDS,
        edge_in_features=3,
        hidden_channels=HIDDEN_CHANNELS,
        out_features=NUM_ATMOSPHERIC_VARIABLES,
    )

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    criterion = torch.nn.MSELoss()

    # Create edges and points
    edge_index, edge_attrs, _, _ = create_neighbooring_edges(k=1)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    # Load the training and validation datasets
    training_dataset = AtmosphericDataset(
        edge_index=edge_index,
        atmosphere_variables=VARIABLES,
        static_fields=STATIC_FIELDS,
        start_year=START_YEAR_TRAINING,
        end_year=END_YEAR_TRAINING,
    )
    validation_dataset = AtmosphericDataset(
        edge_index=edge_index,
        atmosphere_variables=VARIABLES,
        static_fields=STATIC_FIELDS,
        start_year=2003,
        end_year=2006,
    )

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train(
        model=model,
        device=device,
        epochs=EPOCHS,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        training_name=TRAINING_NAME,
        patience=PATIENCE,
        input_graph_attributes=["x", "edge_index", "edge_attr"],
    )
