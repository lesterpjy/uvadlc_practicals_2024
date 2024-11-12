################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt
import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    predicted_classes = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_classes == targets)
    accuracy = correct_predictions / targets.shape[0]
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Test the best model on the test dataset
    predictions = []
    targets = []

    for x_test, y_test in data_loader:
        # print("test shapes", x_test.shape, y_test.shape)
        x_test = x_test.reshape(x_test.shape[0], -1)
        logits_test = model.forward(x_test)
        predictions.append(logits_test)
        targets.append(y_test)

    # Calculate test accuracy
    predictions = np.vstack(predictions)
    targets = np.hstack(targets)
    avg_accuracy = accuracy(predictions, targets)
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=True
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    train_loader = cifar10_loader["train"]
    val_loader = cifar10_loader["validation"]
    test_loader = cifar10_loader["test"]

    # TODO: Initialize model and loss module
    n_inputs = 32 * 32 * 3  # CIFAR-10 image dimensions (32x32 RGB)
    n_classes = 10  # CIFAR-10 has 10 classes
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model = None
    logging_dict = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.clear_cache()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []

        # Training loop
        for x_batch, y_batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"
        ):
            # Flatten input images for MLP
            x_batch = x_batch.reshape(x_batch.shape[0], -1)

            # Forward pass
            logits = model.forward(x_batch)
            loss = loss_module.forward(logits, y_batch)
            epoch_loss += loss

            # Backward pass
            dout = loss_module.backward(logits, y_batch)
            model.backward(dout)

            # SGD parameter update
            for layer in model.layers:
                if hasattr(layer, "params"):
                    layer.params["weight"] -= lr * layer.grads["weight"]
                    layer.params["bias"] -= lr * layer.grads["bias"]

            all_predictions.append(logits)
            all_targets.append(y_batch)

        # Calculate and log epoch training loss and accuracy
        all_predictions = np.vstack(all_predictions)
        all_targets = np.hstack(all_targets)
        train_accuracy = accuracy(all_predictions, all_targets)

        logging_dict["train_loss"].append(epoch_loss / len(train_loader))
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}"
        )

        # Validation loop
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        for x_val, y_val in val_loader:
            x_val = x_val.reshape(x_val.shape[0], -1)
            logits_val = model.forward(x_val)
            # y_val_one_hot = np.eye(n_classes)[y_val]
            val_loss += loss_module.forward(logits_val, y_val)
            val_predictions.append(logits_val)
            val_targets.append(y_val)

        # Calculate validation accuracy for the epoch
        val_predictions = np.vstack(val_predictions)
        val_targets = np.hstack(val_targets)
        val_accuracy = accuracy(val_predictions, val_targets)
        val_accuracies.append(val_accuracy)
        logging_dict["val_loss"].append(val_loss / len(val_loader))
        logging_dict["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

        # Check if the model is the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)

    model = best_model
    # TODO: Test best model
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # TODO: Add any information you might want to save for plotting
    logging_dict["best_val_accuracy"] = best_val_accuracy
    logging_dict["test_accuracy"] = test_accuracy

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR10 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy"
    )
    plt.axhline(y=test_accuracy, color="r", linestyle="--", label="Test Accuracy")
    plt.scatter(len(val_accuracies), test_accuracy, color="red")
    plt.text(
        len(val_accuracies),
        test_accuracy,
        f"{test_accuracy:.4f}",
        fontsize=12,
        color="red",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation and Test Accuracies")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracies_plot.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(logging_dict["train_loss"]) + 1),
        logging_dict["train_loss"],
        label="Training Loss",
    )
    plt.plot(
        range(1, len(logging_dict["val_loss"]) + 1),
        logging_dict["val_loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()
