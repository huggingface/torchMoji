# -*- coding: utf-8 -*-
""" Class average finetuning functions. Before using any of these finetuning
    functions, ensure that the model is set up with nb_classes=2.
"""
from __future__ import print_function

import uuid
from time import sleep
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchmoji.global_variables import (
    FINETUNING_METHODS,
    WEIGHTS_DIR)
from torchmoji.finetuning import (
    freeze_layers,
    get_data_loader,
    fit_model,
    train_by_chain_thaw,
    find_f1_threshold)

def relabel(y, current_label_nr, nb_classes):
    """ Makes a binary classification for a specific class in a
        multi-class dataset.

    # Arguments:
        y: Outputs to be relabelled.
        current_label_nr: Current label number.
        nb_classes: Total number of classes.

    # Returns:
        Relabelled outputs of a given multi-class dataset into a binary
        classification dataset.
    """

    # Handling binary classification
    if nb_classes == 2 and len(y.shape) == 1:
        return y

    y_new = np.zeros(len(y))
    y_cut = y[:, current_label_nr]
    label_pos = np.where(y_cut == 1)[0]
    y_new[label_pos] = 1
    return y_new


def class_avg_finetune(model, texts, labels, nb_classes, batch_size,
                       method, epoch_size=5000, nb_epochs=1000, embed_l2=1E-6,
                       verbose=True):
    """ Compiles and finetunes the given model.

    # Arguments:
        model: Model to be finetuned
        texts: List of three lists, containing tokenized inputs for training,
            validation and testing (in that order).
        labels: List of three lists, containing labels for training,
            validation and testing (in that order).
        nb_classes: Number of classes in the dataset.
        batch_size: Batch size.
        method: Finetuning method to be used. For available methods, see
            FINETUNING_METHODS in global_variables.py. Note that the model
            should be defined accordingly (see docstring for torchmoji_transfer())
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs. Doesn't matter much as early stopping is used.
        embed_l2: L2 regularization for the embedding layer.
        verbose: Verbosity flag.

    # Returns:
        Model after finetuning,
        score after finetuning using the class average F1 metric.
    """

    if method not in FINETUNING_METHODS:
        raise ValueError('ERROR (class_avg_tune_trainable): '
                         'Invalid method parameter. '
                         'Available options: {}'.format(FINETUNING_METHODS))

    (X_train, y_train) = (texts[0], labels[0])
    (X_val, y_val) = (texts[1], labels[1])
    (X_test, y_test) = (texts[2], labels[2])

    checkpoint_path = '{}/torchmoji-checkpoint-{}.bin' \
                      .format(WEIGHTS_DIR, str(uuid.uuid4()))

    f1_init_path = '{}/torchmoji-f1-init-{}.bin' \
                   .format(WEIGHTS_DIR, str(uuid.uuid4()))

    if method in ['last', 'new']:
        lr = 0.001
    elif method in ['full', 'chain-thaw']:
        lr = 0.0001

    loss_op = nn.BCEWithLogitsLoss()

    # Freeze layers if using last
    if method == 'last':
        model = freeze_layers(model, unfrozen_keyword='output_layer')

    # Define optimizer, for chain-thaw we define it later (after freezing)
    if method == 'last':
        adam = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    elif method in ['full', 'new']:
        # Add L2 regulation on embeddings only
        special_params = [id(p) for p in model.embed.parameters()]
        base_params = [p for p in model.parameters() if id(p) not in special_params and p.requires_grad]
        embed_parameters = [p for p in model.parameters() if id(p) in special_params and p.requires_grad]
        adam = optim.Adam([
            {'params': base_params},
            {'params': embed_parameters, 'weight_decay': embed_l2},
            ], lr=lr)

    # Training
    if verbose:
        print('Method:  {}'.format(method))
        print('Classes: {}'.format(nb_classes))

    if method == 'chain-thaw':
        result = class_avg_chainthaw(model, nb_classes=nb_classes,
                                     loss_op=loss_op,
                                     train=(X_train, y_train),
                                     val=(X_val, y_val),
                                     test=(X_test, y_test),
                                     batch_size=batch_size,
                                     epoch_size=epoch_size,
                                     nb_epochs=nb_epochs,
                                     checkpoint_weight_path=checkpoint_path,
                                     f1_init_weight_path=f1_init_path,
                                     verbose=verbose)
    else:
        result = class_avg_tune_trainable(model, nb_classes=nb_classes,
                                          loss_op=loss_op,
                                          optim_op=adam,
                                          train=(X_train, y_train),
                                          val=(X_val, y_val),
                                          test=(X_test, y_test),
                                          epoch_size=epoch_size,
                                          nb_epochs=nb_epochs,
                                          batch_size=batch_size,
                                          init_weight_path=f1_init_path,
                                          checkpoint_weight_path=checkpoint_path,
                                          verbose=verbose)
    return model, result


def prepare_labels(y_train, y_val, y_test, iter_i, nb_classes):
    # Relabel into binary classification
    y_train_new = relabel(y_train, iter_i, nb_classes)
    y_val_new = relabel(y_val, iter_i, nb_classes)
    y_test_new = relabel(y_test, iter_i, nb_classes)
    return y_train_new, y_val_new, y_test_new

def prepare_generators(X_train, y_train_new, X_val, y_val_new, batch_size, epoch_size):
    # Create sample generators
    # Make a fixed validation set to avoid fluctuations in validation
    train_gen = get_data_loader(X_train, y_train_new, batch_size,
                                extended_batch_sampler=True)
    val_gen = get_data_loader(X_val, y_val_new, epoch_size,
                              extended_batch_sampler=True)
    X_val_resamp, y_val_resamp = next(iter(val_gen))
    return train_gen, X_val_resamp, y_val_resamp


def class_avg_tune_trainable(model, nb_classes, loss_op, optim_op, train, val, test,
                             epoch_size, nb_epochs, batch_size,
                             init_weight_path, checkpoint_weight_path, patience=5,
                             verbose=True):
    """ Finetunes the given model using the F1 measure.

    # Arguments:
        model: Model to be finetuned.
        nb_classes: Number of classes in the given dataset.
        train: Training data, given as a tuple of (inputs, outputs)
        val: Validation data, given as a tuple of (inputs, outputs)
        test: Testing data, given as a tuple of (inputs, outputs)
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        batch_size: Batch size.
        init_weight_path: Filepath where weights will be initially saved before
            training each class. This file will be rewritten by the function.
        checkpoint_weight_path: Filepath where weights will be checkpointed to
            during training. This file will be rewritten by the function.
        verbose: Verbosity flag.

    # Returns:
        F1 score of the trained model
    """
    total_f1 = 0
    nb_iter = nb_classes if nb_classes > 2 else 1

    # Unpack args
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    # Save and reload initial weights after running for
    # each class to avoid learning across classes
    torch.save(model.state_dict(), init_weight_path)
    for i in range(nb_iter):
        if verbose:
            print('Iteration number {}/{}'.format(i+1, nb_iter))

        model.load_state_dict(torch.load(init_weight_path))
        y_train_new, y_val_new, y_test_new = prepare_labels(y_train, y_val,
                                                            y_test, i, nb_classes)
        train_gen, X_val_resamp, y_val_resamp = \
            prepare_generators(X_train, y_train_new, X_val, y_val_new,
                               batch_size, epoch_size)

        if verbose:
            print("Training..")
        fit_model(model, loss_op, optim_op, train_gen, [(X_val_resamp, y_val_resamp)],
                  nb_epochs, checkpoint_weight_path, patience, verbose=0)

        # Reload the best weights found to avoid overfitting
        # Wait a bit to allow proper closing of weights file
        sleep(1)
        model.load_state_dict(torch.load(checkpoint_weight_path))

        # Evaluate
        y_pred_val = model(X_val).cpu().numpy()
        y_pred_test = model(X_test).cpu().numpy()

        f1_test, best_t = find_f1_threshold(y_val_new, y_pred_val,
                                            y_test_new, y_pred_test)
        if verbose:
            print('f1_test: {}'.format(f1_test))
            print('best_t:  {}'.format(best_t))
        total_f1 += f1_test

    return total_f1 / nb_iter


def class_avg_chainthaw(model, nb_classes, loss_op, train, val, test, batch_size,
                        epoch_size, nb_epochs, checkpoint_weight_path,
                        f1_init_weight_path, patience=5,
                        initial_lr=0.001, next_lr=0.0001, verbose=True):
    """ Finetunes given model using chain-thaw and evaluates using F1.
        For a dataset with multiple classes, the model is trained once for
        each class, relabeling those classes into a binary classification task.
        The result is an average of all F1 scores for each class.

    # Arguments:
        model: Model to be finetuned.
        nb_classes: Number of classes in the given dataset.
        train: Training data, given as a tuple of (inputs, outputs)
        val: Validation data, given as a tuple of (inputs, outputs)
        test: Testing data, given as a tuple of (inputs, outputs)
        batch_size: Batch size.
        loss: Loss function to be used during training.
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        checkpoint_weight_path: Filepath where weights will be checkpointed to
            during training. This file will be rewritten by the function.
        f1_init_weight_path: Filepath where weights will be saved to and
            reloaded from before training each class. This ensures that
            each class is trained independently. This file will be rewritten.
        initial_lr: Initial learning rate. Will only be used for the first
            training step (i.e. the softmax layer)
        next_lr: Learning rate for every subsequent step.
        seed: Random number generator seed.
        verbose: Verbosity flag.

    # Returns:
        Averaged F1 score.
    """

    # Unpack args
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    total_f1 = 0
    nb_iter = nb_classes if nb_classes > 2 else 1

    torch.save(model.state_dict(), f1_init_weight_path)

    for i in range(nb_iter):
        if verbose:
            print('Iteration number {}/{}'.format(i+1, nb_iter))

        model.load_state_dict(torch.load(f1_init_weight_path))
        y_train_new, y_val_new, y_test_new = prepare_labels(y_train, y_val,
                                                            y_test, i, nb_classes)
        train_gen, X_val_resamp, y_val_resamp = \
                prepare_generators(X_train, y_train_new, X_val, y_val_new,
                                   batch_size, epoch_size)

        if verbose:
            print("Training..")

        # Train using chain-thaw
        train_by_chain_thaw(model=model, train_gen=train_gen,
                            val_gen=[(X_val_resamp, y_val_resamp)],
                            loss_op=loss_op, patience=patience,
                            nb_epochs=nb_epochs,
                            checkpoint_path=checkpoint_weight_path,
                            initial_lr=initial_lr, next_lr=next_lr,
                            verbose=verbose)

        # Evaluate
        y_pred_val = model(X_val).cpu().numpy()
        y_pred_test = model(X_test).cpu().numpy()

        f1_test, best_t = find_f1_threshold(y_val_new, y_pred_val,
                                            y_test_new, y_pred_test)

        if verbose:
            print('f1_test: {}'.format(f1_test))
            print('best_t:  {}'.format(best_t))
        total_f1 += f1_test

    return total_f1 / nb_iter
