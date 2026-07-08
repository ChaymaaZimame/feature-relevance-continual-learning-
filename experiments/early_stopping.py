from __future__ import annotations

import logging
import torch

class EarlyStopper():
    '''
    Class handling Early Stopping by monitoring a metric

    Notes
    -----
    - It is assumed that the logger with the name 'EarlyStopper' has been configured before to use the correct formatting and handlers.
    '''
    def __init__(self, patience : int = 5, threshold : float = 0.0, mode : str = 'min'):
        '''
        Parameters
        ----------
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        threshold : float
            Minimum change in the monitored quantity to qualify as an improvement.
        mode : str
            One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing. In 'max' mode, training will stop when the quantity monitored has stopped increasing.

        Raises
        ------
        ValueError
            If `mode` is not one of {'min', 'max'}.
        '''
        self.patience = patience
        self.threshold = threshold
        self.mode = mode

        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', but got {mode}")

        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.logger = logging.getLogger('EarlyStopper')

    def step(self, score : (float | torch.Tensor)) -> None:
        '''
        Call this method after each epoch to update the early stopping criteria based on the current score.

        Parameters
        ----------
        score : float or torch.Tensor
            The current score to monitor.
        '''
        self.logger.debug(f'EarlyStopper step called.')
        if isinstance(score, torch.Tensor):
            score = score.item()

        if self.best_score is None:
            self.best_score = score
            self.early_stop = False

        if self.mode == 'min':
            if score < self.best_score - self.threshold:
                self.best_score = score
                self.counter = 0
            elif score > self.best_score + self.threshold:
                self.counter += 1

        elif self.mode == 'max':
            if score > self.best_score + self.threshold:
                self.best_score = score
                self.counter = 0
            elif score < self.best_score - self.threshold:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        self.logger.debug(
            f'Current best score: {self.best_score}. Epochs since last improvement: {self.counter}.'
        )

    
    def is_best(self) -> bool:
        '''
        Check if the current score is the best score so far.

        WARNING: If this function is called before step(), its return value is delayed by an epoch, marking the wrong epoch as the best one.

        Returns
        -------
        bool
            True if the current score is the best score so far, False otherwise.
        '''
        return self.counter == 0
    

    def state_dict(self) -> dict:
        '''
        Returns the state of the EarlyStopper as a dictionary. This can be used to reload the state later when resuming training from a checkoint.

        Returns
        -------
        dict
            A dictionary containing the state of the EarlyStopper.
        '''
        return {
            'patience': self.patience,
            'threshold': self.threshold,
            'mode': self.mode,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'counter': self.counter
        }
    

    @classmethod
    def from_state_dict(cls, state_dict : dict) -> EarlyStopper:
        '''
        Create an EarlyStopper instance from a state dictionary.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing the state of the EarlyStopper.

        Returns
        -------
        EarlyStopper
            An instance of EarlyStopper initialized with the state from the dictionary.
        '''
        instance = cls(
            patience = state_dict['patience'],
            threshold = state_dict['threshold'],
            mode = state_dict['mode']
        )
        instance.best_score = state_dict['best_score']
        instance.early_stop = state_dict['early_stop']
        instance.counter = state_dict['counter']
        return instance