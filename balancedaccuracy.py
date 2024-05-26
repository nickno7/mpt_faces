import torch
import numpy as np

# NOTE: This will be the calculation of balanced accuracy for your classification task
# The balanced accuracy is defined as the average accuracy for each class.
# The accuracy for an indiviual class is the ratio between correctly classified example to all examples of that class.
# The code in train.py will instantiate one instance of this class.
# It will call the reset methos at the beginning of each epoch. Use this to reset your
# internal states. The update method will be called multiple times during an epoch, once for each batch of the training.
# You will receive the network predictions, a Tensor of Size (BATCHSIZExCLASSES) containing the logits (output without Softmax).
# You will also receive the groundtruth, an integer (long) Tensor with the respective class index per example.
# For each class, count how many examples were correctly classified and how many total examples exist.
# Then, in the getBACC method, calculate the balanced accuracy by first calculating each individual accuracy
# and then taking the average.


# Balanced Accuracy
class BalancedAccuracy:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

        # TODO: Setup internal variables
        # NOTE: It is good practive to all reset() from here to make sure everything is properly initialized

    def reset(self):
        self.matrix = np.zeros((self.nClasses, self.nClasses))

        # TODO: Reset internal states.
        # Called at the beginning of each epoch

    def update(self, predictions, groundtruth):
        predictions = np.argmax(predictions.detach().numpy(), axis=1)

        for true_label, pred_label in zip(groundtruth, predictions):
            self.matrix[true_label][pred_label] += 1

        # TODO: Implement the update of internal states
        # based on current network predictios and the groundtruth value.
        #
        # Predictions is a Tensor with logits (non-normalized activations)
        # It is a BATCH_SIZE x N_CLASSES float Tensor. The argmax for each samples
        # indicated the predicted class.
        #
        # Groundtruth is a BATCH_SIZE x 1 long Tensor. It contains the index of the
        # ground truth class.

    def getBACC(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class = np.diag(self.matrix) / self.matrix.sum(axis=1)

        return np.mean(per_class)

        # TODO: Calculcate and return balanced accuracy
        # based on current internal state


if __name__ == "__main__":
    predictions_epoch1 = torch.tensor(
        [
            [0.1, 0.8, 0.1],
            [0.3, 0.4, 0.3],
            [0.5, 0.3, 0.2],
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
        ]
    )
    groundtruth_epoch1 = torch.tensor([1, 0, 2, 0, 0, 0])

    predictions_epoch2 = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.9, 0.6],
            [0.7, 0.9, 0.1],
            [0.2, 0.9, 0.3],
            [0.1, 0.9, 0.6],
        ]
    )
    groundtruth_epoch2 = torch.tensor([2, 1, 0, 0, 0, 0])

    nClasses = 3
    bacc = BalancedAccuracy(nClasses)

    # Update after epoch 1
    bacc.update(predictions_epoch1, groundtruth_epoch1)
    print("Balanced Accuracy after epoch 1:", bacc.getBACC())

    # Update after epoch 2
    bacc.update(predictions_epoch2, groundtruth_epoch2)
    print("Balanced Accuracy after epoch 2:", bacc.getBACC())

    from sklearn.metrics import balanced_accuracy_score

    predictions_epoch1 = torch.tensor([np.argmax(p) for p in predictions_epoch1])
    predictions_epoch2 = torch.tensor([np.argmax(p) for p in predictions_epoch2])
    print(
        balanced_accuracy_score(
            np.hstack((groundtruth_epoch1, groundtruth_epoch2)),
            np.hstack((predictions_epoch1, predictions_epoch2)),
        )
    )
