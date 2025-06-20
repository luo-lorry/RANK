# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from torchcp.classification.predictor.base import BasePredictor
from torchcp.classification.utils import coverage_rate, average_size, CovGap, VioClasses, DiffViolation, SSCV, WSC, \
    singleton_hit_ratio
from torchcp.utils.common import calculate_conformal_value


class SplitPredictor(BasePredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.
    
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
    """

    def __init__(self, score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)

    #############################
    # The calibration process
    ############################
    def calibrate(self, cal_dataloader, alpha):

        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha):
        return calculate_conformal_value(scores, alpha)

    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        """
        Generate prediction sets for a batch of instances.

        Args:
            x_batch (torch.Tensor): A batch of instances.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        x_batch = self._model(x_batch.to(self._device)).float()
        x_batch = self._logits_transformation(x_batch).detach()
        sets = self.predict_with_logits(x_batch)
        return sets

    def predict_with_logits(self, logits, q_hat=None):
        """
        Generate prediction sets from logits.

        Args:
            logits (torch.Tensor): Model output before softmax.
            q_hat (torch.Tensor, optional): The conformal threshold. Default is None.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """

        scores = self.score_function(logits).to(self._device)
        if q_hat is None:
            if self.q_hat is None:
                raise ValueError("Ensure self.q_hat is not None. Please perform calibration first.")
            q_hat = self.q_hat

        S = self._generate_prediction_set(scores, q_hat)

        return S

    #############################
    # The evaluation process
    ############################

    def evaluate(self, val_dataloader, alpha: float = None, num_classes: int = None) -> Dict[str, float]:
        """
        Evaluate prediction sets on validation dataset.

        Args:
            val_dataloader (torch.utils.data.DataLoader): Dataloader for validation set.
            alpha (float, optional): Significance level. If None, uses the alpha from calibration.
            num_classes (int, optional): Number of classes. If None, inferred from prediction sets.

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        predictions_sets_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        features_list: List[torch.Tensor] = []

        # Evaluate in inference mode
        self._model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device and get predictions
                inputs = batch[0].to(self._device)
                labels = batch[1].to(self._device)

                # Get logits for difficulty-based metrics
                logits = self._model(inputs)

                # Get predictions as bool tensor (N x C)
                batch_predictions = self.predict(inputs)

                # Get features (flatten inputs as features for WSC)
                features = inputs.view(inputs.size(0), -1)

                # Accumulate predictions, labels, logits, and features
                predictions_sets_list.append(batch_predictions)
                labels_list.append(labels)
                logits_list.append(logits)
                features_list.append(features)

        # Concatenate all batches
        val_prediction_sets = torch.cat(predictions_sets_list, dim=0)
        val_labels = torch.cat(labels_list, dim=0)
        val_logits = torch.cat(logits_list, dim=0)
        val_features = torch.cat(features_list, dim=0)

        # Move to CPU for metrics calculation
        val_prediction_sets = val_prediction_sets.cpu()
        val_labels = val_labels.cpu()
        val_logits = val_logits.cpu()
        val_features = val_features.cpu()

        # Use alpha from calibration if not provided
        if alpha is None:
            alpha = getattr(self, '_alpha', 0.1)

        # Infer num_classes if not provided
        if num_classes is None:
            num_classes = val_prediction_sets.shape[1]

        # Compute all 10 evaluation metrics
        metrics = {}

        # 1. Coverage rate (marginal)
        metrics["coverage_rate"] = coverage_rate(val_prediction_sets, val_labels)

        # Coverage rate (macro) - variant of coverage_rate
        metrics["coverage_rate_macro"] = coverage_rate(val_prediction_sets, val_labels,
                                                       coverage_type="macro", num_classes=num_classes)

        # 2. Average size
        metrics["average_size"] = average_size(val_prediction_sets, val_labels)

        # 3. CovGap - Class-conditional coverage gap
        metrics["CovGap"] = CovGap(val_prediction_sets, val_labels, alpha, num_classes)

        # 4. VioClasses - Number of violated classes
        metrics["VioClasses"] = VioClasses(val_prediction_sets, val_labels, alpha, num_classes)

        # 5. DiffViolation - Difficulty-stratified coverage violation
        try:
            diff_violation, ccss_diff = DiffViolation(val_logits, val_prediction_sets, val_labels, alpha)
            metrics["DiffViolation"] = diff_violation
            metrics["DiffViolation_details"] = ccss_diff
        except Exception as e:
            print(f"DiffViolation calculation failed: {e}")
            metrics["DiffViolation"] = None

        # 6. SSCV - Size-stratified coverage violation
        metrics["SSCV"] = SSCV(val_prediction_sets, val_labels, alpha)

        # 7. WSC - Worst-Slice Coverage
        try:
            metrics["WSC"] = WSC(val_features, val_prediction_sets, val_labels,
                                 delta=0.1, M=100, test_fraction=0.75, random_state=2020, verbose=False)
        except Exception as e:
            print(f"WSC calculation failed: {e}")
            metrics["WSC"] = None

        # 8. Singleton hit ratio
        metrics["singleton_hit_ratio"] = singleton_hit_ratio(val_prediction_sets, val_labels)

        return metrics

    def evaluate_scores(self, prediction_sets, labels, alpha, logits=None, features=None, num_classes=None):
        """
        Evaluate prediction sets using all 10 metrics.

        Args:
            prediction_sets (torch.Tensor): Boolean tensor of shape (N, C)
            labels (torch.Tensor): True labels of shape (N,)
            alpha (float): Significance level
            logits (torch.Tensor, optional): Model logits for difficulty-based metrics
            features (torch.Tensor, optional): Features for WSC metric
            num_classes (int, optional): Number of classes

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Move tensors to CPU for metrics calculation
        prediction_sets = prediction_sets.cpu()
        labels = labels.cpu()
        if logits is not None:
            logits = logits.cpu()
        if features is not None:
            features = features.cpu()

        # Convert labels to long if not already
        if labels.dtype != torch.long:
            labels = labels.long()

        # Infer num_classes if not provided
        if num_classes is None:
            num_classes = prediction_sets.shape[1]

        # Calculate all 10 metrics
        metrics = {}

        # 1. Coverage rate (marginal)
        metrics['coverage_rate'] = coverage_rate(prediction_sets, labels)

        # Coverage rate (macro) - variant of coverage_rate
        metrics['coverage_rate_macro'] = coverage_rate(prediction_sets, labels,
                                                       coverage_type="macro", num_classes=num_classes)

        # 2. Average size
        metrics['average_size'] = average_size(prediction_sets, labels)

        # 3. CovGap - Class-conditional coverage gap
        metrics['CovGap'] = CovGap(prediction_sets, labels, alpha, num_classes)

        # 4. VioClasses - Number of violated classes
        metrics['VioClasses'] = VioClasses(prediction_sets, labels, alpha, num_classes)

        # 5. DiffViolation - Difficulty-stratified coverage violation (requires logits)
        if logits is not None:
            try:
                diff_violation, ccss_diff = DiffViolation(logits, prediction_sets, labels, alpha)
                metrics['DiffViolation'] = diff_violation
                metrics['DiffViolation_details'] = ccss_diff
            except Exception as e:
                print(f"DiffViolation calculation failed: {e}")
                metrics['DiffViolation'] = None
        else:
            metrics['DiffViolation'] = None
            print("DiffViolation requires logits parameter")

        # 6. SSCV - Size-stratified coverage violation
        metrics['SSCV'] = SSCV(prediction_sets, labels, alpha)

        # 7. WSC - Worst-Slice Coverage (requires features)
        if features is not None:
            try:
                metrics['WSC'] = WSC(features, prediction_sets, labels,
                                     delta=0.1, M=100, test_fraction=0.75, random_state=2020, verbose=False)
            except Exception as e:
                print(f"WSC calculation failed: {e}")
                metrics['WSC'] = None
        else:
            metrics['WSC'] = None
            print("WSC requires features parameter")

        # 8. Singleton hit ratio
        metrics['singleton_hit_ratio'] = singleton_hit_ratio(prediction_sets, labels)

        return metrics
