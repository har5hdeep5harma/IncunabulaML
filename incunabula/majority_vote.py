"""True horror isn’t suffering.
True horror is success revealing that nothing was ever yours, 
and no one is coming to explain why.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.estimator_checks import _name_estimators


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple ensemble meta-classifier that combines multiple classifiers
    using majority voting or probability averaging.

    Parameters
    ----------
    classifiers : list
        Base classifiers to be combined in the ensemble.

    vote : {'classlabel', 'probability'}, default='classlabel'
        Voting strategy:
        - 'classlabel': prediction based on majority class votes.
        - 'probability': prediction based on averaged probabilities.

    weights : list of float or None, default=None
        Importance weights for classifiers. If None, all are treated equally.
    """

    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """
        Train each classifier on the provided dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : MajorityVoteClassifier
            Fitted ensemble model.

        Notes
        -----
        - Uses LabelEncoder internally so that class labels are mapped to 0..n.
        - Each base classifier is cloned before fitting, so originals are untouched.
        - Raises ValueError if `vote` is invalid or if weights don't match classifiers.
        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'")

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match "
                f"classifiers length ({len(self.classifiers)})"
            )

        # Encode labels to 0..n_classes-1
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_

        # Clone and fit each classifier
        self.fitted_classifiers_ = [
            clone(clf).fit(X, self.label_encoder_.transform(y))
            for clf in self.classifiers
        ]
        return self

    def predict(self, X):
        """
        Predict class labels for given input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels.

        Notes
        -----
        - If `vote='probability'`, chooses class with highest averaged probability.
        - If `vote='classlabel'`, uses majority rule (with optional weights).
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            preds = np.asarray([clf.predict(X)
                                for clf in self.fitted_classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda row: np.argmax(np.bincount(row, weights=self.weights)),
                axis=1, arr=preds
            )
        return self.label_encoder_.inverse_transform(maj_vote)

    def predict_proba(self, X):
        """
        Predict class membership probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        avg_proba : ndarray of shape (n_samples, n_classes)
            Weighted average probabilities across classifiers.
        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.fitted_classifiers_])
        return np.average(probas, axis=0, weights=self.weights)

    def get_params(self, deep=True):
        """
        Return classifier parameters for scikit-learn compatibility.

        Parameters
        ----------
        deep : bool, default=True
            If True, include parameters of contained classifiers.

        Returns
        -------
        params : dict
            Mapping of parameter names to values.
        """
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f"{name}__{key}"] = value
            return out
