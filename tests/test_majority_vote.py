"""
Tests for MajorityVoteClassifier.

Covers:
- Basic training and prediction with class label voting.
- Probability-based voting and probability normalization.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from incunabula.majority_vote import MajorityVoteClassifier


def test_majority_vote_classlabel():
    """
    Check that the ensemble predicts class labels correctly
    when using majority vote on class labels.
    """
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    clf1 = LogisticRegression(max_iter=200, random_state=1)
    clf2 = DecisionTreeClassifier(max_depth=3, random_state=1)
    clf3 = KNeighborsClassifier(n_neighbors=5)

    mv_clf = MajorityVoteClassifier(
        classifiers=[clf1, clf2, clf3],
        vote='classlabel'
    ).fit(X_train, y_train)

    preds = mv_clf.predict(X_test)

    # Same number of predictions as test samples
    assert preds.shape[0] == X_test.shape[0]

    # Predictions should only contain valid classes
    assert set(np.unique(preds)).issubset(set(y))


def test_majority_vote_probability():
    """
    Check that the ensemble outputs valid probability distributions
    and predictions when using probability-based voting.
    """
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    clf1 = LogisticRegression(max_iter=200, random_state=1)
    clf2 = DecisionTreeClassifier(max_depth=3, random_state=1)
    clf3 = KNeighborsClassifier(n_neighbors=5)

    mv_clf = MajorityVoteClassifier(
        classifiers=[clf1, clf2, clf3],
        vote='probability'
    ).fit(X_train, y_train)

    probas = mv_clf.predict_proba(X_test)

    # Shape should match: (n_samples, n_classes)
    assert probas.shape == (X_test.shape[0], len(np.unique(y)))

    # Probabilities per row should sum to ~1
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)
