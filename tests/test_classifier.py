import numpy as np
import pytest

from decision_service.classifier import Classifier


class FakeModel:
    n_tokens = 1
    scores = np.asarray([[0.0, 1.0, 2.0]])

    def tokenize(self, text, add_bos=False, special=False):
        if text == b"A":
            return [1]
        if text == b"B":
            return [2]
        return [0]

    def eval(self, tokens):
        self.last_tokens = tokens


def test_choice_logits_are_normalized():
    classifier = Classifier(FakeModel())
    logits, log_probs, confidence = classifier.pick(
        "Payroll email", "Which category?", ["Legitimate", "Phishing"]
    )
    assert logits == {"Legitimate": 1.0, "Phishing": 2.0}
    assert sum(confidence.values()) == pytest.approx(1.0)
    assert confidence["Phishing"] == pytest.approx(0.7310586)
    assert np.exp(log_probs["Legitimate"]) == pytest.approx(confidence["Legitimate"])
