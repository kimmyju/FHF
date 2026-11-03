"""Verify balanced accuracy and perfect-prediction metric calculation."""
from fhf.metrics import report


def test_report_perfect():
    metrics=report([0,0,1,1],[0,0,1,1])
    assert metrics['balanced_accuracy_observed_classes']==1
    assert metrics['family_accuracy']==1
    assert metrics['within_family_accuracy_given_correct_family']==1
    assert len(metrics['confusion_matrix'])==18
