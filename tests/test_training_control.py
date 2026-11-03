"""Test stopping semantics and LR schedule without loading ModernBERT weights."""
import pytest
import torch

from fhf.engine import ScoreEarlyStopper, make_scheduler


def test_early_stopping_patience_and_delta():
    stopper = ScoreEarlyStopper({"enabled": True, "patience": 3, "min_delta": 0.001})
    assert stopper.observe(0.26) == (True, False)
    assert stopper.observe(0.2605) == (False, False)
    assert stopper.observe(0.259) == (False, False)
    assert stopper.observe(0.2607) == (False, True)


def test_early_stopping_resets_patience_on_improvement():
    stopper = ScoreEarlyStopper({"enabled": True, "patience": 2, "min_delta": 0.001})
    assert stopper.observe(0.20) == (True, False)
    assert stopper.observe(0.20) == (False, False)
    assert stopper.observe(0.22) == (True, False)
    assert stopper.observe(0.22) == (False, False)
    assert stopper.observe(0.22) == (False, True)


def test_disabled_early_stopping_never_stops():
    stopper = ScoreEarlyStopper({"enabled": False, "patience": 1})
    stopper.observe(0.2)
    for _ in range(10):
        _, stop = stopper.observe(0.1)
        assert not stop


def test_cosine_scheduler_lowers_lr():
    param = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.AdamW([param], lr=1e-4)
    scheduler = make_scheduler(
        opt, {"epochs": 4,
              "scheduler": {"name": "cosine", "warmup_ratio": 0.0,
                            "min_lr_ratio": 0.05}}, updates_per_epoch=5)
    assert scheduler is not None
    initial = opt.param_groups[0]["lr"]
    lrs = []
    for _ in range(20):
        opt.zero_grad()
        (param * 1.0).backward()
        opt.step()
        scheduler.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert initial == pytest.approx(1e-4)
    assert lrs[-1] < lrs[0]
    assert lrs[-1] >= 0.0


def test_none_scheduler_is_disabled():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=1e-4)
    assert make_scheduler(opt, {"epochs": 10, "scheduler": {"name": "none"}}, 2) is None
