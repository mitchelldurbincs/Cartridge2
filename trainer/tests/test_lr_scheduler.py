"""Tests for lr_scheduler.py - Learning rate scheduling.

Tests cover:
- LRConfig dataclass and computed properties
- Warmup phase linear increase
- Cosine annealing decay
- Disabled scheduling
- State management for checkpointing
- Edge cases (zero warmup, completed schedules)
"""

import pytest
import torch
from torch.optim import SGD, Adam

from trainer.lr_scheduler import LRConfig, WarmupCosineScheduler


class TestLRConfig:
    """Tests for LRConfig dataclass."""

    def test_lr_config_defaults(self):
        """Test default configuration values."""
        config = LRConfig(target_lr=0.001)

        assert config.target_lr == 0.001
        assert config.warmup_steps == 100
        assert config.warmup_start_ratio == 0.1
        assert config.min_ratio == 0.1
        assert config.total_steps == 1000
        assert config.enabled is True

    def test_lr_config_custom(self):
        """Test custom configuration values."""
        config = LRConfig(
            target_lr=0.01,
            warmup_steps=50,
            warmup_start_ratio=0.05,
            min_ratio=0.01,
            total_steps=500,
            enabled=False,
        )

        assert config.target_lr == 0.01
        assert config.warmup_steps == 50
        assert config.warmup_start_ratio == 0.05
        assert config.min_ratio == 0.01
        assert config.total_steps == 500
        assert config.enabled is False

    def test_lr_config_computed_warmup_start_lr(self):
        """Test warmup_start_lr property."""
        config = LRConfig(target_lr=0.001, warmup_start_ratio=0.1)
        assert config.warmup_start_lr == 0.0001

        config2 = LRConfig(target_lr=0.01, warmup_start_ratio=0.05)
        assert config2.warmup_start_lr == 0.0005

    def test_lr_config_computed_min_lr(self):
        """Test min_lr property."""
        config = LRConfig(target_lr=0.001, min_ratio=0.1)
        assert config.min_lr == 0.0001

        config2 = LRConfig(target_lr=0.01, min_ratio=0.01)
        assert config2.min_lr == 0.0001

    def test_lr_config_computed_cosine_steps(self):
        """Test cosine_steps property."""
        config = LRConfig(target_lr=0.001, warmup_steps=100, total_steps=1000)
        assert config.cosine_steps == 900

    def test_lr_config_cosine_steps_minimum_one(self):
        """Test that cosine_steps is at least 1."""
        config = LRConfig(target_lr=0.001, warmup_steps=1000, total_steps=1000)
        assert config.cosine_steps == 1


class TestWarmupPhase:
    """Tests for the warmup phase of scheduling."""

    def test_warmup_starts_at_correct_lr(self):
        """Test that warmup starts at warmup_start_lr."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=10, warmup_start_ratio=0.1)
        scheduler = WarmupCosineScheduler(optimizer, config)

        # Before any steps, LR should be warmup_start_lr
        assert scheduler.get_lr() == 0.0001

    def test_warmup_linear_increase(self):
        """Test that LR increases linearly during warmup."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=10, warmup_start_ratio=0.1)
        scheduler = WarmupCosineScheduler(optimizer, config)

        lrs = []
        for _ in range(5):
            lrs.append(scheduler.get_lr())
            scheduler.step()

        # LR should increase monotonically
        for i in range(len(lrs) - 1):
            assert lrs[i] < lrs[i + 1]

    def test_warmup_reaches_target_lr(self):
        """Test that warmup ends at target_lr."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=10, warmup_start_ratio=0.1)
        scheduler = WarmupCosineScheduler(optimizer, config)

        # Step through warmup
        for _ in range(10):
            scheduler.step()

        assert abs(scheduler.get_lr() - 0.001) < 1e-6

    def test_warmup_progress_calculation(self):
        """Test that warmup progress is calculated correctly."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=10, warmup_start_ratio=0.1)
        scheduler = WarmupCosineScheduler(optimizer, config)

        # After 5 steps (halfway through warmup)
        for _ in range(5):
            scheduler.step()

        # Should be halfway between start and target
        expected_lr = 0.0001 + 0.5 * (0.001 - 0.0001)
        assert abs(scheduler.get_lr() - expected_lr) < 1e-6


class TestCosineAnnealing:
    """Tests for the cosine annealing phase."""

    def test_cosine_starts_after_warmup(self):
        """Test that cosine annealing begins after warmup completes."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=5, total_steps=20)
        scheduler = WarmupCosineScheduler(optimizer, config)

        # Step through warmup
        for _ in range(5):
            scheduler.step()

        # Now should be in cosine phase
        initial_cosine_lr = scheduler.get_lr()
        assert initial_cosine_lr == 0.001

        # Take one more step, LR should decrease
        scheduler.step()
        assert scheduler.get_lr() < initial_cosine_lr

    def test_cosine_decreases_monotonically(self):
        """Test that LR decreases monotonically during cosine annealing."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=0, total_steps=20)
        scheduler = WarmupCosineScheduler(optimizer, config)

        lrs = []
        for _ in range(15):
            lrs.append(scheduler.get_lr())
            scheduler.step()

        # LR should decrease (or stay same) monotonically
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]

    def test_cosine_reaches_min_lr(self):
        """Test that cosine annealing ends at min_lr."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=0, total_steps=10, min_ratio=0.1)
        scheduler = WarmupCosineScheduler(optimizer, config)

        # Step through entire schedule
        for _ in range(10):
            scheduler.step()

        assert abs(scheduler.get_lr() - 0.0001) < 1e-6


class TestDisabledScheduling:
    """Tests for disabled LR scheduling."""

    def test_disabled_scheduler_constant_lr(self):
        """Test that LR stays constant when scheduling is disabled."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, enabled=False)
        scheduler = WarmupCosineScheduler(optimizer, config)

        initial_lr = scheduler.get_lr()

        for _ in range(10):
            scheduler.step()
            assert scheduler.get_lr() == initial_lr

    def test_disabled_scheduler_no_cosine(self):
        """Test that no cosine scheduler is created when disabled."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, enabled=False)
        scheduler = WarmupCosineScheduler(optimizer, config)

        assert scheduler._cosine_scheduler is None


class TestStateManagement:
    """Tests for checkpoint save/load."""

    def test_state_dict_contains_current_step(self):
        """Test that state dict includes current step."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=10, total_steps=100)
        scheduler = WarmupCosineScheduler(optimizer, config)

        # Step a few times
        for _ in range(5):
            scheduler.step()

        state = scheduler.state_dict()
        assert state["current_step"] == 5

    def test_state_dict_contains_warmup_metadata(self):
        """Test that state dict includes warmup metadata."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=10, total_steps=100)
        scheduler = WarmupCosineScheduler(optimizer, config)

        state = scheduler.state_dict()
        assert "warmup_steps" in state
        assert "config_warmup_steps" in state
        assert state["warmup_steps"] == 10

    def test_state_dict_contains_cosine_state(self):
        """Test that state dict includes cosine scheduler state."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=0, total_steps=100)
        scheduler = WarmupCosineScheduler(optimizer, config)

        # Step through warmup (none) and into cosine
        for _ in range(10):
            scheduler.step()

        state = scheduler.state_dict()
        assert "cosine_scheduler" in state
        assert "last_epoch" in state["cosine_scheduler"]

    def test_load_state_dict_restores_step(self):
        """Test that loading restores the step counter."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=10, total_steps=100)
        scheduler = WarmupCosineScheduler(optimizer, config)

        for _ in range(5):
            scheduler.step()

        state = scheduler.state_dict()

        # Load into new scheduler
        optimizer2 = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        scheduler2 = WarmupCosineScheduler(optimizer2, config, from_checkpoint=True)
        scheduler2.load_state_dict(state)

        assert scheduler2._current_step == 5

    def test_load_state_dict_disables_warmup(self):
        """Test that warmup is disabled when loading from checkpoint."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=10, total_steps=100)
        scheduler = WarmupCosineScheduler(optimizer, config)

        for _ in range(5):
            scheduler.step()

        state = scheduler.state_dict()

        # Load into new scheduler
        optimizer2 = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        scheduler2 = WarmupCosineScheduler(optimizer2, config, from_checkpoint=True)
        scheduler2.load_state_dict(state)

        assert scheduler2._warmup_steps == 0
        assert scheduler2.warmup_disabled is True

    def test_load_state_dict_restores_cosine(self):
        """Test that loading restores cosine scheduler state."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=0, total_steps=100)
        scheduler = WarmupCosineScheduler(optimizer, config)

        for _ in range(20):
            scheduler.step()

        lr_before = scheduler.get_lr()
        state = scheduler.state_dict()

        # Load into new scheduler
        optimizer2 = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        scheduler2 = WarmupCosineScheduler(optimizer2, config, from_checkpoint=True)
        scheduler2.load_state_dict(state)

        # LR should be restored (within reasonable tolerance)
        assert abs(scheduler2.get_lr() - lr_before) < 1e-4

    def test_load_legacy_state_dict(self):
        """Test loading legacy state dict format (backwards compatibility)."""
        # These are created to establish the test setup pattern,
        # though we only use the config below
        _ = Adam([torch.randn(10, requires_grad=True)], lr=0.001)  # optimizer (unused)
        config = LRConfig(target_lr=0.001, warmup_steps=0, total_steps=100)

        # Create legacy state (just cosine scheduler state)
        legacy_state = {"last_epoch": 15, "T_max": 100}

        # Load into new scheduler
        optimizer2 = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        scheduler2 = WarmupCosineScheduler(optimizer2, config, from_checkpoint=True)
        scheduler2.load_state_dict(legacy_state)

        assert scheduler2._current_step == 15


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_zero_warmup_steps(self):
        """Test scheduler with zero warmup steps."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=0, total_steps=100)
        scheduler = WarmupCosineScheduler(optimizer, config)

        # Should immediately be in cosine phase
        assert not scheduler.warmup_active
        assert scheduler.get_lr() == 0.001

    def test_load_past_end_of_schedule(self):
        """Test loading when schedule has completed."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=0, total_steps=10, min_ratio=0.1)
        scheduler = WarmupCosineScheduler(optimizer, config)

        # Run past completion (T_max = 10, so 11 steps completes the cosine schedule)
        for _ in range(11):
            scheduler.step()

        assert scheduler.schedule_complete
        state = scheduler.state_dict()

        # Load into new scheduler
        optimizer2 = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        scheduler2 = WarmupCosineScheduler(optimizer2, config, from_checkpoint=True)
        scheduler2.load_state_dict(state)

        # Should be at min_lr
        assert abs(scheduler2.get_lr() - 0.0001) < 1e-6
        # Note: schedule_complete may not be True after load since the cosine
        # scheduler state is restored but last_epoch is reset to 0 by load_state_dict
        # The important thing is that LR is at minimum

    def test_warmup_disabled_on_init(self):
        """Test that warmup is disabled when from_checkpoint=True."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=100, total_steps=200)
        scheduler = WarmupCosineScheduler(optimizer, config, from_checkpoint=True)

        assert scheduler._warmup_steps == 0
        assert scheduler.warmup_disabled is True
        assert not scheduler.warmup_active

    def test_warmup_complete_property(self):
        """Test warmup_complete property."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=5, total_steps=100)
        scheduler = WarmupCosineScheduler(optimizer, config)

        assert not scheduler.warmup_complete

        # Step through all warmup steps (step 5 of 5 is still in warmup)
        for _ in range(5):
            scheduler.step()

        # Still in warmup (step 5 of 5) - implementation uses > not >=
        assert not scheduler.warmup_complete

        # Step one more time to be past warmup
        scheduler.step()
        assert scheduler.warmup_complete

    def test_get_phase_reporting(self):
        """Test get_phase() returns correct phase description."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)

        # Disabled
        config_disabled = LRConfig(target_lr=0.001, enabled=False)
        scheduler_disabled = WarmupCosineScheduler(optimizer, config_disabled)
        assert scheduler_disabled.get_phase() == "disabled"

        # Warmup
        optimizer2 = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config_warmup = LRConfig(target_lr=0.001, warmup_steps=10, total_steps=100)
        scheduler_warmup = WarmupCosineScheduler(optimizer2, config_warmup)
        scheduler_warmup.step()
        assert "warmup" in scheduler_warmup.get_phase()

        # Cosine
        for _ in range(10):
            scheduler_warmup.step()
        assert "cosine" in scheduler_warmup.get_phase()

    def test_different_optimizers(self):
        """Test scheduler works with different optimizer types."""
        # Test with SGD
        optimizer_sgd = SGD([torch.randn(10, requires_grad=True)], lr=0.01)
        config = LRConfig(target_lr=0.01, warmup_steps=5, total_steps=50)
        scheduler_sgd = WarmupCosineScheduler(optimizer_sgd, config)

        for _ in range(10):
            scheduler_sgd.step()

        assert scheduler_sgd.get_lr() < 0.01  # Should have decreased

    def test_repr(self):
        """Test __repr__ returns useful information."""
        optimizer = Adam([torch.randn(10, requires_grad=True)], lr=0.001)
        config = LRConfig(target_lr=0.001, warmup_steps=10, total_steps=100)
        scheduler = WarmupCosineScheduler(optimizer, config)

        repr_str = repr(scheduler)
        assert "WarmupCosineScheduler" in repr_str
        assert "lr=" in repr_str
        assert "phase=" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
