import os

import pytest
import torch

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from trl.experimental.gkd import GKDTrainer  # noqa: E402

from liquid_finetune.training.utils.distillation import (  # noqa: E402
    IGNORE_INDEX,
    topk_generalized_jsd_loss,
)

BATCH, SEQ, VOCAB = 2, 6, 13
BETAS = [0.0, 0.25, 0.5, 1.0]


def _inputs(dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(BATCH, SEQ, VOCAB, generator=generator, dtype=dtype)
    teacher = 2 * torch.randn(BATCH, SEQ, VOCAB, generator=generator, dtype=dtype)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator)
    labels[0, :2] = IGNORE_INDEX
    labels[1, -1] = IGNORE_INDEX
    return student.requires_grad_(), teacher, labels


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def _closed_form(student_log, teacher_log, beta):
    """Per-position generalized JSD written out from its definition."""
    if beta == 0:
        return _kl(teacher_log, student_log)
    if beta == 1:
        return _kl(student_log, teacher_log)
    mixture_log = torch.log(beta * teacher_log.exp() + (1 - beta) * student_log.exp())
    return beta * _kl(teacher_log, mixture_log) + (1 - beta) * _kl(
        student_log, mixture_log
    )


def _restricted(log_probs, support, add_tail):
    """Distribution over a support set, with or without a tail bucket."""
    support_log = log_probs.gather(-1, support)
    if not add_tail:
        return torch.log_softmax(support_log, dim=-1)
    tail = torch.log1p(-support_log.exp().sum(-1, keepdim=True))
    return torch.cat([support_log, tail], dim=-1)


@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("beta", BETAS)
def test_full_support_without_tail_matches_trl_loss_and_gradient(beta, temperature):
    student, teacher, labels = _inputs()

    expected = GKDTrainer.generalized_jsd_loss(
        student, teacher, labels=labels, beta=beta, temperature=temperature
    )
    actual = topk_generalized_jsd_loss(
        student,
        teacher,
        labels,
        beta=beta,
        top_k=VOCAB,
        add_tail=False,
        temperature=temperature,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    (expected_grad,) = torch.autograd.grad(expected, student)
    (actual_grad,) = torch.autograd.grad(actual, student)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("beta", BETAS)
def test_full_support_with_tail_matches_trl_loss(beta):
    student, teacher, labels = _inputs()
    expected = GKDTrainer.generalized_jsd_loss(student, teacher, labels, beta=beta)
    actual = topk_generalized_jsd_loss(
        student, teacher, labels, beta=beta, top_k=VOCAB, add_tail=True
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("beta", BETAS)
def test_reductions_match_trl(beta):
    student, teacher, labels = _inputs()
    num_items = torch.tensor(17)
    common = {"beta": beta, "top_k": VOCAB, "add_tail": False}

    torch.testing.assert_close(
        topk_generalized_jsd_loss(
            student, teacher, labels, num_items_in_batch=num_items, **common
        ),
        GKDTrainer.generalized_jsd_loss(
            student, teacher, labels, beta=beta, num_items_in_batch=num_items
        ),
    )
    torch.testing.assert_close(
        topk_generalized_jsd_loss(student, teacher, labels, reduction="sum", **common),
        GKDTrainer.generalized_jsd_loss(
            student, teacher, labels, beta=beta, reduction="sum"
        ),
    )
    # Without labels, TRL's batchmean divides by the batch size.
    torch.testing.assert_close(
        topk_generalized_jsd_loss(student, teacher, None, **common),
        GKDTrainer.generalized_jsd_loss(student, teacher, None, beta=beta),
    )


@pytest.mark.parametrize("beta", BETAS)
def test_full_vocabulary_matches_closed_form(beta):
    student, teacher, labels = _inputs()
    mask = labels != IGNORE_INDEX
    expected = _closed_form(
        torch.log_softmax(student, -1), torch.log_softmax(teacher, -1), beta
    )[mask].mean()

    actual = topk_generalized_jsd_loss(
        student, teacher, labels, beta=beta, top_k=VOCAB, add_tail=False
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        GKDTrainer.generalized_jsd_loss(student, teacher, labels, beta=beta), expected
    )


@pytest.mark.parametrize("add_tail", [False, True])
@pytest.mark.parametrize("beta", BETAS)
def test_top_k_matches_restricted_closed_form(beta, add_tail):
    top_k = 4
    student, teacher, labels = _inputs(seed=1)
    student_log = torch.log_softmax(student, -1)
    teacher_log = torch.log_softmax(teacher, -1)

    expected = []
    for batch, position in (labels != IGNORE_INDEX).nonzero().tolist():
        teacher_top = set(teacher[batch, position].topk(top_k).indices.tolist())
        student_top = set(student[batch, position].topk(top_k).indices.tolist())
        if beta == 0:
            support = teacher_top
        elif beta == 1:
            support = student_top
        else:
            support = teacher_top | student_top
        support = torch.tensor(sorted(support))
        expected.append(
            _closed_form(
                _restricted(student_log[batch, position], support, add_tail),
                _restricted(teacher_log[batch, position], support, add_tail),
                beta,
            )
        )

    actual = topk_generalized_jsd_loss(
        student, teacher, labels, beta=beta, top_k=top_k, add_tail=add_tail
    )
    torch.testing.assert_close(actual, torch.stack(expected).mean())


@pytest.mark.parametrize("add_tail", [False, True])
@pytest.mark.parametrize("beta", BETAS)
def test_top_k_gradient_matches_finite_differences(beta, add_tail):
    student, teacher, labels = _inputs(seed=2)

    def loss(student_logits):
        return topk_generalized_jsd_loss(
            student_logits,
            teacher,
            labels,
            beta=beta,
            top_k=3,
            add_tail=add_tail,
            temperature=1.3,
        )

    assert torch.autograd.gradcheck(loss, (student,))


def test_tail_bucket_keeps_mass_outside_the_support():
    # With top_k=1 and no tail, both renormalized distributions are one-hot,
    # so the divergence vanishes; the tail keeps the signal.
    student, teacher, labels = _inputs(seed=3)
    without_tail = topk_generalized_jsd_loss(
        student, teacher, labels, beta=0.0, top_k=1, add_tail=False
    )
    with_tail = topk_generalized_jsd_loss(
        student, teacher, labels, beta=0.0, top_k=1, add_tail=True
    )
    assert without_tail.item() == pytest.approx(0.0, abs=1e-12)
    assert with_tail.item() > 0


def test_ignored_positions_do_not_contribute():
    student, teacher, labels = _inputs()
    loss = topk_generalized_jsd_loss(student, teacher, labels, beta=0.5, top_k=4)
    (grad,) = torch.autograd.grad(loss, student)

    ignored = labels == IGNORE_INDEX
    assert torch.all(grad[ignored] == 0)
    assert torch.any(grad[~ignored] != 0)

    perturbed_student = student.detach().clone()
    perturbed_teacher = teacher.clone()
    perturbed_student[ignored] += 5.0
    perturbed_teacher[ignored] -= 5.0
    torch.testing.assert_close(
        topk_generalized_jsd_loss(
            perturbed_student, perturbed_teacher, labels, beta=0.5, top_k=4
        ),
        loss.detach(),
    )


def test_vocab_size_ignores_padded_logit_columns():
    student, teacher, labels = _inputs()
    padded_teacher = torch.cat(
        [teacher, torch.full((BATCH, SEQ, 3), 50.0, dtype=teacher.dtype)], dim=-1
    )
    torch.testing.assert_close(
        topk_generalized_jsd_loss(
            student, padded_teacher, labels, beta=0.0, top_k=4, vocab_size=VOCAB
        ),
        topk_generalized_jsd_loss(student, teacher, labels, beta=0.0, top_k=4),
    )


@pytest.mark.parametrize("beta", BETAS)
def test_bfloat16_logits_track_float32_reference(beta):
    student, teacher, labels = _inputs(dtype=torch.float32)
    reference = topk_generalized_jsd_loss(student, teacher, labels, beta=beta, top_k=5)
    low_precision_student = student.detach().bfloat16().requires_grad_()
    low_precision = topk_generalized_jsd_loss(
        low_precision_student, teacher.bfloat16(), labels, beta=beta, top_k=5
    )
    assert low_precision.dtype == torch.float32
    torch.testing.assert_close(low_precision, reference.detach(), rtol=5e-2, atol=5e-3)
    low_precision.backward()
    assert low_precision_student.grad.dtype == torch.bfloat16
    assert torch.isfinite(low_precision_student.grad).all()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"top_k": 0}, "top_k"),
        ({"top_k": 2, "beta": 1.5}, "beta"),
        ({"top_k": 2, "reduction": "mean"}, "reduction"),
        ({"top_k": 2, "temperature": 0.0}, "temperature"),
    ],
)
def test_invalid_arguments_rejected(kwargs, message):
    student, teacher, labels = _inputs()
    with pytest.raises(ValueError, match=message):
        topk_generalized_jsd_loss(student, teacher, labels, **kwargs)
