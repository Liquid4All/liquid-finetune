def alternating_reward(completions, **kwargs) -> list[float]:
    """Give each adjacent generation pair distinct rewards for E2E plumbing."""
    return [float(index % 2) for index in range(len(completions))]
