"""Choose one swarm agent using verified desktop task outcomes."""


def select_desktop_operator(scores, agent_ids):
    agent_ids = [int(agent_id) for agent_id in agent_ids]
    if not agent_ids:
        return None, "No agents are available for desktop operation."

    proven = []
    for agent_id in agent_ids:
        score = scores.get(agent_id, {})
        successes = int(score.get("verified_successes", 0))
        attempts = int(score.get("attempts", 0))
        if successes > 0 and attempts >= successes:
            proven.append((agent_id, successes, attempts))

    if proven:
        agent_id, successes, attempts = max(
            proven,
            key=lambda row: (
                row[1] / row[2],
                row[1],
                -row[2],
                -row[0],
            ),
        )
        return agent_id, (
            f"Selected from verified desktop history: {successes}/{attempts} "
            f"successful tasks ({successes / attempts:.0%})."
        )

    agent_id = max(agent_ids)
    return agent_id, (
        "Cold start: no agent has verified desktop history; assigning the "
        "completion-check role provisionally."
    )
