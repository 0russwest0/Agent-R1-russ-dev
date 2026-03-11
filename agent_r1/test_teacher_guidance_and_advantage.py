import torch

from agent_r1.core_algos import compute_reinforce_plus_plus_baseline_outcome_advantage
from agent_r1.env.base import Action
from agent_r1.env.envs.teacher_guidance import TeacherGuidanceEnv
from agent_r1.metric_utils import compute_cumulative_guidance_success


def test_rfpp_baseline_aggregates_over_full_trajectory():
    token_level_rewards = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    # Two rollouts for the same uid, each rollout has 3 steps.
    trajectory_uids = ["traj-a", "traj-a", "traj-a", "traj-b", "traj-b", "traj-b"]
    uid = ["sample-0"] * 6

    advantages, returns = compute_reinforce_plus_plus_baseline_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=uid,
        trajectory_uids=trajectory_uids,
    )

    # Each step inside the same trajectory should receive the same broadcasted score.
    assert torch.allclose(advantages[0], advantages[1])
    assert torch.allclose(advantages[1], advantages[2])
    assert torch.allclose(advantages[3], advantages[4])
    assert torch.allclose(advantages[4], advantages[5])
    assert torch.allclose(advantages, returns)


def test_cumulative_guidance_success_uses_leq_n_rounds_semantics():
    success_by_round = compute_cumulative_guidance_success(
        trajectory_uids=["traj-a", "traj-a", "traj-b", "traj-b"],
        step_indices=[0, 1, 0, 1],
        step_scores=[0.0, 1.0, 1.0, 0.0],
    )

    assert success_by_round[0].tolist() == [0.0, 1.0]
    assert success_by_round[1].tolist() == [1.0, 1.0]


async def test_teacher_guidance_env_keeps_binary_reward_and_guidance_in_observation():
    env = TeacherGuidanceEnv(teacher_endpoint="http://teacher.example/v1")
    env.reset(
        raw_prompt=[{"role": "user", "content": "What is 1 + 1?"}],
        data_source="openai/gsm8k",
        reward_model={"ground_truth": "2"},
        extra_info={"answer": "1 + 1 = 2\n#### 2", "question": "What is 1 + 1?"},
    )

    async def fake_guidance(student_answer: str) -> str:
        assert student_answer == "#### 3"
        return "Check the addition carefully."

    env._generate_guidance = fake_guidance

    next_obs, reward, done, _ = await env.step(Action(text="#### 3"))
    assert reward == 0.0
    assert done is False
    assert next_obs.messages[-1]["role"] == "user"
    assert next_obs.messages[-1]["content"] == "Check the addition carefully."

    final_obs, reward, done, _ = await env.step(Action(text="#### 2"))
    assert reward == 1.0
    assert done is True
    assert final_obs.messages[-1]["role"] == "assistant"
