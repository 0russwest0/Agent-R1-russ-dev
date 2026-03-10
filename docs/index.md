# Agent-R1

## Training Powerful LLM Agents with End-to-End Reinforcement Learning

Agent-R1 is an open-source framework for training powerful language agents with end-to-end reinforcement learning. With Agent-R1, you can build custom agent workflows, define interactive environments and tools, and train multi-step agents in a unified RL pipeline.

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **Step-level MDP**

    ---

    A principled MDP formulation that enables flexible context management and per-step reward signals.

    [:octicons-arrow-down-24: Learn more](core-concepts/step-level-mdp.md)

-   :material-layers-outline:{ .lg .middle } **Layered Abstractions**

    ---

    From maximum flexibility to out-of-the-box, choose the right level of abstraction for your use case.

    [:octicons-arrow-down-24: Learn more](core-concepts/layered-abstractions.md)

</div>

---

## What Agent-R1 Focuses On

Agent-R1 is designed for **agent tasks**, not just single-step prompting. The framework is built around multi-step interaction, where an LLM acts inside an environment, receives the next observation, and improves through reinforcement learning over the whole trajectory.

The most important idea is a **step-level MDP**: each step has its own prompt, response, and reward. This makes it natural to train agents that call tools, interact with environments, and revise their behavior across multiple turns instead of treating everything as one long token stream.

See [`Step-level MDP`](core-concepts/step-level-mdp.md) for the full formulation.

---

## Layered Abstractions

Agent-R1 provides a **layered abstraction** stack so you can choose the right level of structure for your task:

- `AgentFlowBase` for maximum control over the rollout logic.
- `AgentEnvLoop + AgentEnv` for multi-step interaction with a custom environment.
- `ToolEnv + BaseTool` for tool-augmented agent tasks out of the box.

The framework's center of gravity is the second and third layers, where an LLM interacts with an environment across multiple steps. See [`Layered Abstractions`](core-concepts/layered-abstractions.md) for the complete breakdown.

---

## Reading Guide

- Start with [`Getting Started`](getting-started/index.md) if you want the minimal path: use the same environment as `verl`, run a sanity check, and confirm the repository is ready.
- Read [`Step-level MDP`](core-concepts/step-level-mdp.md) and [`Layered Abstractions`](core-concepts/layered-abstractions.md) if you want to understand the framework design before touching code.
- Follow [`Agent Task Tutorial`](tutorials/agent-task.md) if you want to see the main Agent-R1 workflow: multi-step interaction through `AgentEnvLoop` and `ToolEnv`.

## Scope of This Documentation

This version of the documentation is intentionally compact. It focuses on the parts that are already central to Agent-R1 today and leaves room for future tutorials as more environments and tools are added.

---

<div style="text-align: center; color: #888; margin-top: 2em;" markdown>
Built on [verl](https://github.com/volcengine/verl){ target=_blank } -- a flexible, efficient RL training framework for LLMs.
</div>
