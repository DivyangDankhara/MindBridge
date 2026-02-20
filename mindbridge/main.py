from pathlib import Path
import re

from config import DEFAULT_MODEL
from evaluator.evaluator import evaluate_goal
from executor.tool_registry import execute_plan
from intent.parser import parse_intent_file
from llm.openai_provider import OpenAIProvider
from memory.procedural_extractor import extract_procedural_strategies
from memory.procedural_store import load_procedural_strategies, save_procedural_strategy
from memory.retrieval import find_similar_experiences
from memory.schema import MissionExperience
from memory.semantic_extractor import extract_semantic_rules
from memory.semantic_store import load_semantic_rules, save_semantic_rule
from memory.store import save_experience
from planner.planner import create_plan


MAX_MISSION_ATTEMPTS = 5
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _summarize_plan(plan: object) -> str:
    if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
        descriptions: list[str] = []
        for step in plan["steps"]:
            if not isinstance(step, dict):
                continue
            description = step.get("description")
            if isinstance(description, str) and description.strip():
                descriptions.append(description.strip())

        if descriptions:
            return " | ".join(descriptions[:5])
        return "Structured plan with no step descriptions."

    if isinstance(plan, str):
        return plan.strip() or "Planner returned empty message."

    return f"Unsupported plan type: {type(plan).__name__}"


def _collect_tools_used(mission_history: list[dict[str, object]]) -> list[str]:
    tools: set[str] = set()
    for attempt in mission_history:
        plan = attempt.get("plan")
        if not isinstance(plan, dict):
            continue
        steps = plan.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool")
            if isinstance(tool_name, str) and tool_name.strip():
                tools.add(tool_name.strip())
    return sorted(tools)


def _collect_failure_reasons(mission_history: list[dict[str, object]]) -> list[str]:
    reasons: list[str] = []
    for attempt in mission_history:
        evaluation = attempt.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        if evaluation.get("goal_satisfied") is True:
            continue
        reason = evaluation.get("reason")
        if isinstance(reason, str) and reason.strip():
            reasons.append(reason.strip())
    return reasons


def _select_relevant_semantic_rules(
    task: str,
    goal: str,
    semantic_rules: list[dict[str, object]],
    limit: int = 3,
) -> list[dict[str, object]]:
    if limit <= 0:
        return []

    query_text = f"{task} {goal}".lower()
    query_terms = set(_WORD_PATTERN.findall(query_text))
    ranked: list[tuple[float, dict[str, object]]] = []

    for rule in semantic_rules:
        if not isinstance(rule, dict):
            continue
        rule_text = rule.get("rule")
        context_text = rule.get("context")
        confidence = rule.get("confidence")

        if not isinstance(rule_text, str) or not rule_text.strip():
            continue
        if not isinstance(context_text, str):
            context_text = ""

        candidate = f"{rule_text} {context_text}".lower()
        candidate_terms = set(_WORD_PATTERN.findall(candidate))
        overlap = len(query_terms.intersection(candidate_terms))

        score = float(overlap)
        if query_text and (query_text in candidate):
            score += 2.0

        try:
            score += float(confidence) * 0.5
        except Exception:
            pass

        if score > 0:
            ranked.append((score, rule))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [rule for _, rule in ranked[:limit]]


def _select_relevant_procedural_strategies(
    task: str,
    goal: str,
    strategies: list[dict[str, object]],
    limit: int = 3,
) -> list[dict[str, object]]:
    if limit <= 0:
        return []

    query_text = f"{task} {goal}".lower()
    query_terms = set(_WORD_PATTERN.findall(query_text))
    ranked: list[tuple[float, dict[str, object]]] = []

    for strategy in strategies:
        if not isinstance(strategy, dict):
            continue

        strategy_name = strategy.get("strategy_name")
        applicable_context = strategy.get("applicable_context")
        steps_template = strategy.get("steps_template")
        confidence = strategy.get("confidence")

        if not isinstance(strategy_name, str) or not strategy_name.strip():
            continue
        if not isinstance(applicable_context, str):
            applicable_context = ""
        if not isinstance(steps_template, list):
            continue

        steps_text = " ".join(step.strip() for step in steps_template if isinstance(step, str) and step.strip())
        candidate = f"{strategy_name} {applicable_context} {steps_text}".lower()
        candidate_terms = set(_WORD_PATTERN.findall(candidate))
        overlap = len(query_terms.intersection(candidate_terms))

        score = float(overlap)
        if query_text and (query_text in candidate):
            score += 2.0

        try:
            score += float(confidence) * 0.5
        except Exception:
            pass

        if score > 0:
            ranked.append((score, strategy))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [strategy for _, strategy in ranked[:limit]]


def main() -> None:
    project_root = Path(__file__).parent
    intent_path = project_root / "examples" / "test.intent"

    intent = parse_intent_file(intent_path)
    llm = OpenAIProvider(model=DEFAULT_MODEL)

    relevant_experiences = find_similar_experiences(intent.task, intent.goal, limit=3)
    if relevant_experiences:
        print(f"Loaded {len(relevant_experiences)} similar past experience(s).")
    else:
        print("No similar past experiences found.")

    semantic_rules = load_semantic_rules()
    relevant_semantic_rules = _select_relevant_semantic_rules(
        intent.task,
        intent.goal,
        semantic_rules,
        limit=3,
    )
    if relevant_semantic_rules:
        print(f"Loaded {len(relevant_semantic_rules)} relevant learned principle(s).")
    else:
        print("No relevant learned principles found.")

    procedural_strategies = load_procedural_strategies()
    relevant_procedural_strategies = _select_relevant_procedural_strategies(
        intent.task,
        intent.goal,
        procedural_strategies,
        limit=3,
    )
    if relevant_procedural_strategies:
        print(f"Loaded {len(relevant_procedural_strategies)} relevant learned strategy(ies).")
    else:
        print("No relevant learned strategies found.")

    mission_history: list[dict[str, object]] = []
    mission_accomplished = False

    for attempt in range(1, MAX_MISSION_ATTEMPTS + 1):
        print(f"\nMission attempt {attempt}/{MAX_MISSION_ATTEMPTS}")
        plan = create_plan(
            intent,
            llm,
            mission_history=mission_history,
            relevant_experiences=relevant_experiences,
            semantic_rules=relevant_semantic_rules,
            procedural_strategies=relevant_procedural_strategies,
        )

        execution_results: list[dict[str, object]] = []
        if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
            execution_results = execute_plan(plan, llm=llm)
            print("\nExecution results:")
            for item in execution_results:
                print(item)
        elif isinstance(plan, str):
            print(f"Planner message:\n{plan}")
        else:
            print(f"Planner returned unsupported output type: {type(plan).__name__}")

        evaluation: dict[str, object]
        if execution_results:
            evaluation = evaluate_goal(intent, execution_results, llm)
        else:
            evaluation = {
                "goal_satisfied": False,
                "reason": "No executable plan/results produced.",
            }

        print("\nGoal evaluation:")
        print(evaluation)

        mission_history.append(
            {
                "attempt": attempt,
                "plan": plan,
                "execution_results": execution_results,
                "evaluation": evaluation,
            }
        )

        if evaluation.get("goal_satisfied") is True:
            print("Mission accomplished")
            mission_accomplished = True
            break

        print("Goal not satisfied. Replanning...")

    if not mission_accomplished:
        print("Mission failed after maximum attempts")

    attempts_used = len(mission_history)
    final_attempt = mission_history[-1] if mission_history else {}
    final_plan = final_attempt.get("plan") if isinstance(final_attempt, dict) else None
    final_evaluation = final_attempt.get("evaluation") if isinstance(final_attempt, dict) else None
    final_reason = ""
    if isinstance(final_evaluation, dict) and isinstance(final_evaluation.get("reason"), str):
        final_reason = final_evaluation["reason"].strip()

    result_summary_prefix = "Mission succeeded" if mission_accomplished else "Mission failed"
    result_summary = f"{result_summary_prefix} after {attempts_used} attempt(s). {final_reason}".strip()

    experience = MissionExperience(
        timestamp=MissionExperience.now(),
        intent_task=intent.task,
        intent_goal=intent.goal,
        success=mission_accomplished,
        attempts=attempts_used,
        final_plan_summary=_summarize_plan(final_plan),
        failure_reasons=_collect_failure_reasons(mission_history),
        tools_used=_collect_tools_used(mission_history),
        result_summary=result_summary,
    )
    save_experience(experience)
    print("Mission experience saved to episodic memory.")

    extracted_rules = extract_semantic_rules(experience, llm)
    for rule_data in extracted_rules:
        save_semantic_rule(rule_data, llm=llm)
    print(f"Saved {len(extracted_rules)} semantic rule(s) to semantic memory.")

    if mission_accomplished:
        extracted_strategies = extract_procedural_strategies(experience, llm)
        for strategy_data in extracted_strategies:
            save_procedural_strategy(strategy_data)
        print(f"Saved {len(extracted_strategies)} procedural strategy(ies) to procedural memory.")


if __name__ == "__main__":
    main()
