import asyncio

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.data_generation.tests.classes import CheckEventTest
from autoppia_iwa.src.evaluation.stateful_evaluator.evaluator import AsyncStatefulEvaluator
from autoppia_iwa.src.web_agents.apified_iterative_agent import ApifiedWebAgent


async def run_task(*, task: Task, max_steps: int = 12) -> bool:
    evaluator = AsyncStatefulEvaluator(task=task, web_agent_id="traj-test-agent", should_record_gif=False, capture_screenshot=False)
    agent = ApifiedWebAgent(base_url="http://127.0.0.1:7000", name="forced-trajectory-operator")
    try:
        step_result = await evaluator.reset()
        for step_index in range(max_steps):
            actions = await agent.act(
                task=task,
                snapshot_html=step_result.snapshot.html,
                url=step_result.snapshot.url,
                step_index=step_index,
                history=None,
            )
            if not actions:
                break
            for action in actions:
                step_result = await evaluator.step(action)
                if step_result.score.success:
                    return True
        score = await evaluator.get_score_details()
        return bool(score.success)
    finally:
        await evaluator.close()


async def main() -> None:
    tasks: list[Task] = [
        Task(
            web_project_id="autocinema",
            url="http://localhost:8000/login",
            prompt="Login where username equals <username> and password equals <password>",
            tests=[CheckEventTest(event_name="LOGIN", event_criteria={"username": {"operator": "contains", "value": "user"}})],
        ),
        Task(
            web_project_id="autocinema",
            url="http://localhost:8000/register",
            prompt="Register where username equals signup_username, email equals signup_email and password equals signup_password",
            tests=[CheckEventTest(event_name="REGISTRATION", event_criteria={"username": {"operator": "contains", "value": "user"}})],
        ),
        Task(
            web_project_id="autocinema",
            url="http://localhost:8000/login",
            prompt="Login where username equals <username> and password equals <password>, then logout",
            tests=[CheckEventTest(event_name="LOGOUT", event_criteria={"username": {"operator": "contains", "value": "user"}})],
        ),
    ]

    names = ["LOGIN", "REGISTRATION", "LOGOUT"]
    for name, task in zip(names, tasks, strict=True):
        ok = await run_task(task=task, max_steps=15)
        print(f"{name}: {'SUCCESS ✅' if ok else 'FAIL ❌'}")


if __name__ == "__main__":
    asyncio.run(main())
