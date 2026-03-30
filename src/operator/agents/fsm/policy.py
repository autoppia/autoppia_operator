from __future__ import annotations

from .utils import *
from .state import *
from .candidates import *
from .observation import *
from .meta_tools import *

class Policy:
    def __init__(self, llm_call: Callable[..., Dict[str, Any]]) -> None:
        self.llm_call = llm_call
        self.debug_dir = str(os.getenv("FSM_POLICY_DEBUG_DIR", "") or "").strip()

    def _debug_log(self, task_id: str, payload: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        try:
            base = Path(self.debug_dir).expanduser().resolve()
            safe_task = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(task_id or "task"))[:120] or "task"
            _append_jsonl(base / f"{safe_task}.jsonl", payload)
        except Exception:
            return

    def decide(
        self,
        *,
        task_id: str,
        prompt: str,
        mode: str,
        policy_obs: Dict[str, Any],
        allowed_tools: set[str],
        model_name: str,
        plan_model_name: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        max_actions_per_step = max(1, min(_env_int("FSM_MAX_ACTIONS_PER_STEP", 3), 5))
        if mode == "POPUP":
            return {"type": "meta", "name": "META.SOLVE_POPUPS", "arguments": {}}, {"source": "deterministic"}
        if mode == "REPORT":
            facts = policy_obs.get("memory", {}).get("facts") if isinstance(policy_obs.get("memory"), dict) else []
            fact = _candidate_text((facts or [""])[0] if isinstance(facts, list) and facts else "")
            content = fact or "Task appears complete."
            return {"type": "final", "done": True, "content": content}, {"source": "deterministic"}

        meta_enabled = any(str(tool or "").startswith("META.") for tool in allowed_tools)
        direct_mode = mode == "DIRECT"
        if direct_mode:
            system = (
                "You are a browser-use-style web operator.\n"
                "You have the user task, the current browser state, the indexed interactive elements, the allowed browser tools, and the unavailable tools.\n"
                "Choose the next browser step sequence.\n"
                "Return ONE JSON object only. No markdown. No prose.\n"
                "Valid outputs:\n"
                '1) {"type":"browser","tool_call":{...}}\n'
                '2) {"type":"browser","tool_calls":[{...},{...}]}\n'
                '3) {"type":"final","done":true,"content":"..."}\n'
                "Rules:\n"
                f"- This runtime allows up to {max_actions_per_step} browser actions per step.\n"
                "- First decide whether the current page already satisfies the task. If yes, finish immediately with final or browser.done.\n"
                "- content must be the actual user-facing answer, result, or extracted value.\n"
                "- Do not keep exploring when the current page already satisfies the task.\n"
                f"- Never return more than {max_actions_per_step} browser actions.\n"
                "- If you return multiple actions, they must belong to the same local workflow and be safe to execute consecutively without re-observing.\n"
                "- Prefer arguments.index that refers to INTERACTIVE ELEMENT SHORTLIST.\n"
                "- For browser.select_dropdown, include a non-empty arguments.text.\n"
                "- browser.done is the standard way to finish once the page already satisfies the task.\n"
                "- Never emit unavailable tools.\n"
                "- If the task includes filters or explicit constraints, use visible controls first before opening result items.\n"
                "- Preserve placeholders such as <username>, <password>, <signup_email> exactly when typing.\n"
                "- When useful, include reasoning as a short human-readable operator note grounded in visible page evidence.\n"
                "- reasoning must be 1-2 short sentences, concrete, and suitable for product UI.\n"
                "- reasoning must say what is visible now and why the chosen next action or final answer follows.\n"
                "- reasoning must not contain chain-of-thought, filler, generic status text, or speculation without visible support.\n"
                "- Before choosing actions, infer one short local workflow plan for the current page and keep it stable until that workflow is completed or visibly blocked.\n"
                "- Update reasoning_trace.current_subgoal and reasoning_trace.plan to reflect the current local milestone, not the whole task from scratch.\n"
                "- Do not replan the whole task every step unless the page changed materially or the current workflow clearly failed.\n"
                "- If SCORE FEEDBACK is present in state and marks success=true or score=1.0, treat it as strong completion evidence and prefer final/browser.done unless visible evidence clearly contradicts it.\n"
            )
        else:
            system = (
            "You are a browser-use-style web automation policy.\n"
            "Given the task and the current browser state, choose the next browser step sequence.\n"
            "Return ONE JSON object only. No markdown. No prose. No chain-of-thought.\n"
            "You must choose exactly one of:\n"
            "1) browser tool_call or browser tool_calls\n"
            + ("2) meta_tool\n" if meta_enabled else "")
            + f"{'3' if meta_enabled else '2'}) final (done=true + content)\n\n"
            "Rules:\n"
            f"- This runtime allows up to {max_actions_per_step} browser actions per step.\n"
            "- Prefer a concrete browser action when there is a reasonable actionable target.\n"
            + ("- Use a meta_tool only when inspection/disambiguation materially improves the next browser action.\n" if meta_enabled else "")
            + f"- Never return more than {max_actions_per_step} browser actions.\n"
            + "- If you return multiple browser actions, they must stay within the same local workflow and should usually be a short form-filling or commit sequence.\n"
            "- If the current page already contains the answer, return final immediately.\n"
            "- For question-answering and data-extraction tasks, DONE is the correct action once the answer is visible on the current page.\n"
            "- Do NOT keep exploring once the current page already answers the task.\n"
            "- Use final/done or browser.done with a concrete content string when the task is satisfied.\n"
            "- content must be the actual answer for the user, not a status message.\n"
            "- Avoid repeating low-value actions when the page did not materially change.\n"
            "- Prefer arguments.index that refers to INTERACTIVE ELEMENT SHORTLIST.\n"
            "- For browser.select_dropdown, provide arguments.text with the option text/value to choose.\n"
            "- Never emit unavailable tools.\n"
            "- Preserve placeholders such as <username>, <password>, <signup_email> exactly when typing.\n"
            "- For informational tasks, use the current visible content before navigating more.\n"
            "- When useful, include reasoning as a short human-readable operator note grounded in visible page evidence.\n"
            "- reasoning must be 1-2 short sentences, concrete, and suitable for product UI.\n"
            "- reasoning must say what is visible now and why the chosen next action or final answer follows.\n"
            "- reasoning must not contain chain-of-thought, filler, generic status text, or speculation without visible support.\n"
            "- Before choosing actions, infer one short local workflow plan for the current page and keep it stable until that workflow is completed or visibly blocked.\n"
            "- Update reasoning_trace.current_subgoal and reasoning_trace.plan to reflect the current local milestone, not the whole task from scratch.\n"
            "- Do not replan the whole task every step unless the page changed materially or the current workflow clearly failed.\n"
            "- If SCORE FEEDBACK is present in state and marks success=true or score=1.0, treat it as strong completion evidence and prefer final/browser.done unless visible evidence clearly contradicts it.\n"
            )
        if direct_mode:
            user_parts = [
                "Choose the next browser step sequence.",
                f"TASK: {str(policy_obs.get('prompt') or '')[:1600]}",
                "TASK CONSTRAINTS:",
                json.dumps(
                    (
                        policy_obs.get("task_constraints")
                        if isinstance(policy_obs.get("task_constraints"), dict)
                        else {}
                    ),
                    ensure_ascii=False,
                ),
                f"STEP: {int(policy_obs.get('step_index') or 0)}",
                f"URL: {str(policy_obs.get('url') or '')[:1000]}",
                "",
                "RUNTIME:",
                "browser-use-like operator runtime",
                f"max_actions_per_step={max_actions_per_step}",
                "tabs_supported=false",
                "file_tools_supported=false",
                "",
                "UNAVAILABLE TOOLS:",
                ", ".join(policy_obs.get("unavailable_browser_tools") if isinstance(policy_obs.get("unavailable_browser_tools"), list) else list(_unavailable_browser_tools())),
                "",
                "TOOL USAGE GUIDE:",
                "- browser.click: buttons, links, toggles, tabs, submit controls, checkboxes, radios.",
                "- browser.input: text-entry fields only.",
                "- browser.select_dropdown: only when a concrete option text is known.",
                "- browser.dropdown_options: inspect a select before choosing if the option is unclear.",
                "- browser.extract: deterministic extraction from visible page content.",
                "- browser.search: external web search, not in-page site search boxes.",
                "",
                "REASONING FIELD CONTRACT:",
                "- You may include reasoning as a short operator note for humans.",
                "- reasoning must be 1-2 short sentences, max about 220 characters total.",
                "- reasoning must mention visible evidence or current page state and the next action or final answer.",
                "- Good reasoning example: 'The comment form is already open and only the message field is still empty, so fill it and submit.'",
                "- Bad reasoning example: 'I think this is probably the right thing to do next because it seems correct.'",
                "- Do not include hidden reasoning, self-talk, generic filler, or unsupported guesses.",
                "",
                "PLAN MAINTENANCE CONTRACT:",
                "- First infer a short local workflow for the current page before selecting actions.",
                "- Keep that workflow stable across steps unless visible evidence shows it is blocked or no longer relevant.",
                "- reasoning_trace.current_subgoal should name the current local milestone.",
                "- reasoning_trace.plan should describe the next short sequence on this page, not the whole task history.",
                "- If a form/filter/comment workflow is already active, prefer finishing it before exploring unrelated controls.",
                "",
                "REASONING TRACE CONTRACT:",
                "- You may include reasoning_trace as a short JSON object with keys task_interpretation, success_state, current_subgoal, next_expected_proof, drift_risks, where_am_i, state_assessment, plan.",
                "- Keep each field short and concrete.",
                "- reasoning_trace must describe task meaning and success, not hidden chain-of-thought.",
                "",
                "WORKING STATE CONTRACT:",
                "- You should include working_state as a short JSON object with keys current_page_kind, active_region, active_workflow, completed_fields, pending_fields, completion_evidence_missing, next_milestone, completion_state.",
                "- working_state should describe the current operational state, not hidden reasoning.",
                "",
                "ACTIVE OBJECTIVE (JSON):",
                json.dumps(policy_obs.get("active_objective") if isinstance(policy_obs.get("active_objective"), dict) else {}, ensure_ascii=False),
                "",
                "WORKING STATE (JSON):",
                json.dumps(policy_obs.get("working_state") if isinstance(policy_obs.get("working_state"), dict) else {}, ensure_ascii=False),
                "",
                *(
                    [
                        f"LLMJudgeEvaluator says score {policy_obs.get('state_score')}",
                        "",
                    ]
                    if policy_obs.get("state_score") is not None
                    else []
                ),
                "SCORE FEEDBACK (JSON):",
                json.dumps(policy_obs.get("score_feedback") if isinstance(policy_obs.get("score_feedback"), dict) else {}, ensure_ascii=False),
                "",
                "LOCAL WORKFLOW CLOSURE (JSON):",
                json.dumps(policy_obs.get("local_workflow_closure") if isinstance(policy_obs.get("local_workflow_closure"), dict) else {}, ensure_ascii=False),
                "",
                "LOCAL HTML CONTEXT (JSON):",
                json.dumps(policy_obs.get("local_html_context") if isinstance(policy_obs.get("local_html_context"), dict) else {}, ensure_ascii=False),
                "",
                "KNOWN SITE MAP (JSON):",
                json.dumps(policy_obs.get("site_knowledge") if isinstance(policy_obs.get("site_knowledge"), dict) else {}, ensure_ascii=False),
                "",
                "AVOID REPEATING (JSON):",
                json.dumps(policy_obs.get("avoid_repeating") if isinstance(policy_obs.get("avoid_repeating"), dict) else {}, ensure_ascii=False),
                "",
                "PREVIOUS REASONING TRACE (JSON):",
                json.dumps(policy_obs.get("reasoning_trace") if isinstance(policy_obs.get("reasoning_trace"), dict) else {}, ensure_ascii=False),
                "",
                "DONE / CONTENT CONTRACT:",
                "- If the answer or completed result is already visible on the current page, return final now.",
                "- final.content or browser.done.arguments.content must be the concrete answer for the user.",
                "- Do not return generic status text like 'task complete' or 'done'.",
                "- Before taking another action, check BROWSER SNAPSHOT, VISIBLE TEXT / PAGE SUMMARY, and INTERACTIVE ELEMENT SHORTLIST.",
                "- If a select already shows the target value, do not select the same value again.",
                "- Prefer arguments.index when targeting an interactive element.",
                "- Follow ACTIVE OBJECTIVE unless visible evidence proves the answer is already on the page.",
                "- Respect AVOID REPEATING unless the page materially changed.",
                "- If LOCAL WORKFLOW CLOSURE shows ready_to_commit=true and a visible commit control exists, strongly prefer finishing that local workflow before exploring unrelated controls.",
                "- Use LOCAL HTML CONTEXT to understand which inputs and commit controls belong to the same active form or region.",
                "- If multiple actions are returned, explain the local workflow in reasoning_trace.plan.",
                "- If SCORE FEEDBACK is present and success=true or score=1.0, prefer finishing now unless the page visibly contradicts that signal.",
                "",
                "BROWSER SNAPSHOT:",
                str(policy_obs.get("browser_state_snapshot") or "")[:3000],
                "",
                "VISIBLE TEXT / PAGE SUMMARY:",
                str(policy_obs.get("page_ir_text") or "")[:14000],
                "",
                "VISIBLE EVIDENCE (JSON):",
                json.dumps(
                    {
                        "likely_answers": (
                            policy_obs.get("page_observations", {}).get("likely_answers")
                            if isinstance(policy_obs.get("page_observations"), dict)
                            and isinstance(policy_obs.get("page_observations", {}).get("likely_answers"), list)
                            else []
                        )[:6],
                        "relevant_lines": (
                            policy_obs.get("page_observations", {}).get("relevant_lines")
                            if isinstance(policy_obs.get("page_observations"), dict)
                            and isinstance(policy_obs.get("page_observations", {}).get("relevant_lines"), list)
                            else []
                        )[:10],
                    },
                    ensure_ascii=False,
                ),
                "",
                "RECENT ACTIONS AND RESULTS (JSON):",
                json.dumps(policy_obs.get("history_recent") if isinstance(policy_obs.get("history_recent"), list) else [], ensure_ascii=False),
                "",
                "PAGE OBSERVATIONS (JSON):",
                json.dumps(policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}, ensure_ascii=False),
                "",
                "INTERACTIVE ELEMENTS (indexed tree-style):",
                str(policy_obs.get("browser_state_text") or "")[:9000],
                "",
                "INTERACTIVE ELEMENT SHORTLIST (JSON):",
                json.dumps(
                    (policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else [])[:14],
                    ensure_ascii=False,
                ),
                "",
                "ALLOWED BROWSER TOOLS: " + ", ".join(sorted([t for t in list(allowed_tools) if str(t).startswith("browser.")]) if allowed_tools else []),
                "",
                "Output schema examples:",
                '{"type":"browser","tool_call":{"name":"browser.click","arguments":{"index":0}}}',
                '{"type":"browser","reasoning":"The comment form is already open and only the message field is still missing, so complete it and submit on this page.","reasoning_trace":{"task_interpretation":"Open the correct movie and submit a valid comment.","success_state":"A new comment is visibly posted on the target movie.","current_subgoal":"Finish the visible comment form.","next_expected_proof":"A post/submit action or the new comment appears.","drift_risks":"Opening related movies or share widgets.","where_am_i":"A movie detail page with a visible comment form.","state_assessment":"The form is ready and only needs local completion.","plan":"Fill the visible fields and submit without leaving this page."},"working_state":{"current_page_kind":"movie detail page","active_region":"comment form","active_workflow":"submit a valid movie comment","completed_fields":["name"],"pending_fields":["comment"],"completion_evidence_missing":["posted comment visible"],"next_milestone":"Submit the visible comment form.","completion_state":"awaiting_local_completion"},"tool_calls":[{"name":"browser.input","arguments":{"index":0,"text":"Jordan"}},{"name":"browser.input","arguments":{"index":1,"text":"Great pacing and atmosphere."}},{"name":"browser.click","arguments":{"index":2}}]}',
                '{"type":"browser","tool_call":{"name":"browser.done","arguments":{"content":"The total value is 2844."}}}',
                '{"type":"final","done":true,"content":"The total value is 2844.","reasoning":"The total value is already visible on the current page, so return it directly."}',
            ]
        else:
            user_parts = [
            "You have a task and must choose the next browser step sequence.",
            f"TASK: {str(policy_obs.get('prompt') or '')[:1600]}",
            f"STEP: {int(policy_obs.get('step_index') or 0)}",
            f"MODE: {mode}",
            f"URL: {str(policy_obs.get('url') or '')[:1000]}",
            f"SCREENSHOT_AVAILABLE: {bool(policy_obs.get('screenshot_available'))}",
            "",
            "RUNTIME:",
            "browser-use-like operator runtime",
            f"max_actions_per_step={max_actions_per_step}",
            "tabs_supported=false",
            "file_tools_supported=false",
            "",
            "TOOL USAGE GUIDE:",
            "- browser.click: buttons, links, toggles, tabs, submit controls, checkboxes, radios.",
            "- browser.input: text-entry fields only.",
            "- browser.select_dropdown: only when a concrete option text is known.",
            "- browser.dropdown_options: inspect a select before choosing if the option is unclear.",
            "- browser.extract: deterministic extraction from visible page content.",
            "- browser.search: external web search, not in-page site search boxes.",
            "",
            "REASONING FIELD CONTRACT:",
            "- You may include reasoning as a short operator note for humans.",
            "- reasoning must be 1-2 short sentences, max about 220 characters total.",
            "- reasoning must mention visible evidence or current page state and the next action or final answer.",
            "- Good reasoning example: 'The filter panel is already visible and the apply button is in the same group, so finish this local filter sequence first.'",
            "- Bad reasoning example: 'I will think step by step and try something that might work.'",
            "- Do not include hidden reasoning, self-talk, generic filler, or unsupported guesses.",
            "",
            "PLAN MAINTENANCE CONTRACT:",
            "- First infer a short local workflow for the current page before selecting actions.",
            "- Keep that workflow stable across steps unless visible evidence shows it is blocked or no longer relevant.",
            "- reasoning_trace.current_subgoal should name the current local milestone.",
            "- reasoning_trace.plan should describe the next short sequence on this page, not the whole task history.",
            "- If a form/filter/comment workflow is already active, prefer finishing it before exploring unrelated controls.",
            "",
            "REASONING TRACE CONTRACT:",
            "- You may include reasoning_trace as a short JSON object with keys task_interpretation, success_state, current_subgoal, next_expected_proof, drift_risks, where_am_i, state_assessment, plan.",
            "- Keep each field short and concrete.",
            "- reasoning_trace must describe task meaning and success, not hidden chain-of-thought.",
            "",
            "WORKING STATE CONTRACT:",
            "- You should include working_state as a short JSON object with keys current_page_kind, active_region, active_workflow, completed_fields, pending_fields, completion_evidence_missing, next_milestone, completion_state.",
            "- working_state should describe the current operational state, not hidden reasoning.",
            "",
            "ACTIVE OBJECTIVE (JSON):",
            json.dumps(policy_obs.get("active_objective") if isinstance(policy_obs.get("active_objective"), dict) else {}, ensure_ascii=False),
            "",
            "WORKING STATE (JSON):",
            json.dumps(policy_obs.get("working_state") if isinstance(policy_obs.get("working_state"), dict) else {}, ensure_ascii=False),
            "",
            *(
                [
                    f"LLMJudgeEvaluator says score {policy_obs.get('state_score')}",
                    "",
                ]
                if policy_obs.get("state_score") is not None
                else []
            ),
            "SCORE FEEDBACK (JSON):",
            json.dumps(policy_obs.get("score_feedback") if isinstance(policy_obs.get("score_feedback"), dict) else {}, ensure_ascii=False),
            "",
            "LOCAL WORKFLOW CLOSURE (JSON):",
            json.dumps(policy_obs.get("local_workflow_closure") if isinstance(policy_obs.get("local_workflow_closure"), dict) else {}, ensure_ascii=False),
            "",
            "LOCAL HTML CONTEXT (JSON):",
            json.dumps(policy_obs.get("local_html_context") if isinstance(policy_obs.get("local_html_context"), dict) else {}, ensure_ascii=False),
            "",
            "KNOWN SITE MAP (JSON):",
            json.dumps(policy_obs.get("site_knowledge") if isinstance(policy_obs.get("site_knowledge"), dict) else {}, ensure_ascii=False),
            "",
            "AVOID REPEATING (JSON):",
            json.dumps(policy_obs.get("avoid_repeating") if isinstance(policy_obs.get("avoid_repeating"), dict) else {}, ensure_ascii=False),
            "",
            "PREVIOUS REASONING TRACE (JSON):",
            json.dumps(policy_obs.get("reasoning_trace") if isinstance(policy_obs.get("reasoning_trace"), dict) else {}, ensure_ascii=False),
            "",
            "DONE / CONTENT CONTRACT:",
            "- If the task is already answered by the current page, return final now.",
            "- final.content must contain the user-facing answer.",
            "- Do not output a browser action when the answer is already visible.",
            "- Prefer quoting the visible metric / value directly in content.",
            "- Follow ACTIVE OBJECTIVE unless visible evidence already satisfies the task.",
            "- Respect AVOID REPEATING unless the page materially changed.",
            "- If LOCAL WORKFLOW CLOSURE shows ready_to_commit=true and a visible commit control exists, strongly prefer finishing that local workflow before exploring unrelated controls.",
            "- Use LOCAL HTML CONTEXT to understand which inputs and commit controls belong to the same active form or region.",
            "- If multiple actions are returned, they must form one short local workflow and reasoning_trace.plan must say why.",
            "- If SCORE FEEDBACK is present and success=true or score=1.0, prefer finishing now unless the page visibly contradicts that signal.",
            "",
            "BROWSER SNAPSHOT:",
            str(policy_obs.get("browser_state_snapshot") or "")[:3000],
            "",
            "VISIBLE TEXT / PAGE SUMMARY:",
            str(policy_obs.get("page_ir_text") or "")[:14000],
            "",
            "VISIBLE EVIDENCE (JSON):",
            json.dumps(
                {
                    "likely_answers": (
                        policy_obs.get("page_observations", {}).get("likely_answers")
                        if isinstance(policy_obs.get("page_observations"), dict)
                        and isinstance(policy_obs.get("page_observations", {}).get("likely_answers"), list)
                        else []
                    )[:6],
                    "relevant_lines": (
                        policy_obs.get("page_observations", {}).get("relevant_lines")
                        if isinstance(policy_obs.get("page_observations"), dict)
                        and isinstance(policy_obs.get("page_observations", {}).get("relevant_lines"), list)
                        else []
                    )[:10],
                },
                ensure_ascii=False,
            ),
            "",
            "INTERACTIVE ELEMENTS (indexed tree-style):",
            str(policy_obs.get("browser_state_text") or "")[:14000],
            "",
            "INTERACTIVE ELEMENT SHORTLIST (JSON):",
            json.dumps(
                (policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else [])[:24],
                ensure_ascii=False,
            ),
            "",
            "RECENT ACTIONS AND RESULTS (JSON):",
            json.dumps(
                (policy_obs.get("history_recent") if isinstance(policy_obs.get("history_recent"), list) else [])[:10],
                ensure_ascii=False,
            ),
            "",
            "PAGE OBSERVATIONS (JSON):",
            json.dumps(policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}, ensure_ascii=False),
            "",
            "PAGE GROUPS (FORMS / CONTROL GROUPS / ITEM GROUPS JSON):",
            json.dumps(
                {
                    "forms": (
                        policy_obs.get("text_ir", {}).get("forms")
                        if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("forms"), list)
                        else []
                    )[:6],
                    "control_groups": (
                        policy_obs.get("text_ir", {}).get("control_groups")
                        if isinstance(policy_obs.get("text_ir"), dict)
                        and isinstance(policy_obs.get("text_ir", {}).get("control_groups"), list)
                        else []
                    )[:8],
                    "cards": (
                        policy_obs.get("text_ir", {}).get("cards")
                        if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("cards"), list)
                        else []
                    )[:8],
                },
                ensure_ascii=False,
            ),
            "",
            "UNAVAILABLE TOOLS:",
            ", ".join(policy_obs.get("unavailable_browser_tools") if isinstance(policy_obs.get("unavailable_browser_tools"), list) else list(_unavailable_browser_tools())),
            "",
            "PREVIOUS STEP VERDICT:",
            json.dumps(policy_obs.get("previous_step_verdict") if isinstance(policy_obs.get("previous_step_verdict"), dict) else {}, ensure_ascii=False),
            ]
            history_summary = str(policy_obs.get("history_summary") or "").strip()
            if history_summary:
                user_parts.extend(["", "HISTORY SUMMARY:", history_summary[:2000]])
            history_recent = policy_obs.get("history_recent") if isinstance(policy_obs.get("history_recent"), list) else []
            recent_failures = [
                item
                for item in history_recent[-8:]
                if isinstance(item, dict) and (not bool(item.get("exec_ok", True)) or str(item.get("error") or "").strip())
            ]
            if recent_failures:
                user_parts.extend(["", "RECENT FAILURES (JSON):", json.dumps(recent_failures[:4], ensure_ascii=False)])
            loop_nudges = policy_obs.get("loop_nudges") if isinstance(policy_obs.get("loop_nudges"), list) else []
            if loop_nudges:
                user_parts.extend(["", "LOOP NUDGES:"] + [f"- {str(item)[:220]}" for item in loop_nudges[:8]])
            memory = policy_obs.get("memory") if isinstance(policy_obs.get("memory"), dict) else {}
            strategy_summary = str(memory.get("strategy_summary") or "").strip()
            capability_gap = (
                policy_obs.get("page_observations", {}).get("capability_gap")
                if isinstance(policy_obs.get("page_observations"), dict)
                and isinstance(policy_obs.get("page_observations", {}).get("capability_gap"), dict)
                else {}
            )
            if strategy_summary or capability_gap:
                user_parts.extend(
                    [
                        "",
                        "STRATEGY / CAPABILITY GAP:",
                        json.dumps(
                            {
                                "strategy_summary": strategy_summary,
                                "capability_gap": capability_gap,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
            plan = policy_obs.get("plan") if isinstance(policy_obs.get("plan"), dict) else {}
            counters = policy_obs.get("counters") if isinstance(policy_obs.get("counters"), dict) else {}
            visual_notes = memory.get("visual_notes") if isinstance(memory.get("visual_notes"), list) else []
            visual_hints = memory.get("visual_element_hints") if isinstance(memory.get("visual_element_hints"), list) else []
            if visual_notes or visual_hints:
                user_parts.extend(
                    [
                        "",
                        "VISUAL OBSERVATIONS:",
                        json.dumps(
                            {"notes": visual_notes[:8], "element_hints": visual_hints[:12]},
                            ensure_ascii=False,
                        ),
                    ]
                )
            user_parts.extend(
                [
                "",
                "PLAN / MEMORY (JSON):",
                json.dumps(
                    {
                        "task_constraints": policy_obs.get("task_constraints") if isinstance(policy_obs.get("task_constraints"), dict) else {},
                        "active_subgoal": policy_obs.get("active_subgoal") if isinstance(policy_obs.get("active_subgoal"), dict) else {},
                        "plan": plan,
                        "memory": memory,
                        "counters": counters,
                        "frontier": policy_obs.get("frontier") if isinstance(policy_obs.get("frontier"), dict) else {},
                    },
                    ensure_ascii=False,
                ),
                "ALLOWED BROWSER TOOLS: " + ", ".join(sorted([t for t in list(allowed_tools) if str(t).startswith("browser.")]) if allowed_tools else []),
                "",
                "Output schema examples:",
                '{"type":"browser","tool_call":{"name":"browser.click","arguments":{"index":0}}}',
                '{"type":"browser","reasoning":"The filter panel is already visible and the next useful step is to finish the local filter sequence before opening any result cards.","reasoning_trace":{"task_interpretation":"Use the current page controls to narrow results.","success_state":"The required filtered result is visible.","current_subgoal":"Apply the current filter set.","next_expected_proof":"An apply/search result change is visible.","drift_risks":"Opening unrelated result cards too early.","where_am_i":"A filter panel with visible controls.","state_assessment":"The target controls are already visible.","plan":"Complete the filter sequence locally before opening any result cards."},"working_state":{"current_page_kind":"filter results page","active_region":"filter panel","active_workflow":"apply the current filter set","completed_fields":["genre"],"pending_fields":["apply filters"],"completion_evidence_missing":["updated result set"],"next_milestone":"Apply the visible filter controls.","completion_state":"awaiting_local_completion"},"tool_calls":[{"name":"browser.select_dropdown","arguments":{"index":1,"text":"Comedy"}},{"name":"browser.click","arguments":{"index":2}}]}',
                '{"type":"browser","tool_call":{"name":"browser.select_dropdown","arguments":{"index":1,"text":"Comedy"}}}',
                '{"type":"browser","tool_call":{"name":"browser.done","arguments":{"content":"The total value is 2844."}}}',
                '{"type":"final","done":true,"content":"The total value is 2844.","reasoning":"The total value is already visible on the page, so the task can finish without another browser action."}',
                "",
                "Instructions:",
                "- Output JSON only.",
                f"- Return at most {max_actions_per_step} browser actions.",
                "- Do not burn steps on the same no-op pattern if nothing changed.",
                "- Prefer arguments.index over selector or element_id when targeting an interactive element.",
                "- For browser.select_dropdown, include a non-empty arguments.text.",
                "- If the task is about narrowing results, use current-page controls before opening result items.",
                "- Preserve placeholders exactly when typing.",
                "- For informational tasks, prefer answering from what is already visible on the current page before opening more pages.",
                "- If the current page is sufficient, finish now.",
                "- Never emit unavailable tools.",
                "- Do not return content like 'task completed'; return the actual answer.",
                ]
            )
            if meta_enabled:
                user_parts.insert(-5, '{"type":"meta","meta_tool":{"name":"META.FIND_ELEMENTS","arguments":{"role":"input","text":"search","limit":6}}}')
                if "META.VISION_QA" in _obs_meta_tools():
                    user_parts.insert(-5, '{"type":"meta","meta_tool":{"name":"META.VISION_QA","arguments":{"question":"Which visible control best applies the current filters?"}}}')
        user_text = "\n".join([part for part in user_parts if part is not None])[:45000]
        model = plan_model_name if (mode == "PLAN" or mode == "STUCK") else model_name
        self._debug_log(
            str(task_id or "task"),
            {
                "ts": _utc_now(),
                "phase": "llm_request",
                "mode": mode,
                "model": model,
                "system": system,
                "user": user_text,
                "allowed_tools_count": len(allowed_tools),
            },
        )
        try:
            raw = self.llm_call(
                task_id=str(task_id or "local"),
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.1,
                max_tokens=1600,
            )
            content = str(
                (((raw or {}).get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
            )
            usage = self._normalize_usage(raw)
            try:
                obj = self._parse_json(content)
                normalized = self._normalize_decision(obj, allowed_tools)
            except Exception as parse_or_schema_err:
                repaired = self._attempt_repair(
                    task_id=str(task_id or "local"),
                    model=model,
                    mode=mode,
                    raw_content=content,
                    allowed_tools=allowed_tools,
                )
                if repaired is None:
                    raise parse_or_schema_err
                normalized, repair_usage, repair_model, repair_content = repaired
                self._debug_log(
                    str(task_id or "task"),
                    {
                        "ts": _utc_now(),
                        "phase": "llm_repair_response",
                        "mode": mode,
                        "model": str(repair_model or model),
                        "raw_content": repair_content,
                        "normalized": normalized,
                        "usage": repair_usage,
                        "original_error": str(parse_or_schema_err),
                    },
                )
                return normalized, {
                    "source": "llm_repair",
                    "usage": repair_usage,
                    "model": str(repair_model or model),
                }
            self._debug_log(
                str(task_id or "task"),
                {
                    "ts": _utc_now(),
                    "phase": "llm_response",
                    "mode": mode,
                    "model": str((raw or {}).get("model") or model),
                    "raw_content": content,
                    "parsed": obj,
                    "normalized": normalized,
                    "usage": usage,
                },
            )
            return normalized, {
                "source": "llm",
                "usage": usage,
                "model": str((raw or {}).get("model") or model),
            }
        except Exception as e:
            self._debug_log(
                str(task_id or "task"),
                {
                    "ts": _utc_now(),
                    "phase": "llm_error",
                    "mode": mode,
                    "model": model,
                    "error": str(e),
                },
            )
            fallback = self._fallback(prompt=prompt, mode=mode, policy_obs=policy_obs, allowed_tools=allowed_tools)
            self._debug_log(
                str(task_id or "task"),
                {
                    "ts": _utc_now(),
                    "phase": "fallback_decision",
                    "mode": mode,
                    "decision": fallback,
                },
            )
            return fallback, {"source": "fallback"}

    def _normalize_usage(self, raw: Dict[str, Any] | None) -> Dict[str, int]:
        usage = (raw or {}).get("usage") if isinstance((raw or {}).get("usage"), dict) else {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    def _extract_first_json_object(self, raw: str) -> str | None:
        if not raw:
            return None
        in_str = False
        esc = False
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
                continue
            if ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        return raw[start : i + 1]
        return None

    def _parse_json(self, content: str) -> Dict[str, Any]:
        raw = str(content or "").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        first_obj = self._extract_first_json_object(raw)
        if first_obj:
            try:
                obj = json.loads(first_obj)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            # Weak fallback for single-quoted dict-like outputs.
            try:
                lit = ast.literal_eval(first_obj)
                if isinstance(lit, dict):
                    return json.loads(json.dumps(lit, ensure_ascii=False))
            except Exception:
                pass
        raise ValueError("invalid_json_policy_output")

    def _normalize_decision(self, obj: Dict[str, Any], allowed_tools: set[str]) -> Dict[str, Any]:
        reasoning_trace = _normalize_reasoning_trace(obj.get("reasoning_trace"))
        working_state = _normalize_working_state(obj.get("working_state"))
        reasoning_summary = _candidate_text(_reasoning_trace_summary(reasoning_trace), obj.get("reasoning"))
        working_state_summary = _working_state_summary(working_state)
        if working_state_summary:
            reasoning_summary = _candidate_text(reasoning_summary, working_state_summary)
        max_actions_per_step = max(1, min(_env_int("FSM_MAX_ACTIONS_PER_STEP", 3), 5))
        t = str(obj.get("type") or "").strip().lower()
        if not t:
            if isinstance(obj.get("tool_call"), dict):
                t = "browser"
            elif isinstance(obj.get("tool_calls"), list):
                t = "browser"
            elif isinstance(obj.get("meta_tool"), dict):
                t = "meta"
            elif str(obj.get("name") or obj.get("tool") or "").strip():
                maybe_name = _canonical_allowed_tool_name(str(obj.get("name") or obj.get("tool") or ""))
                if maybe_name.startswith("browser."):
                    t = "browser"
                elif maybe_name.startswith("meta.") or maybe_name.startswith("META."):
                    t = "meta"
            elif bool(obj.get("done")) or isinstance(obj.get("content"), str):
                t = "final"
        if t == "final":
            return {
                "type": "final",
                "done": True,
                "content": _candidate_text(obj.get("content"), "Task complete."),
                "reasoning": reasoning_summary,
                "reasoning_trace": reasoning_trace,
                "working_state": working_state,
            }
        if t == "meta":
            mt = obj.get("meta_tool") if isinstance(obj.get("meta_tool"), dict) else {}
            name = str(mt.get("name") or obj.get("name") or "").strip().upper()
            if name in _meta_tools() and (not allowed_tools or name in allowed_tools):
                return {
                    "type": "meta",
                    "name": name,
                    "arguments": (
                        mt.get("arguments")
                        if isinstance(mt.get("arguments"), dict)
                        else (obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {})
                    ),
                    "reasoning": reasoning_summary,
                    "reasoning_trace": reasoning_trace,
                    "working_state": working_state,
                }
        if t == "browser":
            raw_calls = obj.get("tool_calls") if isinstance(obj.get("tool_calls"), list) else None
            normalized_calls: List[Dict[str, Any]] = []
            if raw_calls:
                for item in raw_calls[:max_actions_per_step]:
                    if isinstance(item, dict):
                        normalized_calls.append(item)
            else:
                tc = obj.get("tool_call") if isinstance(obj.get("tool_call"), dict) else {}
                if not tc:
                    tc = {
                        "name": str(obj.get("name") or obj.get("tool") or "").strip(),
                        "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
                    }
                normalized_calls = [tc]
            cleaned_calls: List[Dict[str, Any]] = []
            saw_done = False
            for tc in normalized_calls:
                raw_name = str(tc.get("name") or "").strip()
                name = _canonical_allowed_tool_name(raw_name)
                args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                if name == "browser.done":
                    saw_done = True
                    continue
                if name == "browser.input":
                    if _is_generic_tool_placeholder(_candidate_text(args.get("text"), args.get("value")), kind="input"):
                        raise ValueError("invalid_browser_input_text")
                if name == "browser.select_dropdown":
                    selected_text = _candidate_text(args.get("text"), args.get("value"))
                    if _is_generic_tool_placeholder(selected_text, kind="select"):
                        if (not allowed_tools) or ("browser.dropdown_options" in allowed_tools):
                            tc = {
                                "name": "browser.dropdown_options",
                                "arguments": {
                                    key: value
                                    for key, value in args.items()
                                    if key in {"index", "element_id", "_element_id", "selector"}
                                },
                            }
                            name = "browser.dropdown_options"
                        else:
                            raise ValueError("invalid_browser_select_text")
                if name.startswith("browser.") and (not allowed_tools or name in allowed_tools):
                    cleaned_calls.append(
                        {
                            "name": name,
                            "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {},
                        }
                    )
            if saw_done and not cleaned_calls:
                first_args = normalized_calls[0].get("arguments") if normalized_calls and isinstance(normalized_calls[0], dict) else {}
                return {
                    "type": "final",
                    "done": True,
                    "content": _candidate_text(
                        (first_args.get("content") if isinstance(first_args, dict) else None),
                        obj.get("content"),
                        "Task complete.",
                    ),
                    "reasoning": reasoning_summary,
                    "reasoning_trace": reasoning_trace,
                    "working_state": working_state,
                }
            if cleaned_calls:
                out: Dict[str, Any] = {
                    "type": "browser",
                    "reasoning": reasoning_summary,
                    "reasoning_trace": reasoning_trace,
                    "working_state": working_state,
                }
                if len(cleaned_calls) == 1:
                    out["tool_call"] = cleaned_calls[0]
                out["tool_calls"] = cleaned_calls
                return out
        raise ValueError("invalid_policy_decision")

    def _attempt_repair(
        self,
        *,
        task_id: str,
        model: str,
        mode: str,
        raw_content: str,
        allowed_tools: set[str],
    ) -> tuple[Dict[str, Any], Dict[str, int], str, str] | None:
        repair_system = (
            "You fix malformed agent policy outputs.\n"
            "Return exactly ONE valid JSON object and nothing else.\n"
            "Output must match one of:\n"
            "1) {\"type\":\"browser\",\"tool_call\":{\"name\":\"browser.<tool>\",\"arguments\":{\"index\":0}}}\n"
            "2) {\"type\":\"meta\",\"meta_tool\":{\"name\":\"META.<TOOL>\",\"arguments\":{}}}\n"
            "3) {\"type\":\"final\",\"done\":true,\"content\":\"...\"}"
        )
        repair_user = {
            "mode": mode,
            "allowed_browser_tools": sorted(list(allowed_tools)) if allowed_tools else [],
            "raw_response": str(raw_content or "")[:12000],
        }
        try:
            raw = self.llm_call(
                task_id=str(task_id or "local"),
                model=str(model),
                messages=[
                    {"role": "system", "content": repair_system},
                    {"role": "user", "content": json.dumps(repair_user, ensure_ascii=False)},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            content = str((((raw or {}).get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "")
            obj = self._parse_json(content)
            normalized = self._normalize_decision(obj, allowed_tools)
            usage = self._normalize_usage(raw)
            model_name = str((raw or {}).get("model") or model or "")
            return normalized, usage, model_name, content
        except Exception:
            return None

    def _fallback(
        self,
        *,
        prompt: str,
        mode: str,
        policy_obs: Dict[str, Any],
        allowed_tools: set[str],
    ) -> Dict[str, Any]:
        def allow(name: str) -> bool:
            return (not allowed_tools) or (name in allowed_tools)

        candidates = policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else []
        partitions = policy_obs.get("candidate_partitions") if isinstance(policy_obs.get("candidate_partitions"), dict) else {}
        local_candidates = partitions.get("local") if isinstance(partitions.get("local"), list) else []
        escape_candidates = partitions.get("escape") if isinstance(partitions.get("escape"), list) else []
        global_candidates = partitions.get("global") if isinstance(partitions.get("global"), list) else []
        memory = policy_obs.get("memory") if isinstance(policy_obs.get("memory"), dict) else {}
        visual_hints = {
            str(x)
            for x in (
                memory.get("visual_element_hints")
                if isinstance(memory.get("visual_element_hints"), list)
                else []
            )
            if str(x).strip()
        }
        typed_candidate_ids = {
            str(x)
            for x in (
                memory.get("typed_candidate_ids")
                if isinstance(memory.get("typed_candidate_ids"), list)
                else []
            )
            if str(x).strip()
        }
        first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        flags = policy_obs.get("flags") if isinstance(policy_obs.get("flags"), dict) else {}
        counters = policy_obs.get("counters") if isinstance(policy_obs.get("counters"), dict) else {}
        loop_level = str(flags.get("loop_level") or "none")
        stall_count = int(counters.get("stall_count") or 0)
        repeat_count = int(counters.get("repeat_action_count") or 0)
        route_like_stuck = mode in {"STUCK", "PLAN"} or loop_level == "high" or stall_count >= 4 or repeat_count >= 4

        def candidate_id(item: Dict[str, Any]) -> str:
            return str(item.get("id") or item.get("element_id") or item.get("_element_id") or "").strip()

        def candidate_index(item: Dict[str, Any]) -> int | None:
            try:
                return int(item.get("index"))
            except (TypeError, ValueError):
                return None

        def browser_action_for_candidate(item: Dict[str, Any]) -> Dict[str, Any] | None:
            role = str(item.get("role") or "").strip().lower()
            selector = item.get("selector") if isinstance(item.get("selector"), dict) else None
            if not isinstance(selector, dict):
                return None
            element_id = candidate_id(item)
            index = candidate_index(item)

            def target_args(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
                args: Dict[str, Any] = {}
                if index is not None:
                    args["index"] = index
                elif element_id:
                    args["element_id"] = element_id
                if extra:
                    args.update(extra)
                return args

            if role in {"input", "textarea"} and element_id and element_id in typed_candidate_ids:
                return None
            if role in {"button", "link"} and allow("browser.click"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.click",
                        "arguments": target_args(),
                    },
                }
            if role in {"input", "textarea"} and allow("browser.click"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.click",
                        "arguments": target_args(),
                    },
                }
            if role == "select" and allow("browser.dropdown_options"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.dropdown_options",
                        "arguments": target_args(),
                    },
                }
            if role == "input" and allow("browser.click"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.click",
                        "arguments": target_args(),
                    },
                }
            return None

        if route_like_stuck and escape_candidates:
            for cand in escape_candidates:
                if isinstance(cand, dict):
                    act = browser_action_for_candidate(cand)
                    if act is not None:
                        return act
        if route_like_stuck:
            if allow("browser.go_back"):
                return {"type": "browser", "tool_call": {"name": "browser.go_back", "arguments": {}}}
        ordered = []
        ordered.extend([cand for cand in local_candidates if isinstance(cand, dict)])
        ordered.extend([cand for cand in escape_candidates if isinstance(cand, dict)])
        ordered.extend([cand for cand in candidates if isinstance(cand, dict)])
        ordered.extend([cand for cand in global_candidates if isinstance(cand, dict)])
        for cand in ordered:
            if not isinstance(cand, dict):
                continue
            if candidate_id(cand) in visual_hints:
                action = browser_action_for_candidate(cand)
                if action is not None:
                    return action
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            action = browser_action_for_candidate(cand)
            if action is not None:
                return action
        if first and allow("browser.click"):
            try:
                first_index = int(first.get("index"))
            except (TypeError, ValueError):
                first_index = None
            if first_index is not None:
                return {"type": "browser", "tool_call": {"name": "browser.click", "arguments": {"index": first_index}}}
            sel = first.get("selector") if isinstance(first.get("selector"), dict) else None
            if sel:
                return {"type": "browser", "tool_call": {"name": "browser.click", "arguments": {"selector": sel}}}
        if allow("browser.wait"):
            return {"type": "browser", "tool_call": {"name": "browser.wait", "arguments": {"time_seconds": 1.0}}}
        if allow("browser.scroll"):
            return {"type": "browser", "tool_call": {"name": "browser.scroll", "arguments": {"direction": "down", "amount": 600}}}
        return {"type": "final", "done": True, "content": "No safe browser action available from allowed_tools."}


__all__ = [name for name in globals() if not name.startswith("__")]

