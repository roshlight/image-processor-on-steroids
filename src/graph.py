from pathlib import Path
import shutil

from langgraph.graph import START, END, StateGraph

from src.executor import execute_action
from src.planner import call_planner
from src.reporting import save_logs
from src.state import EditAgentState
from src.validator import parse_plan, validate_plan


def prepare_input_node(state: EditAgentState) -> EditAgentState:
    logs = state.get("logs", [])

    input_image_path = state["input_image_path"]
    output_dir = state["output_dir"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logs.append(
        {
            "node": "prepare_input",
            "status": "ok",
            "input_image_path": input_image_path,
            "user_instruction": state["user_instruction"],
            "output_dir": output_dir,
        }
    )

    return {
        **state,
        "current_image_path": input_image_path,
        "current_step": 0,
        "intermediate_images": [],
        "last_mask_path": None,
        "final_image_path": None,
        "logs": logs,
        "error": None,
    }


def planner_node(state: EditAgentState) -> EditAgentState:
    logs = state.get("logs", [])

    raw_plan = call_planner(
        user_instruction=state["user_instruction"],
        image_path=state["current_image_path"],
    )

    logs.append(
        {
            "node": "planner",
            "status": "ok",
            "raw_plan": raw_plan,
        }
    )

    return {
        **state,
        "raw_plan": raw_plan,
        "logs": logs,
    }


def parse_plan_node(state: EditAgentState) -> EditAgentState:
    logs = state.get("logs", [])

    try:
        plan = parse_plan(state["raw_plan"])
        actions = plan.get("actions", [])

        logs.append(
            {
                "node": "parse_plan",
                "status": "ok",
                "plan": plan,
            }
        )

        return {
            **state,
            "plan": plan,
            "actions": actions,
            "logs": logs,
            "error": None,
        }

    except Exception as exc:
        logs.append(
            {
                "node": "parse_plan",
                "status": "failed",
                "error": str(exc),
            }
        )

        return {
            **state,
            "plan": {},
            "actions": [],
            "logs": logs,
            "error": str(exc),
        }


def validate_plan_node(state: EditAgentState) -> EditAgentState:
    logs = state.get("logs", [])

    is_valid, error = validate_plan(state["plan"])

    logs.append(
        {
            "node": "validate_plan",
            "status": "ok" if is_valid else "failed",
            "error": error,
        }
    )

    return {
        **state,
        "logs": logs,
        "error": error,
    }


def execute_next_tool_node(state: EditAgentState) -> EditAgentState:
    logs = state.get("logs", [])

    step = state["current_step"]
    actions = state["actions"]
    action = actions[step]

    try:
        result = execute_action(
            current_image_path=state["current_image_path"],
            output_dir=state["output_dir"],
            step_idx=step + 1,
            action=action,
            last_mask_path=state.get("last_mask_path"),
        )

        output_image_path = result["output_image_path"]
        last_mask_path = result["last_mask_path"]

        intermediate_images = list(state.get("intermediate_images", []))

        if output_image_path != state["current_image_path"]:
            intermediate_images.append(output_image_path)

        logs.append(
            {
                "node": "execute_next_tool",
                "status": "ok",
                "step": step + 1,
                "action": action,
                "result": result,
            }
        )

        return {
            **state,
            "current_image_path": output_image_path,
            "last_mask_path": last_mask_path,
            "current_step": step + 1,
            "intermediate_images": intermediate_images,
            "logs": logs,
            "error": None,
        }

    except Exception as exc:
        logs.append(
            {
                "node": "execute_next_tool",
                "status": "failed",
                "step": step + 1,
                "action": action,
                "error": str(exc),
            }
        )

        return {
            **state,
            "logs": logs,
            "error": str(exc),
        }


def save_final_node(state: EditAgentState) -> EditAgentState:
    logs = state.get("logs", [])

    output_dir = Path(state["output_dir"])
    final_path = output_dir / "final.png"

    shutil.copyfile(state["current_image_path"], final_path)

    logs.append(
        {
            "node": "save_final",
            "status": "ok",
            "final_image_path": str(final_path),
        }
    )

    saved_logs = save_logs(logs, str(output_dir))

    return {
        **state,
        "final_image_path": str(final_path),
        "logs": logs,
        "error": None,
        **saved_logs,
    }


def handle_error_node(state: EditAgentState) -> EditAgentState:
    logs = state.get("logs", [])

    logs.append(
        {
            "node": "handle_error",
            "status": "failed",
            "error": state.get("error"),
        }
    )

    save_logs(logs, state["output_dir"])

    return {
        **state,
        "logs": logs,
    }


def route_after_validation(state: EditAgentState) -> str:
    if state.get("error"):
        return "error"

    if len(state.get("actions", [])) == 0:
        return "finish"

    return "execute"


def route_after_execution(state: EditAgentState) -> str:
    if state.get("error"):
        return "error"

    if state["current_step"] < len(state["actions"]):
        return "continue"

    return "finish"


def build_graph():
    graph = StateGraph(EditAgentState)

    graph.add_node("prepare_input", prepare_input_node)
    graph.add_node("planner", planner_node)
    graph.add_node("parse_plan", parse_plan_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("execute_next_tool", execute_next_tool_node)
    graph.add_node("save_final", save_final_node)
    graph.add_node("handle_error", handle_error_node)

    graph.add_edge(START, "prepare_input")
    graph.add_edge("prepare_input", "planner")
    graph.add_edge("planner", "parse_plan")
    graph.add_edge("parse_plan", "validate_plan")

    graph.add_conditional_edges(
        "validate_plan",
        route_after_validation,
        {
            "execute": "execute_next_tool",
            "finish": "save_final",
            "error": "handle_error",
        },
    )

    graph.add_conditional_edges(
        "execute_next_tool",
        route_after_execution,
        {
            "continue": "execute_next_tool",
            "finish": "save_final",
            "error": "handle_error",
        },
    )

    graph.add_edge("save_final", END)
    graph.add_edge("handle_error", END)

    return graph.compile()
