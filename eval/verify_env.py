"""验证 τ-bench retail 环境：手动执行 ground truth 工具动作，确认评分比对逻辑。
不实例化完整环境（避免加载用户模拟器），直接组装 data/tools/tasks。
"""
from tau_bench.envs.retail.data import load_data
from tau_bench.envs.retail.tools import ALL_TOOLS
from tau_bench.envs.retail.tasks_test import TASKS_TEST
from tau_bench.envs.base import RESPOND_ACTION_NAME, to_hashable, consistent_hash

TASK_INDEX = 70
TERMINATE = "transfer_to_human_agents"

tools_map = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
task = TASKS_TEST[TASK_INDEX]
print("=" * 70)
print("任务指令:", task.instruction[:160])
print("=" * 70)

tool_actions = [
    a for a in task.actions
    if a.name not in (RESPOND_ACTION_NAME, TERMINATE)
]
print("需要的工具动作:", [a.name for a in tool_actions])
print("-" * 70)

# 模拟 agent 按 ground truth 执行
agent_data = load_data()
for action in tool_actions:
    obs = tools_map[action.name].invoke(data=agent_data, **action.kwargs)
    print(f"[调用 {action.name}] -> {str(obs)[:100]}")
agent_hash = consistent_hash(to_hashable(agent_data))

# 独立重放 ground truth，得到基准 hash
gt_data = load_data()
for action in tool_actions:
    tools_map[action.name].invoke(data=gt_data, **action.kwargs)
gt_hash = consistent_hash(to_hashable(gt_data))

print("-" * 70)
print("agent 数据 hash:", agent_hash)
print("基准 数据 hash:", gt_hash)
print("结论:", "PASS（环境与评分逻辑正常）" if agent_hash == gt_hash else "FAIL")
