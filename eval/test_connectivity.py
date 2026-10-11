"""连通性测试：验证百炼 Qwen3-4B 和 DeepSeek 两个 API key 是否有效。"""
import litellm

def load_env(path=".env"):
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env

env = load_env()
litellm.drop_params = True

tests = [
    ("百炼 Qwen3-4B (Agent)", "dashscope/qwen3-4b-instruct", env["DASHSCOPE_API_KEY"]),
    ("DeepSeek (用户模拟器)", "deepseek/deepseek-chat", env["DEEPSEEK_API_KEY"]),
]

for name, model, key in tests:
    try:
        r = litellm.completion(
            model=model,
            api_key=key,
            messages=[{"role": "user", "content": "回复两个字：正常"}],
            max_tokens=20,
            timeout=30,
        )
        print(f"[OK] {name}: {r.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"[失败] {name}: {str(e)[:200]}")
