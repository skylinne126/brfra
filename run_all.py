"""
一键跑通脚本：仿真实验 → 图表 → 附录代码截图 → Word 报告。

用法
----
    python run_all.py                 # 快速跑通（30 名受试者，约 1 分钟）
    python run_all.py --full          # 论文中的完整队列（120 名受试者，约 3 分钟）
    python run_all.py --n 60          # 自定义人数
    python run_all.py --skip-report   # 只跑实验，不生成 Word 报告
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_URL = "https://github.com/skylinne126/brfra"


def run(cmd: list) -> None:
    """执行子命令（输出直接透传到当前终端）。"""
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"命令执行失败（退出码 {result.returncode}）：{' '.join(map(str, cmd))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="一键运行平衡能力与跌倒风险评估复现实验")
    parser.add_argument("--n", type=int, default=30, help="受试者人数（默认 30）")
    parser.add_argument("--full", action="store_true", help="使用完整队列（120 人）")
    parser.add_argument("--seed", type=int, default=20240908, help="随机种子")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL,
                        help="写入报告标题下方的仓库地址")
    parser.add_argument("--skip-report", action="store_true",
                        help="只运行实验与图表，跳过 Word 报告生成")
    args = parser.parse_args()

    n = 120 if args.full else args.n
    run([sys.executable, "main.py", "--n", n, "--seed", args.seed])

    if not args.skip_report:
        run([sys.executable, "tools/render_code_shots.py"])
        run([sys.executable, "tools/build_report.py", "--repo-url", args.repo_url])

    print("\n全部完成。结果位于 output/ 目录，Word 报告位于项目根目录。")


if __name__ == "__main__":
    main()
