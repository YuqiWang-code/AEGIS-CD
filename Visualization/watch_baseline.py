"""
轮询服务器 baseline 训练进度；四个数据集全部生成 .run_complete 后，
自动下载日志并重新生成全部可视化（loss + 混淆矩阵 + 对比图），然后退出。

用法（后台运行）:
    python Visualization/watch_baseline.py
"""

import os
import sys
import time
import subprocess

import paramiko

# 服务器凭据从 .vscode/sftp.json 读取（该文件不在 git 中），避免把密码写进仓库。
_SFTP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     ".vscode", "sftp.json")
try:
    import json
    with open(_SFTP, "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
    HOST, USER, PWD = _cfg["host"], _cfg["username"], _cfg["password"]
except Exception as e:  # noqa: BLE001
    sys.exit(f"无法读取服务器凭据 {_SFTP}: {e}")

BASE = "/home/hzeng/project/ZH/AIC/saved_models/baseline"
DS = ["LEVIR-CD-256", "WHU-CD-256", "SYSU-CD-256", "CDD-CD-256"]
LOCAL = "Visualization/logs/baseline"
POLL = 600  # 秒


def _ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=22, username=USER, password=PWD, timeout=30)
    return c


def done_list():
    try:
        c = _ssh()
        cmd = "for d in " + " ".join(DS) + "; do test -f " + BASE + "/$d/.run_complete && echo $d; done"
        _, out, _ = c.exec_command(cmd, timeout=60)
        done = out.read().decode().strip().split()
        c.close()
        return done
    except Exception as e:  # noqa: BLE001
        print(f"[{time.strftime('%H:%M:%S')}] SSH 异常: {e}", flush=True)
        return None


def finalize():
    print("== baseline 全部完成，开始下载日志 ==", flush=True)
    c = _ssh()
    sftp = c.open_sftp()
    for ds in DS:
        os.makedirs(os.path.join(LOCAL, ds), exist_ok=True)
        for fn in ("trainValLog.txt", "run_config.json"):
            try:
                sftp.get(f"{BASE}/{ds}/{fn}", os.path.join(LOCAL, ds, fn))
                print(f"  已下载 {ds}/{fn}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  下载失败 {ds}/{fn}: {e}", flush=True)
    sftp.close()
    c.close()

    print("== 重新生成可视化 ==", flush=True)
    r = subprocess.run(
        [sys.executable, "Visualization/generate_visualizations.py",
         "--results-dir", LOCAL, "--what", "all"],
        capture_output=True, text=True,
    )
    if r.stdout:
        print(r.stdout, flush=True)
    if r.returncode != 0 and r.stderr:
        print(r.stderr, flush=True)
    print("== FINALIZE DONE ==", flush=True)


def main():
    print("watcher 启动：每 10 分钟轮询一次 baseline 完成状态", flush=True)
    while True:
        done = done_list()
        if done is None:
            time.sleep(POLL)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] 已完成 {len(done)}/4: {done}", flush=True)
        if len(done) >= 4:
            finalize()
            sys.exit(0)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
