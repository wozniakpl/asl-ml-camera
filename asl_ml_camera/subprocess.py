import subprocess


def run_subprocess(args):
    return subprocess.run(
        args,
        check=False,
        encoding="utf8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
