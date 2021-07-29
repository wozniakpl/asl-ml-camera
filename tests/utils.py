def reduce_good_stderr(stderr):
    lines = stderr.splitlines()
    return [line for line in lines if "TensorFlow" not in stderr]


def run_successfully(application):
    result = application.invoke()
    assert result.returncode == 0
    assert len(result.stdout) != 0
    assert len(reduce_good_stderr(result.stderr)) == 0
    return result


def run_unsuccessfully(application):
    result = application.invoke()
    assert result.returncode == 1
    assert len(result.stdout) == 0
    assert len(result.stderr) != 0
    return result
