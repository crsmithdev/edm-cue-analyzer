import argparse
import inspect

import pytest

# Skip tests if the CLI module cannot be imported (avoids pulling heavy deps)
cli = pytest.importorskip("edm_cue_analyzer.cli")


def test_add_common_args_parses_options():
    p = argparse.ArgumentParser()
    cli._add_common_args(p)

    args = p.parse_args(
        [
            "dummy.mp3",
            "-o",
            "out.xml",
            "-a",
            "bpm",
            "-j",
            "2",
            "--bpm-precision",
            "1",
            "--log-file",
            "log.txt",
        ]
    )

    assert args.input == ["dummy.mp3"]
    assert args.output.name == "out.xml"
    assert args.analyses == ["bpm"]
    assert args.jobs == 2
    assert args.bpm_precision == 1
    assert args.log_file.name == "log.txt"


def test_analyze_functions_are_async():
    # Ensure the CLI exposes the expected async functions without executing them
    assert inspect.iscoroutinefunction(cli.analyze_track)
    assert inspect.iscoroutinefunction(cli.batch_analyze)
