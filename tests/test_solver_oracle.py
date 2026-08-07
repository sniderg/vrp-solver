from scripts.run_solver_oracle import build_oracle_command


def test_oracle_command_preserves_constructor_only_arguments() -> None:
    command = build_oracle_command(
        executable="Solver.exe",
        instance="Z:\\input.xml",
        output="Z:\\output.xml",
        seed=7,
        time_limit=120,
        iterations=0,
        workers=1,
        run_id="cold-v212",
    )

    assert command == [
        "Solver.exe",
        "-p",
        "Z:\\input.xml",
        "-o",
        "Z:\\output.xml",
        "-s",
        "7",
        "-t",
        "120",
        "-iter",
        "0",
        "-j",
        "1",
        "-id",
        "cold-v212",
    ]


def test_oracle_command_appends_config_without_changing_other_arguments() -> None:
    command = build_oracle_command(
        executable="Solver.exe",
        instance="Z:\\input.xml",
        output="Z:\\output.xml",
        seed=7,
        time_limit=120,
        iterations=0,
        workers=1,
        run_id="cold-v212",
        config="Z:\\constructor.cfg",
    )

    assert command[-2:] == ["-cfg", "Z:\\constructor.cfg"]
