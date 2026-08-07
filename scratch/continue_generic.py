import sys, time
sys.path.insert(0, "src")

def main():
    name = sys.argv[1]
    checkpoint = sys.argv[2]
    out = sys.argv[3]
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    from vrp_solver.xml_io import load_instance, load_solution, save_solution
    from vrp_solver.solver.surgical_search import SurgicalSearchConfig, surgical_search
    from vrp_solver.rules import validate_solution
    from vrp_solver.inventory import tank_aggregates

    inst = load_instance(f"roadef_2016_data/set_B/Instances_B_V25-11042016/{name}.xml")
    # Native checkpoint: produced by native-solve from instance data only.
    sol = load_solution(checkpoint)
    end_day = max(1, (inst.horizon * inst.unit + 1439) // 1440)
    for round_index in range(rounds):
        errors = sum(v.severity == "error" for v in validate_solution(inst, sol))
        _, neg, over, safety = tank_aggregates(inst, sol)
        print(f"continue_round,{round_index},errors,{errors},safety_qm,{safety:.0f},neg_qm,{neg:.0f}", flush=True)
        if errors == 0:
            break
        sol, steps = surgical_search(
            inst, sol,
            config=SurgicalSearchConfig(
                end_day=end_day, iterations=400, candidates_per_move=120,
                seed=200 + round_index, time_limit_seconds=600, workers=1,
                no_improvement_limit=10_000,
                first_operator=(
                    None, "pressure_band_resource_block", "multiroute_pressure_block",
                    "recombine_route_blocks", "create_shift", "insert_operation",
                )[round_index % 6],
                output_xml=out,
            ),
            progress=None,
        )
        save_solution(sol, out)
    errors = sum(v.severity == "error" for v in validate_solution(inst, sol))
    save_solution(sol, out)
    print(f"final_errors,{errors}", flush=True)

if __name__ == "__main__":
    main()
