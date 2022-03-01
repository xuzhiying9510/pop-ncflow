#! /usr/bin/env python

import numpy as np
import os
import pickle
import sys

sys.path.append("..")

from benchmark_helpers import (
    NCFLOW_HYPERPARAMS,
    PATH_FORM_HYPERPARAMS,
    get_args_and_problems,
    print_,
)
from lib.algorithms import NcfEpi, Objective, POP
from lib.problem import Problem
from lib.graph_utils import check_feasibility
from lib.graph_utils import compute_in_or_out_flow

TOP_DIR = "pop-vs-ncflow-logs"
OUTPUT_CSV_TEMPLATE = "pop-vs-ncflow-{}.csv"

HEADERS = [
    "k",
    "s_k",
    "t_k",
    "d_k",
    "ncflow_out_flow",
    "pop_out_flow",
    "pop_minus_ncflow",
]
PLACEHOLDER = ",".join("{}" for _ in HEADERS)


def run_pop(args, formatted_fname_substr, problem):
    _, topo_fname, tm_fname = problem
    obj = args.obj
    num_subproblems = args.num_subproblems
    split_method = args.split_method
    split_fraction = args.split_fraction

    problem = Problem.from_file(topo_fname, tm_fname)
    print(problem.name, tm_fname)

    num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS

    pop = POP(
        objective=Objective.get_obj_from_str(obj),
        num_subproblems=num_subproblems,
        split_method=split_method,
        split_fraction=split_fraction,
        num_paths=num_paths,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        DEBUG=True,
    )
    pop.solve(problem)
    print("{}: {}".format(obj, pop.obj_val))
    sol_dict = pop.sol_dict
    check_feasibility(problem, [sol_dict])
    with open("pop-sol-dict-{}.pkl".format(formatted_fname_substr), "wb") as w:
        pickle.dump(sol_dict, w)
    return sol_dict


def run_ncflow(args, formatted_fname_substr, problem):
    _, topo_fname, tm_fname = problem
    obj = args.obj
    problem = Problem.from_file(topo_fname, tm_fname)
    print(problem.name, tm_fname)

    (
        num_paths,
        edge_disjoint,
        dist_metric,
        partition_cls,
        num_parts_scale_factor,
    ) = NCFLOW_HYPERPARAMS[problem.name]
    num_partitions_to_set = num_parts_scale_factor * int(np.sqrt(len(problem.G.nodes)))
    partitioner = partition_cls(num_partitions_to_set)

    ncflow = NcfEpi(
        objective=Objective.get_obj_from_str(obj),
        num_paths=num_paths,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
        DEBUG=True,
        VERBOSE=False,
    )
    ncflow.solve(problem, partitioner)
    print("{}: {}".format(obj, ncflow.obj_val))
    sol_dict = ncflow.sol_dict
    check_feasibility(problem, [sol_dict])
    with open(
        "ncflow-sol-dict-{}.pkl".format(formatted_fname_substr),
        "wb",
    ) as w:
        pickle.dump(sol_dict, w)
    return sol_dict


def run_pop_vs_ncflow(args, formatted_fname_substr, problem):
    output_csv = OUTPUT_CSV_TEMPLATE.format(formatted_fname_substr)
    with open(output_csv, "a") as results:
        print_(",".join(HEADERS), file=results)
        pop_sol_dict = run_pop(args, formatted_fname_substr, problem)
        ncflow_sol_dict = run_ncflow(args, formatted_fname_substr, problem)

        for commod_key, pop_flow_list in pop_sol_dict.items():
            k, (s_k, t_k, d_k) = commod_key
            if d_k == 0:
                continue
            ncflow_flow_list = ncflow_sol_dict[commod_key]
            ncflow_out_flow = compute_in_or_out_flow(ncflow_flow_list, 0, {s_k})
            pop_out_flow = compute_in_or_out_flow(pop_flow_list, 0, {s_k})
            result_line = PLACEHOLDER.format(
                k,
                s_k,
                t_k,
                d_k,
                ncflow_out_flow,
                pop_out_flow,
                pop_out_flow - ncflow_out_flow,
            )
            print_(result_line, file=results)


if __name__ == "__main__":
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args, formatted_fname, problems = get_args_and_problems(
        "{}-{}",
        [
            [
                "--num-subproblems",
                {
                    "type": int,
                    "choices": [1, 2, 4, 8, 16, 32, 64],
                    "required": True,
                    "help": "Number of subproblems to use",
                },
            ],
            [
                "--split-method",
                {
                    "type": str,
                    "choices": ["random", "means", "tailored", "skewed", "covs"],
                    "required": True,
                    "help": "Split method to use",
                },
            ],
            [
                "--split-fraction",
                {
                    "type": float,
                    "choices": [0, 0.25, 0.5, 0.75, 1.0],
                    "required": True,
                    "help": "Split fractions to use",
                },
            ],
        ],
        many_problems=False,
    )
    assert len(problems) == 1
    run_pop_vs_ncflow(args, formatted_fname, problems[0])
