import os
import json

from train import train


def main():
    data_root = "./data/COCO2017"
    results_dir = "results"
    sched_types = ["annealed", "learnable", "adaptive"]

    os.makedirs(results_dir, exist_ok=True)
    summary = {}

    for sched in sched_types:
        print(f"\n=== Running scheduler: {sched} ===")
        metrics = train(
            data_root=data_root,
            scheduler_type=sched,
            image_size=256,
            batch_size=32,
            num_epochs=12,
            device="cuda",
            run_eval=True,
            n_eval_samples=5000,
            results_dir=results_dir,
        )
        summary[sched] = metrics

    with open(
        os.path.join(results_dir, "summary_metrics.json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)
    print("\nAll schedulers finished. Summary metrics saved.")def main():
    data_root = "./data/COCO2017"
    results_dir = "results"
    base_sched_types = ["annealed", "learnable", "adaptive"]

    os.makedirs(results_dir, exist_ok=True)
    summary = {}

    # 基础三种 scheduler
    for sched in base_sched_types:
        print(f"\n=== Running scheduler: {sched} ===")
        metrics = train(
            data_root=data_root,
            scheduler_type=sched,
            image_size=256,
            batch_size=32,
            num_epochs=12,
            device="cuda",
            run_eval=True,
            n_eval_samples=5000,
            results_dir=results_dir,
        )
        summary[sched] = metrics

    # LearnableMonotoneScheduler 的 k_bins ablation
    learnable_k_values = [8, 16, 32]
    for k in learnable_k_values:
        name = f"learnable_k{k}"
        print(f"\n=== Running scheduler: {name} ===")
        metrics = train(
            data_root=data_root,
            scheduler_type="learnable",
            image_size=256,
            batch_size=32,
            num_epochs=12,
            device="cuda",
            run_eval=True,
            n_eval_samples=5000,
            results_dir=results_dir,
            scheduler_kwargs={"k_bins": k},
        )
        summary[name] = metrics

    # AdaptiveAnnealedScheduler 的 hidden_dim ablation
    adaptive_hidden_values = [16, 32, 64]
    for h in adaptive_hidden_values:
        name = f"adaptive_h{h}"
        print(f"\n=== Running scheduler: {name} ===")
        metrics = train(
            data_root=data_root,
            scheduler_type="adaptive",
            image_size=256,
            batch_size=32,
            num_epochs=15,
            device="cuda",
            run_eval=True,
            n_eval_samples=5000,
            results_dir=results_dir,
            scheduler_kwargs={"hidden_dim": h},
        )
        summary[name] = metrics

    with open(
        os.path.join(results_dir, "summary_metrics.json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)
    print("\nAll schedulers and ablations finished. Summary metrics saved.")



if __name__ == "__main__":
    main()


