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
    print("\nAll schedulers finished. Summary metrics saved.")


if __name__ == "__main__":
    main()
