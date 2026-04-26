import json
from datetime import datetime

from inference_engine.engine import InferenceEngine


# ---------------- RUN MODE ---------------- #

def run_mode(mode="rl", num_episodes=5, difficulty="hard", verbose=True):

    use_rl = (mode == "rl")

    engine = InferenceEngine(
        difficulty=difficulty,
        use_rl=use_rl   # ✅ FIXED
    )

    all_metrics = []
    trajectories = []

    print("\n🚀 RUN MODE:", mode.upper())
    print(f"Difficulty: {difficulty} | Episodes: {num_episodes}")
    print("=" * 60)

    for i in range(num_episodes):
        print(f"\n===== EPISODE {i + 1} =====")

        try:
            metrics, trajectory = engine.run_episode(
                verbose=verbose,
                return_trajectory=True
            )

            all_metrics.append(metrics)
            trajectories.append(trajectory)

        except Exception as e:
            print(f"❌ Error in episode {i+1}: {e}")
            continue

    return all_metrics, trajectories


# ---------------- SUMMARY ---------------- #

def summarize(metrics_list):

    if not metrics_list:
        return {}

    avg_reward = sum(m["total_reward"] for m in metrics_list) / len(metrics_list)
    avg_steps = sum(m["steps"] for m in metrics_list) / len(metrics_list)
    success_rate = sum(1 for m in metrics_list if m["success"]) / len(metrics_list)

    avg_efficiency = sum(m["efficiency"] for m in metrics_list) / len(metrics_list)
    avg_stability = sum(m["stability_score"] for m in metrics_list) / len(metrics_list)

    avg_recovery = sum(m.get("recovery_speed", 0) for m in metrics_list) / len(metrics_list)

    return {
        "avg_reward": round(avg_reward, 3),
        "avg_steps": round(avg_steps, 2),
        "success_rate": round(success_rate, 3),
        "efficiency": round(avg_efficiency, 3),
        "stability": round(avg_stability, 3),
        "recovery_speed": round(avg_recovery, 3),
    }


# ---------------- COMPARISON ---------------- #

def compare_modes(num_episodes=5, difficulty="hard"):

    rl_metrics, _ = run_mode("rl", num_episodes, difficulty, verbose=False)
    rule_metrics, _ = run_mode("rule", num_episodes, difficulty, verbose=False)

    rl_summary = summarize(rl_metrics)
    rule_summary = summarize(rule_metrics)

    print("\n" + "=" * 60)
    print("📊 RL vs RULE COMPARISON")
    print("=" * 60)

    for key in rl_summary:
        print(f"{key.upper():<20} RL={rl_summary[key]} | RULE={rule_summary[key]}")

    return {
        "rl": rl_summary,
        "rule": rule_summary,
    }


# ---------------- SAVE ---------------- #

def save_results(results, filename="results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {filename}")


# ---------------- MAIN ---------------- #

def main():

    config = {
        "num_episodes": 3,
        "difficulty": "medium",
    }

    results = compare_modes(
        num_episodes=config["num_episodes"],
        difficulty=config["difficulty"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_results(results, f"results_{timestamp}.json")


if __name__ == "__main__":
    main()