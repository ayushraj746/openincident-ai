from inference_engine.engine import InferenceEngine


def run_experiments(num_episodes=5, difficulty="hard", verbose=True):
    engine = InferenceEngine(difficulty=difficulty)

    all_metrics = []

    print("\n🚀 Starting Inference Runs")
    print(f"Difficulty: {difficulty}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    for i in range(num_episodes):
        print(f"\n===== EPISODE {i + 1} =====")

        try:
            metrics = engine.run_episode(verbose=verbose)
            all_metrics.append(metrics)

        except Exception as e:
            print(f"❌ Error in episode {i+1}: {e}")
            continue

    # ---------------- AGGREGATED METRICS ---------------- #

    if not all_metrics:
        print("⚠️ No successful episodes")
        return

    avg_reward = sum(m["total_reward"] for m in all_metrics) / len(all_metrics)
    avg_steps = sum(m["steps"] for m in all_metrics) / len(all_metrics)
    success_rate = sum(1 for m in all_metrics if m["success"]) / len(all_metrics)

    avg_efficiency = sum(m["efficiency"] for m in all_metrics) / len(all_metrics)
    avg_stability = sum(m["stability_score"] for m in all_metrics) / len(all_metrics)

    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY (AGGREGATED METRICS)")
    print("=" * 60)

    print(f"✅ Success Rate: {round(success_rate * 100, 2)}%")
    print(f"💰 Avg Total Reward: {round(avg_reward, 3)}")
    print(f"⏱️ Avg Steps: {round(avg_steps, 2)}")
    print(f"⚡ Avg Efficiency: {round(avg_efficiency, 3)}")
    print(f"🔁 Avg Stability: {round(avg_stability, 3)}")

    print("=" * 60)


def main():
    # 🔥 CONFIGURATION (easy to tweak)
    config = {
        "num_episodes": 5,
        "difficulty": "hard",
        "verbose": True,
    }

    run_experiments(
        num_episodes=config["num_episodes"],
        difficulty=config["difficulty"],
        verbose=config["verbose"],
    )


if __name__ == "__main__":
    main()