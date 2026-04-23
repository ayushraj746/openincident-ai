from inference_engine.engine import InferenceEngine


def main():
    # initialize inference engine
    engine = InferenceEngine(difficulty="hard")

    # number of episodes to run (for demo)
    num_episodes = 3

    for i in range(num_episodes):
        print(f"\n===== EPISODE {i+1} =====")
        engine.run_episode(verbose=True)


if __name__ == "__main__":
    main()