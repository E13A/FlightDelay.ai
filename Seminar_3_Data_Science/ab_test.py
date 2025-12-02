import numpy as np
import pandas as pd
import random

class BanditRouter:
    def __init__(self, arms, epsilon=0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms} # Average reward
        self.history = []

    def select_arm(self):
        if random.random() < self.epsilon:
            # Explore
            return random.choice(self.arms)
        else:
            # Exploit
            # Find max value
            max_val = max(self.values.values())
            # Handle ties randomly
            best_arms = [arm for arm, val in self.values.items() if val == max_val]
            return random.choice(best_arms)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        # Incremental average update
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm] = new_value
        self.history.append({'arm': arm, 'reward': reward, 'new_value': new_value})

def simulate_ab_test(n_trials=1000):
    print("Simulating A/B Test (Multi-Armed Bandit)...")
    arms = ['Model_A', 'Model_B', 'Baseline']
    router = BanditRouter(arms, epsilon=0.1)
    
    # Simulation:
    # Baseline: 10% conversion
    # Model_A: 12% conversion
    # Model_B: 15% conversion
    true_conversion = {
        'Baseline': 0.10,
        'Model_A': 0.12,
        'Model_B': 0.15
    }
    
    results = []
    for i in range(n_trials):
        arm = router.select_arm()
        # Simulate reward (1 = conversion, 0 = no conversion)
        reward = 1 if random.random() < true_conversion[arm] else 0
        router.update(arm, reward)
        results.append({'trial': i, 'arm': arm, 'reward': reward})
        
    print("Final Estimated Values (Conversion Rates):")
    for arm in arms:
        print(f"{arm}: {router.values[arm]:.4f} (True: {true_conversion[arm]})")
        
    print(f"Total Traffic per Arm: {router.counts}")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('sprint_3/ab_test_results.csv', index=False)
    print("A/B Test simulation complete.")

if __name__ == "__main__":
    simulate_ab_test()
