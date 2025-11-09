"""
Simple Local Hyperparameter Optimization Example

This example demonstrates basic hyperparameter optimization using the Container Backend
with sequential (non-parallel) trial execution for easier debugging and understanding.

Prerequisites:
    Since you're developing locally, install in development mode:
    
    cd sdk-narayanasabari
    pip install -e '.[docker]' optuna

    For zsh users, quote the brackets:
    pip install -e '.[docker]' optuna

    Ensure Docker or Podman is installed and running.

Note:
    First run will download pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime (~5GB).
    This is a one-time download - subsequent runs will be fast.
"""

try:
    from kubeflow.optimizer import (
        OptimizerClient,
        ContainerBackendConfig,
        Search,
        Objective,
        TrialConfig,
        TrainJobTemplate,
    )
    from kubeflow.trainer import CustomTrainer, Runtime
except ImportError as e:
    print("❌ Import Error!")
    print(f"   {e}")
    print("\n💡 Solution: Install the package in development mode")
    print("   cd sdk-narayanasabari")
    print("   pip install -e '.[docker]' optuna")
    print("\n   For zsh users:")
    print("   pip install -e '.[docker]' optuna")
    exit(1)


def main():
    print("=" * 60)
    print("Simple Local Hyperparameter Optimization")
    print("=" * 60)
    print()

    # Step 1: Check prerequisites
    print("📋 Checking prerequisites...")
    try:
        import docker
        docker_client = docker.from_env()
        docker_client.ping()
        print("✅ Docker runtime is accessible")
    except Exception as e:
        print(f"❌ Docker not available: {e}")
        print("Please install and start Docker, then try again.")
        return

    print("\n📝 Step 1: Configure Container Backend")
    print("-" * 60)
    backend_config = ContainerBackendConfig(
        storage_path="~/.kubeflow/optimizer/simple-example",
        max_parallel_trials=1,
        pull_policy="IfNotPresent",
    )
    print("✅ Sequential execution (max_parallel_trials=1)")
    print(f"Storage path: {backend_config.storage_path}")

    print("\n📝 Step 2: Create OptimizerClient")
    print("-" * 60)
    optimizer = OptimizerClient(backend_config=backend_config)
    print("✅ OptimizerClient created")

    print("\n📝 Step 3: Define Training Function")
    print("-" * 60)
    
    def train_fn(learning_rate: float, batch_size: int):
        """Simple training function that returns a metric based on hyperparameters.
        
        Args:
            learning_rate: Learning rate hyperparameter
            batch_size: Batch size hyperparameter
        """
        import time
        
        print(f"Training with lr={learning_rate}, batch_size={batch_size}")
        time.sleep(2)
        
        accuracy = 0.5 + (0.1 - learning_rate) * 10 + (batch_size / 1000)
        
        print(f"Training complete! Accuracy: {accuracy:.4f}")
        print(f"accuracy: {accuracy:.4f}")
        return accuracy

    trainer = CustomTrainer(
        func=train_fn,
        packages_to_install=["kubeflow"],
    )
    
    template = TrainJobTemplate(trainer=trainer)
    print("✅ Training template defined")

    print("\n📝 Step 4: Configure Search Space")
    print("-" * 60)
    search_space = {
        "learning_rate": Search.uniform(min=0.001, max=0.1),
        "batch_size": Search.choice([32, 64]),
    }
    print("Search space:")
    for param, config in search_space.items():
        print(f"  - {param}: {config}")

    print("\n📝 Step 5: Run Optimization")
    print("-" * 60)
    print("Number of trials: 3")
    print("⏳ Starting optimization...\n")

    try:
        job_name = optimizer.optimize(
            trial_template=template,
            search_space=search_space,
            objectives=[Objective(metric="accuracy", direction="maximize")],
            trial_config=TrialConfig(num_trials=3, parallel_trials=1),
        )
        print(f"\n✅ Optimization job created: {job_name}")

        print("\n📝 Step 6: Get Optimization Results")
        print("-" * 60)
        results = optimizer.get_best_results(name=job_name)
        
        if results:
            print("\n🎯 Best Hyperparameters Found:")
            print("-" * 60)
            for param, value in results.parameters.items():
                print(f"  {param}: {value}")
            
            print("\n📊 Best Metrics:")
            print("-" * 60)
            for metric in results.metrics:
                print(f"  {metric.name}: {metric.latest}")
        else:
            print("⚠️ No successful trials completed")

        print("\n📝 Step 7: View Trial Logs")
        print("-" * 60)
        print("Logs from all trials:\n")
        for log_line in optimizer.get_job_logs(name=job_name):
            print(log_line, end="")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n📝 Step 8: Cleanup")
        print("-" * 60)
        try:
            optimizer.delete_job(name=job_name)
            print("✅ Optimization job deleted")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
