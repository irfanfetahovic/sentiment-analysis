#!/usr/bin/env python
"""
End-to-End Sentiment Analysis Pipeline

This script orchestrates the complete machine learning pipeline:
1. Data loading and preprocessing
2. Model training (classical and/or transformer)
3. Model evaluation and comparison
4. Results reporting

Run the entire pipeline with a single command.
"""

import argparse
import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


from sentiment_analysis.utils import setup_logging
from sentiment_analysis.settings import settings
from sentiment_analysis.constants import (
    DEFAULT_CONFIG_PATH,
    MODEL_TYPE_TRANSFORMER,
    MODEL_TYPE_CLASSICAL,
)


class PipelineRunner:
    """Orchestrates the complete ML pipeline."""

    def __init__(
        self,
        data_path: str,
        models_dir: str,
        output_dir: str,
        config_path: str = None,
        log_file: str = None,
    ):
        """
        Initialize pipeline runner.

        Args:
            data_path: Path to input data CSV
            models_dir: Directory to save trained models
            output_dir: Directory to save results and reports
            config_path: Path to configuration file
            log_file: Path to log file
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.config_path = config_path
        self.log_file = log_file
        self.scripts_dir = Path(__file__).parent

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Track results
        self.results = {
            "start_time": datetime.now().isoformat(),
            "config": {},
            "training": {},
            "evaluation": {},
            "artifacts": {},
        }

    def run_script(self, script_name: str, args: list) -> tuple:
        """
        Run a Python script with arguments.

        Args:
            script_name: Name of script to run
            args: List of command-line arguments

        Returns:
            Tuple of (return_code, stdout, stderr, duration)
        """
        script_path = self.scripts_dir / script_name
        # Build command, sys.executabe is the Python interpreter
        cmd = [sys.executable, str(script_path)] + args

        print(f"\n{'='*80}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*80}\n")

        start_time = time.time()

        result = subprocess.run(cmd, capture_output=True, text=True)

        duration = time.time() - start_time

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        print(f"\nCompleted in {duration:.2f} seconds")

        if result.returncode != 0:
            print(f"WARNING: Script exited with code {result.returncode}")

        return result.returncode, result.stdout, result.stderr, duration

    def train_classical_model(
        self, max_features: int = 5000, sample_frac: float = 0.1
    ) -> bool:
        """
        Train classical ML model.

        Args:
            max_features: Maximum number of TF-IDF features
            sample_frac: Fraction of data to use

        Returns:
            True if successful
        """
        print("\n" + "=" * 80)
        print("STEP 1: Training Classical Model (Logistic Regression + TF-IDF)")
        print("=" * 80)

        args = [
            "--data-path",
            self.data_path,
            "--models-dir",
            self.models_dir,
            "--max-features",
            str(max_features),
            "--sample-frac",
            str(sample_frac),
        ]

        if self.config_path:
            args.extend(["--config", self.config_path])
        if self.log_file:
            args.extend(["--log-file", self.log_file])

        returncode, stdout, stderr, duration = self.run_script(
            "run_train_classical.py", args
        )

        self.results["training"]["classical"] = {
            "success": returncode == 0,
            "duration": duration,
            "max_features": max_features,
            "sample_frac": sample_frac,
        }

        return returncode == 0

    def train_transformer_model(
        self, num_epochs: int = 3, batch_size: int = 16, sample_frac: float = 0.1
    ) -> bool:
        """
        Train transformer model.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            sample_frac: Fraction of data to use

        Returns:
            True if successful
        """
        print("\n" + "=" * 80)
        print("STEP 2: Training Transformer Model (DistilBERT)")
        print("=" * 80)

        args = [
            "--data-path",
            self.data_path,
            "--output-dir",
            str(Path(self.models_dir) / "distilbert_sentiment"),
            "--num-epochs",
            str(num_epochs),
            "--batch-size",
            str(batch_size),
            "--sample-frac",
            str(sample_frac),
        ]

        if self.config_path:
            args.extend(["--config", self.config_path])
        if self.log_file:
            args.extend(["--log-file", self.log_file])

        returncode, stdout, stderr, duration = self.run_script(
            "run_train_transformer.py", args
        )

        self.results["training"]["transformer"] = {
            "success": returncode == 0,
            "duration": duration,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "sample_frac": sample_frac,
        }

        return returncode == 0

    def evaluate_model(self, model_path: str, model_type: str, model_name: str) -> bool:
        """
        Evaluate a single model.

        Args:
            model_path: Path to model
            model_type: Type of model
            model_name: Name for results

        Returns:
            True if successful
        """
        output_file = str(Path(self.output_dir) / f"{model_type}_evaluation.json")

        args = [
            "--model-path",
            model_path,
            "--model-type",
            model_type,
            "--data-path",
            self.data_path,
            "--output-file",
            output_file,
        ]

        if self.config_path:
            args.extend(["--config", self.config_path])
        if self.log_file:
            args.extend(["--log-file", self.log_file])

        returncode, stdout, stderr, duration = self.run_script("run_evaluate.py", args)

        self.results["evaluation"][model_name] = {
            "success": returncode == 0,
            "duration": duration,
            "results_file": output_file if returncode == 0 else None,
        }

        return returncode == 0

    def compare_models(self, models_config_path: str) -> bool:
        """
        Compare multiple models.

        Args:
            models_config_path: Path to models configuration JSON

        Returns:
            True if successful
        """
        print("\n" + "=" * 80)
        print("STEP 3: Comparing Models")
        print("=" * 80)

        output_file = str(Path(self.output_dir) / "models_comparison.json")

        args = [
            "--compare",
            "--models-json",
            models_config_path,
            "--data-path",
            self.data_path,
            "--output-file",
            output_file,
        ]

        if self.config_path:
            args.extend(["--config", self.config_path])
        if self.log_file:
            args.extend(["--log-file", self.log_file])

        returncode, stdout, stderr, duration = self.run_script("run_evaluate.py", args)

        self.results["evaluation"]["comparison"] = {
            "success": returncode == 0,
            "duration": duration,
            "results_file": output_file if returncode == 0 else None,
        }

        return returncode == 0

    def generate_report(self):
        """Generate final pipeline report."""
        self.results["end_time"] = datetime.now().isoformat()

        report_path = str(Path(self.output_dir) / "pipeline_report.json")
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)

        # Training summary
        if self.results["training"]:
            print("\nTraining:")
            for model_name, info in self.results["training"].items():
                status = "✓" if info["success"] else "✗"
                print(f"  {status} {model_name}: {info['duration']:.2f}s")

        # Evaluation summary
        if self.results["evaluation"]:
            print("\nEvaluation:")
            for model_name, info in self.results["evaluation"].items():
                status = "✓" if info["success"] else "✗"
                print(f"  {status} {model_name}: {info['duration']:.2f}s")
                if info.get("results_file"):
                    print(f"      Results: {info['results_file']}")

        print(f"\nFull report saved to: {report_path}")
        print("=" * 80 + "\n")


def main():
    """Main entry point for pipeline."""
    parser = argparse.ArgumentParser(
        description="Run complete sentiment analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (both models)
  python run_pipeline.py --train-all

  # Train only classical model
  python run_pipeline.py --train-classical

  # Train only transformer model
  python run_pipeline.py --train-transformer

  # Just evaluate existing models
  python run_pipeline.py --skip-training --compare

  # Full pipeline with custom settings
  python run_pipeline.py --train-all --sample-frac 0.1 --num-epochs 2
        """,
    )

    # Training mode
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train both classical and transformer models",
    )
    parser.add_argument(
        "--train-classical", action="store_true", help="Train only classical model"
    )
    parser.add_argument(
        "--train-transformer", action="store_true", help="Train only transformer model"
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training, only evaluate"
    )

    # Evaluation mode
    parser.add_argument(
        "--compare", action="store_true", help="Compare models after training"
    )
    parser.add_argument(
        "--models-json", type=str, help="Path to models config for comparison"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.data_dir / "Reviews.csv"),
        help="Path to data CSV",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.1, help="Fraction of data to use"
    )

    # Classical model arguments
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Max TF-IDF features for classical model",
    )

    # Transformer model arguments
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Training epochs for transformer"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for transformer"
    )

    # Path arguments
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(settings.model_dir),
        help="Directory to save models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.project_root / "results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config file"
    )
    parser.add_argument("--log-file", type=str, help="Path to log file")

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file, level="INFO")

    # Determine what to train
    train_classical = args.train_all or args.train_classical
    train_transformer = args.train_all or args.train_transformer

    if not args.skip_training and not (train_classical or train_transformer):
        print(
            "Error: Must specify --train-all, --train-classical, --train-transformer, or --skip-training"
        )
        sys.exit(1)

    # Initialize pipeline
    pipeline = PipelineRunner(
        data_path=args.data_path,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        log_file=args.log_file,
    )

    print("\n" + "=" * 80)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Data: {args.data_path}")
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample fraction: {args.sample_frac}")
    print("=" * 80)

    # Training phase
    if not args.skip_training:
        if train_classical:
            success = pipeline.train_classical_model(
                max_features=args.max_features, sample_frac=args.sample_frac
            )
            if not success:
                print("WARNING: Classical model training failed")

        if train_transformer:
            success = pipeline.train_transformer_model(
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                sample_frac=args.sample_frac,
            )
            if not success:
                print("WARNING: Transformer model training failed")

    # Comparison phase
    if args.compare:
        models_json = args.models_json or "models_config.json"

        # Create models config if it doesn't exist
        if not Path(models_json).exists():
            models_to_compare = []

            classical_path = str(
                Path(args.models_dir) / "classical_models" / "logistic_tfidf_model.pkl"
            )
            if Path(classical_path).exists():
                models_to_compare.append(
                    {
                        "name": "Logistic Regression + TF-IDF",
                        "path": classical_path,
                        "type": MODEL_TYPE_CLASSICAL,
                    }
                )

            transformer_path = str(Path(args.models_dir) / "distilbert_sentiment")
            if Path(transformer_path).exists():
                models_to_compare.append(
                    {
                        "name": "DistilBERT Fine-tuned",
                        "path": transformer_path,
                        "type": MODEL_TYPE_TRANSFORMER,
                    }
                )

            if models_to_compare:
                with open(models_json, "w") as f:
                    json.dump(models_to_compare, f, indent=2)
                print(f"Created models config: {models_json}")

        if Path(models_json).exists():
            pipeline.compare_models(models_json)
        else:
            print(f"WARNING: Models config not found: {models_json}")

    # Generate final report
    pipeline.generate_report()

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
