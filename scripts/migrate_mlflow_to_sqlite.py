#!/usr/bin/env python3
"""
Migrate MLflow data from file-based storage to SQLite backend.

This script reads all experiments and runs from the file-based mlruns directory
and migrates them to a SQLite database, preserving all metadata, parameters,
metrics, tags, and artifacts.
"""

import os
import sys
from pathlib import Path
import shutil
from typing import Optional

import mlflow
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def migrate_experiments_and_runs(
    file_store_path: str,
    sqlite_db_path: str,
    artifacts_dir: Optional[str] = None,
):
    """
    Migrate MLflow experiments and runs from file store to SQLite store.

    Args:
        file_store_path: Path to the file-based MLflow store (mlruns directory)
        sqlite_db_path: Path to the SQLite database file
        artifacts_dir: Optional directory for artifacts (defaults to mlruns/../mlartifacts)
    """
    print(f"Starting migration from {file_store_path} to {sqlite_db_path}")

    # Ensure SQLite database directory exists
    sqlite_db_path = Path(sqlite_db_path)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up tracking URIs
    file_tracking_uri = f"file://{os.path.abspath(file_store_path)}"
    sqlite_tracking_uri = f"sqlite:///{os.path.abspath(sqlite_db_path)}"

    print(f"Source (file store): {file_tracking_uri}")
    print(f"Destination (SQLite): {sqlite_tracking_uri}")

    # Create clients for both stores
    file_client = MlflowClient(tracking_uri=file_tracking_uri)
    sqlite_client = MlflowClient(tracking_uri=sqlite_tracking_uri)

    # Get all experiments from file store
    print("\nFetching experiments from file store...")
    try:
        file_experiments = file_client.search_experiments()
        print(f"Found {len(file_experiments)} experiments")
    except Exception as e:
        print(f"Error fetching experiments: {e}")
        return False

    if not file_experiments:
        print("No experiments found to migrate.")
        return True

    # Migrate each experiment
    experiment_id_mapping = {}  # Map old experiment IDs to new ones
    total_runs = 0

    for exp in file_experiments:
        print(f"\nMigrating experiment: {exp.name} (ID: {exp.experiment_id})")

        # Create experiment in SQLite store (or get existing)
        try:
            # Try to get existing experiment by name
            try:
                sqlite_exp = sqlite_client.get_experiment_by_name(exp.name)
                print(f"  Experiment '{exp.name}' already exists in SQLite store")
                new_exp_id = sqlite_exp.experiment_id
            except:
                # Create new experiment
                new_exp_id = sqlite_client.create_experiment(
                    name=exp.name,
                    tags=exp.tags,
                    artifact_location=exp.artifact_location,
                )
                print(f"  Created experiment '{exp.name}' with ID: {new_exp_id}")
        except Exception as e:
            print(f"  Error creating experiment: {e}")
            continue

        experiment_id_mapping[exp.experiment_id] = new_exp_id

        # Get all runs for this experiment
        print(f"  Fetching runs for experiment '{exp.name}'...")
        try:
            runs = file_client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=10000,  # Large number to get all runs
            )
            print(f"  Found {len(runs)} runs")
        except Exception as e:
            print(f"  Error fetching runs: {e}")
            continue

        # Migrate each run
        for run in runs:
            total_runs += 1
            run_id = run.info.run_id
            print(f"    Migrating run {total_runs}: {run_id[:8]}...")

            try:
                # Set tracking URI to SQLite for this run
                mlflow.set_tracking_uri(sqlite_tracking_uri)
                mlflow.set_experiment(exp.name)
                
                # Create run in SQLite store using mlflow.start_run()
                with mlflow.start_run(
                    experiment_id=new_exp_id,
                    run_name=run.info.run_name,
                ) as new_run:
                    new_run_id = new_run.info.run_id
                    
                    # Log parameters
                    if run.data.params:
                        mlflow.log_params(run.data.params)

                    # Log metrics
                    if run.data.metrics:
                        for metric_key, metric_value in run.data.metrics.items():
                            # Get metric history to preserve all values
                            try:
                                metric_history = file_client.get_metric_history(
                                    run_id, metric_key
                                )
                                for metric_entry in metric_history:
                                    mlflow.log_metric(
                                        metric_key,
                                        metric_entry.value,
                                        timestamp=metric_entry.timestamp,
                                        step=metric_entry.step,
                                    )
                            except Exception as e:
                                # Fallback to single value if history unavailable
                                mlflow.log_metric(metric_key, metric_value)

                    # Log tags (excluding system tags that are set automatically)
                    system_tags = {"mlflow.runName", "mlflow.user", "mlflow.source.name", "mlflow.source.type"}
                    for tag_key, tag_value in run.data.tags.items():
                        if tag_key not in system_tags:
                            mlflow.set_tag(tag_key, tag_value)

                    # Handle artifacts - copy artifact directory structure
                    if artifacts_dir:
                        source_artifacts = Path(artifacts_dir) / run_id / "artifacts"
                        if source_artifacts.exists():
                            # MLflow will handle artifacts through the artifact store
                            # We'll copy them to the new location
                            dest_artifacts = (
                                Path(sqlite_db_path).parent
                                / "mlartifacts"
                                / new_run_id
                                / "artifacts"
                            )
                            if source_artifacts.exists():
                                dest_artifacts.parent.mkdir(parents=True, exist_ok=True)
                                if dest_artifacts.exists():
                                    shutil.rmtree(dest_artifacts)
                                shutil.copytree(source_artifacts, dest_artifacts)
                                print(f"      Copied artifacts")
                
                # Update run status after the run context exits
                if run.info.status != RunStatus.RUNNING:
                    sqlite_client.set_terminated(
                        new_run_id,
                        status=run.info.status,
                        end_time=run.info.end_time,
                    )

            except Exception as e:
                print(f"      Error migrating run {run_id[:8]}: {e}")
                import traceback

                traceback.print_exc()
                continue

    print(f"\n\nMigration complete!")
    print(f"  Migrated {len(experiment_id_mapping)} experiments")
    print(f"  Migrated {total_runs} runs")
    return True


def main():
    """Main migration function."""
    project_root = PROJECT_ROOT
    mlruns_dir = project_root / "mlruns"
    sqlite_db_path = project_root / "mlflow.db"
    artifacts_dir = project_root / "mlartifacts"

    if not mlruns_dir.exists():
        print(f"Error: mlruns directory not found at {mlruns_dir}")
        sys.exit(1)

    print("=" * 60)
    print("MLflow File Store to SQLite Migration")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Source (file store): {mlruns_dir}")
    print(f"Destination (SQLite): {sqlite_db_path}")
    print(f"Artifacts directory: {artifacts_dir}")
    print("=" * 60)

    # Check if SQLite database already exists
    if sqlite_db_path.exists():
        response = input(
            f"\nSQLite database already exists at {sqlite_db_path}. "
            "Do you want to continue? This may add duplicate data. (y/N): "
        )
        if response.lower() != "y":
            print("Migration cancelled.")
            sys.exit(0)

    # Perform migration
    success = migrate_experiments_and_runs(
        str(mlruns_dir),
        str(sqlite_db_path),
        str(artifacts_dir) if artifacts_dir.exists() else None,
    )

    if success:
        print("\n" + "=" * 60)
        print("Migration completed successfully!")
        print("=" * 60)
        
        # Backup the old mlruns directory
        backup_dir = project_root / "mlruns.backup"
        if mlruns_dir.exists():
            if backup_dir.exists():
                response = input(
                    f"\nBackup directory {backup_dir} already exists. "
                    "Do you want to remove it and create a new backup? (y/N): "
                )
                if response.lower() == "y":
                    shutil.rmtree(backup_dir)
                else:
                    print("Skipping backup (backup directory already exists).")
                    return
            
            print(f"\nBacking up old mlruns directory to {backup_dir}...")
            try:
                shutil.move(str(mlruns_dir), str(backup_dir))
                print(f"✓ Successfully backed up mlruns to {backup_dir}")
            except Exception as e:
                print(f"⚠ Warning: Could not backup mlruns directory: {e}")
                print(f"  You may want to manually backup {mlruns_dir} to {backup_dir}")
        
        print(f"\nNext steps:")
        print(f"1. Verify that MLflow can access the new SQLite store")
        print(f"2. Test creating a new run to ensure everything works")
        print(f"3. The old mlruns directory has been backed up to: {backup_dir}")
        print("=" * 60)
    else:
        print("\nMigration failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

