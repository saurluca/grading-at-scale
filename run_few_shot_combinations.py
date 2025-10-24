#!/usr/bin/env python3
"""
Run few-shot evaluation for all combinations of model configuration parameters.

This script runs few-shot.py with all combinations of:
- with_prompt: True/False
- pass_reference: True/False  
- pass_reference_answer: True/False

Total: 2^3 = 8 different configurations
"""

import itertools
import subprocess
import sys
from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent

def run_configuration(with_prompt: bool, pass_reference: bool, pass_reference_answer: bool):
    """Run few-shot.py with specified configuration."""
    
    config_name = f"prompt_{with_prompt}_ref_{pass_reference}_refans_{pass_reference_answer}"
    print("=" * 80)
    print(f"Running configuration: {config_name}")
    print(f"  with_prompt: {with_prompt}")
    print(f"  pass_reference: {pass_reference}")
    print(f"  pass_reference_answer: {pass_reference_answer}")
    print("=" * 80)
    
    # Load the base config
    config_path = PROJECT_ROOT / "configs" / "few_shot.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Modify the configuration
    cfg.model.with_prompt = with_prompt
    cfg.model.pass_reference = pass_reference
    cfg.model.pass_reference_answer = pass_reference_answer
    
    # Save temporary config
    temp_config_path = PROJECT_ROOT / "configs" / f"few_shot_temp_{config_name}.yaml"
    OmegaConf.save(cfg, temp_config_path)
    
    try:
        # Run the few-shot script with the temporary config
        # We'll modify few-shot.py to accept a config file path via environment variable
        env = {
            "FEW_SHOT_CONFIG": str(temp_config_path),
            **dict(subprocess.os.environ)
        }
        
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "evaluation" / "few-shot.py")],
            env=env,
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR: Configuration {config_name} failed with return code {result.returncode}")
            return False
        else:
            print(f"SUCCESS: Configuration {config_name} completed")
            return True
            
    finally:
        # Clean up temporary config
        if temp_config_path.exists():
            temp_config_path.unlink()
    
    print()
    

def main():
    """Run all combinations of configuration parameters."""
    
    print("Starting few-shot evaluation sweep across all parameter combinations")
    print()
    
    # Define all parameter combinations
    with_prompt_options = [True, False]
    pass_reference_options = [True, False]
    pass_reference_answer_options = [True, False]
    
    # Generate all combinations
    combinations = list(itertools.product(
        with_prompt_options,
        pass_reference_options,
        pass_reference_answer_options
    ))
    
    total_combinations = len(combinations)
    print(f"Total combinations to run: {total_combinations}")
    print()
    
    # Track results
    results = []
    
    # Run each combination
    for idx, (with_prompt, pass_reference, pass_reference_answer) in enumerate(combinations, 1):
        print(f"\n[{idx}/{total_combinations}] Running combination...")
        
        success = run_configuration(with_prompt, pass_reference, pass_reference_answer)
        
        results.append({
            "with_prompt": with_prompt,
            "pass_reference": pass_reference,
            "pass_reference_answer": pass_reference_answer,
            "success": success
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r["success"])
    failed = total_combinations - successful
    
    print(f"Total runs: {total_combinations}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("Failed configurations:")
        for r in results:
            if not r["success"]:
                print(f"  - with_prompt={r['with_prompt']}, "
                      f"pass_reference={r['pass_reference']}, "
                      f"pass_reference_answer={r['pass_reference_answer']}")
    
    print("\nAll configurations completed!")
    

if __name__ == "__main__":
    main()

