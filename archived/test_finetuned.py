"""
Quick test of fine-tuned vs base model
"""
import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python test_finetuned.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

print("="*70)
print("TESTING FINE-TUNED PHI-2 FEATURE INTERPRETER")
print("="*70)
print(f"\nImage: {image_path}\n")

print("\n" + "="*70)
print("Running with FINE-TUNED model...")
print("="*70 + "\n")

subprocess.run([
    "python", 
    "explain_features_with_interactions.py", 
    image_path, 
    "--finetuned"
])

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
print("\nTo compare with base model (few-shot), run:")
print(f"  python explain_features_with_interactions.py \"{image_path}\"")
print("="*70)

