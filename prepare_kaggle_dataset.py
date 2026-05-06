"""
Dataset Preparation Script for Kaggle
"""

import pandas as pd
import os

print("\n" + "="*60)
print("📊 DATASET PREPARATION FOR KAGGLE")
print("="*60 + "\n")

print("📁 Loading existing dataset...")
try:
    df = pd.read_csv('dataset/unified_cleaned_dataset.csv')
    print(f"✅ Loaded: {len(df)} rows")
except FileNotFoundError:
    print("❌ Error: unified_cleaned_dataset.csv not found!")
    exit(1)

print(f"📋 Columns: {df.columns.tolist()}")

print("\n🔍 Preparing data...")
# ✅ CHANGE: 'text' को 'clean_text' में बदल दो
df_clean = df[['clean_text', 'label']].dropna()
print(f"✅ Cleaned: {len(df_clean)} rows")

print(f"\n📈 Label distribution:")
print(df_clean['label'].value_counts())

# ✅ CHANGE: Rename करो 'clean_text' को 'text' में (Kaggle के लिए)
df_clean = df_clean.rename(columns={'clean_text': 'text'})

output_file = 'fake_news_dataset_for_mbert.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n✅ Dataset saved: {output_file}")

file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"📦 File size: {file_size_mb:.2f} MB")

print("\n" + "="*60)
print("✅ DATASET READY FOR KAGGLE!")
print("="*60 + "\n")