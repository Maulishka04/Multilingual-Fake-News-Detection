from pathlib import Path
import pickle

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# =========================================================
# HELPERS
# =========================================================

def format_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)


# =========================================================
# CHECK PICKLE FILES
# =========================================================

def check_pickle(file_path: Path) -> bool:

    filename = file_path.name

    print("-" * 70)
    print(f"File: {file_path.resolve()}")

    exists = file_path.exists()

    print(f"Exists: {exists}")

    if not exists:

        print(f"❌ {filename} error: File does not exist")

        return False

    try:

        size_bytes = file_path.stat().st_size
        size_mb = format_size_mb(file_path)

        print(f"Size: {size_bytes} bytes ({size_mb:.2f} MB)")

        if size_bytes == 0:

            print("⚠️ Potentially corrupted: file is 0 bytes")

        elif size_mb < 1:

            print("⚠️ Potentially corrupted: file size is less than 1 MB")

        with file_path.open("rb") as f:

            obj = pickle.load(f)

        print(f"✅ {filename} loaded successfully")
        print(f"Loaded object type: {type(obj)}")

        return True

    except Exception as exc:

        print(f"❌ {filename} error: {exc}")

        return False


# =========================================================
# CHECK MBERT
# =========================================================

def check_mbert(mbert_path: Path) -> bool:

    print("-" * 70)
    print("Checking mBERT model")
    print(f"Folder: {mbert_path.resolve()}")

    if not mbert_path.exists():

        print("❌ mBERT folder does not exist")

        return False

    try:

        required_files = [
            "config.json",
        ]

        print("\nChecking required files:")

        for file_name in required_files:

            file_path = mbert_path / file_name

            if file_path.exists():

                print(f"✅ {file_name}")

            else:

                print(f"❌ Missing: {file_name}")

        print("\nLoading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            str(mbert_path)
        )

        print("✅ mBERT tokenizer loaded successfully")
        print(f"Tokenizer type: {type(tokenizer)}")

        print("\nLoading model...")

        model = AutoModelForSequenceClassification.from_pretrained(
            str(mbert_path)
        )

        print("✅ mBERT model loaded successfully")
        print(f"Model type: {type(model)}")

        return True

    except Exception as exc:

        print(f"❌ mBERT loading failed: {exc}")

        return False


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    models_dir = Path("models")

    model_path = models_dir / "linear_svc_calibrated_tfidf.pkl"

    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"

    mbert_path = models_dir / "mbert_model"

    print("=" * 70)
    print("CHECKING MODEL ARTIFACTS")
    print("=" * 70)

    print(f"Working directory: {Path.cwd()}")
    print(f"Models directory: {models_dir.resolve()}")

    # =====================================================
    # CHECK SVM + TFIDF
    # =====================================================

    print("\n")
    print("=" * 70)
    print("CHECKING CLASSICAL ML ARTIFACTS")
    print("=" * 70)

    model_ok = check_pickle(model_path)

    vectorizer_ok = check_pickle(vectorizer_path)

    # =====================================================
    # CHECK MBERT
    # =====================================================

    print("\n")
    print("=" * 70)
    print("CHECKING TRANSFORMER MODEL")
    print("=" * 70)

    mbert_ok = check_mbert(mbert_path)

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"linear_svc_calibrated_tfidf.pkl: "
        f"{'✅ VALID' if model_ok else '❌ INVALID'}"
    )

    print(
        f"tfidf_vectorizer.pkl: "
        f"{'✅ VALID' if vectorizer_ok else '❌ INVALID'}"
    )

    print(
        f"mBERT model: "
        f"{'✅ VALID' if mbert_ok else '❌ INVALID'}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()