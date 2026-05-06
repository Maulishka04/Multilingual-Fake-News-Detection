from pathlib import Path
import pickle


def format_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)


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

        print(f"✅ {filename} loaded: {type(obj)}")
        return True
    except Exception as exc:
        print(f"❌ {filename} error: {exc}")
        return False


def main() -> None:
    models_dir = Path("models")
    model_path = models_dir / "linear_svc_calibrated_tfidf.pkl"
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"

    print("Checking model artifacts")
    print(f"Working directory: {Path.cwd()}")
    print(f"Models directory: {models_dir.resolve()}")

    model_ok = check_pickle(model_path)
    vectorizer_ok = check_pickle(vectorizer_path)

    print("-" * 70)
    print("Summary")
    print(
        f"linear_svc_calibrated_tfidf.pkl: {'VALID' if model_ok else 'INVALID'}"
    )
    print(f"tfidf_vectorizer.pkl: {'VALID' if vectorizer_ok else 'INVALID'}")


if __name__ == "__main__":
    main()
