import os

def make_dir(path):
    os.makedirs(path, exist_ok=True)
    gitkeep = os.path.join(path, ".gitkeep")
    if not os.path.exists(gitkeep):
        open(gitkeep, "w").close()


def create_project():
    print("\n🚀 Creating PRODUCTION-GRADE MLOps project structure...\n")

    folders = [
        "src/data",
        "src/model",
        "src/utils",
        "sagemaker/processing",
        "sagemaker/training",
        "sagemaker/evaluation",
        "sagemaker/pipelines",
        "config",
        "scripts"
    ]

    for f in folders:
        make_dir(f)
        print(f"📁 {f}")

    # root placeholder files
    files = {
        "README.md": "# Production MLOps Project\n",
        "config/params.yaml": "test_size: 0.2\nmodel_type: logistic_regression\n",
        "config/sagemaker_config.yaml": "# SageMaker settings\n",
        "scripts/run_local_pipeline.py": "# Execute full pipeline locally\n",
    }

    for fpath, content in files.items():
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                f.write(content)
            print(f"📝 {fpath}")

    print("\n✅ Structure created.")
    print("👉 Now place your existing scripts into src/data/ and src/model/")


if __name__ == "__main__":
    create_project()
