#!/usr/bin/env python3

"""
Kubeflow pipeline for electronics parts country of origin prediction
with full MLflow experiment tracking.
"""

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics


MLFLOW_SETUP = """
import os, json

def _setup_mlflow():
    config_str = os.environ.get('KFP_MLFLOW_CONFIG', '')
    if not config_str:
        print("[MLflow] No KFP_MLFLOW_CONFIG found - skipping MLflow logging")
        return None

    config = json.loads(config_str)
    import mlflow

    os.environ['MLFLOW_TRACKING_URI'] = config['endpoint']
    os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

    token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    if os.path.exists(token_path):
        with open(token_path) as f:
            os.environ['MLFLOW_TRACKING_TOKEN'] = f.read().strip()

    run_id = config.get('parentRunId')
    print(f"[MLflow] Connected to {config['endpoint']}, run_id={run_id}")
    return run_id
"""


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["pandas", "numpy", "scikit-learn", "python-dotenv", "boto3", "mlflow"],
)
def load_and_prepare_data(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    data_filename: str,
    processed_data: Output[Dataset],
    label_encoder: Output[Model],
    labels: Output[Dataset],
    descriptions: Output[Dataset],
):
    """Load and prepare data for the embedding pipeline"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import pickle
    import boto3
    import os
    import json

    # --- MLflow setup ---
    config_str = os.environ.get('KFP_MLFLOW_CONFIG', '')
    mlflow_run_id = None
    if config_str:
        config = json.loads(config_str)
        import mlflow
        os.environ['MLFLOW_TRACKING_URI'] = config['endpoint']
        os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        if os.path.exists(token_path):
            with open(token_path) as f:
                os.environ['MLFLOW_TRACKING_TOKEN'] = f.read().strip()
        mlflow_run_id = config.get('parentRunId')
        print(f"[MLflow] Connected, run_id={mlflow_run_id}")

    # Setup MinIO client
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False,
    )

    # Download data from MinIO
    local_data_path = "/tmp/input_data.csv"
    s3_client.download_file(bucket_name, f"data/{data_filename}", local_data_path)

    # Load the synthetic electronics data
    df = pd.read_csv(local_data_path)

    print(f"Loaded {len(df)} records from MinIO bucket {bucket_name}")
    print(f"Columns: {list(df.columns)}")

    # Prepare labels
    label_encoder_obj = LabelEncoder()
    y = label_encoder_obj.fit_transform(df["Country_Of_Origin"])
    class_names = label_encoder_obj.classes_

    print(f"Number of countries: {len(class_names)}")
    print(f"Countries: {list(class_names)}")

    # --- MLflow logging ---
    if mlflow_run_id:
        import mlflow
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_params({
                "data_filename": data_filename,
                "bucket_name": bucket_name,
                "minio_endpoint": minio_endpoint.split("//")[-1],
            })
            mlflow.log_metrics({
                "dataset/num_records": len(df),
                "dataset/num_features": len(df.columns),
                "dataset/num_classes": len(class_names),
            })

            class_dist = df["Country_Of_Origin"].value_counts()
            for country, count in class_dist.items():
                safe_name = str(country).replace(" ", "_").replace(".", "")[:50]
                mlflow.log_metric(f"dataset/class_{safe_name}", count)

            mlflow.set_tags({
                "pipeline_step": "load_and_prepare_data",
                "dataset_columns": ", ".join(df.columns.tolist()),
            })

            local_data_path_copy = "/tmp/dataset_sample.csv"
            df.head(50).to_csv(local_data_path_copy, index=False)
            mlflow.log_artifact(local_data_path_copy, "data")

        print("[MLflow] Logged dataset params and metrics")

    # Save processed data and encoder
    df.to_csv(processed_data.path, index=False)

    with open(label_encoder.path, "wb") as f:
        pickle.dump(label_encoder_obj, f)

    with open(labels.path, "wb") as f:
        pickle.dump(y, f)

    # Save descriptions for embedding generation
    descriptions_list = df["Part_Description"].tolist()
    with open(descriptions.path, "wb") as f:
        pickle.dump(descriptions_list, f)

    print("Data preparation complete")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["numpy", "python-dotenv", "requests", "mlflow"],
)
def generate_embeddings(
    descriptions: Input[Dataset],
    endpoint: str,
    embedding_model: str,
    api_key: str,
    embeddings: Output[Dataset],
):
    """Generate embeddings using vLLM BGE-Large model"""
    import pickle
    import numpy as np
    import requests
    import json
    import os
    import time

    # --- MLflow setup ---
    config_str = os.environ.get('KFP_MLFLOW_CONFIG', '')
    mlflow_run_id = None
    if config_str:
        config = json.loads(config_str)
        import mlflow
        os.environ['MLFLOW_TRACKING_URI'] = config['endpoint']
        os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        if os.path.exists(token_path):
            with open(token_path) as f:
                os.environ['MLFLOW_TRACKING_TOKEN'] = f.read().strip()
        mlflow_run_id = config.get('parentRunId')
        print(f"[MLflow] Connected, run_id={mlflow_run_id}")

    # Load descriptions
    with open(descriptions.path, "rb") as f:
        descriptions_list = pickle.load(f)

    print(f"Generating embeddings for {len(descriptions_list)} descriptions...")

    # Create embeddings via API call
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    embeddings_list = []
    latencies = []

    start_total = time.time()
    for i, desc in enumerate(descriptions_list):
        payload = {"model": embedding_model, "input": desc}

        t0 = time.time()
        response = requests.post(
            f"{endpoint}/v1/embeddings", headers=headers, json=payload, verify=False
        )
        latencies.append(time.time() - t0)

        if response.status_code == 200:
            result = response.json()
            embedding = result["data"][0]["embedding"]
            embeddings_list.append(embedding)
        else:
            print(f"Error generating embedding: {response.status_code}")
            raise Exception(f"Failed to generate embedding for: {desc[:50]}...")

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(descriptions_list)} embeddings...")

    total_time = time.time() - start_total

    if embeddings_list:
        X = np.array(embeddings_list)
        print(f"Generated embeddings with shape: {X.shape}")

        # --- MLflow logging ---
        if mlflow_run_id:
            import mlflow
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_params({
                    "embedding_model": embedding_model,
                    "embedding_endpoint": endpoint.split("//")[-1].split(".")[0],
                })
                mlflow.log_metrics({
                    "embeddings/num_samples": len(embeddings_list),
                    "embeddings/dimension": X.shape[1],
                    "embeddings/total_time_sec": round(total_time, 2),
                    "embeddings/avg_latency_ms": round(np.mean(latencies) * 1000, 2),
                    "embeddings/p50_latency_ms": round(np.median(latencies) * 1000, 2),
                    "embeddings/p95_latency_ms": round(np.percentile(latencies, 95) * 1000, 2),
                    "embeddings/p99_latency_ms": round(np.percentile(latencies, 99) * 1000, 2),
                    "embeddings/throughput_per_sec": round(len(embeddings_list) / total_time, 2),
                    "embeddings/mean_norm": round(float(np.linalg.norm(X, axis=1).mean()), 4),
                    "embeddings/std_norm": round(float(np.linalg.norm(X, axis=1).std()), 4),
                })
                mlflow.set_tag("pipeline_step", "generate_embeddings")
            print("[MLflow] Logged embedding metrics and latencies")

        # Save embeddings
        with open(embeddings.path, "wb") as f:
            pickle.dump(X, f)

        print("Embeddings saved")
    else:
        raise Exception("Failed to generate embeddings")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["numpy", "scikit-learn", "joblib", "mlflow"],
)
def train_model(
    embeddings: Input[Dataset],
    labels: Input[Dataset],
    label_encoder: Input[Model],
    trained_model: Output[Model],
    test_results: Output[Dataset],
    metrics: Output[Metrics],
):
    """Train KNN classifier on embeddings"""
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix,
    )
    import joblib
    import json
    import os
    import time

    # --- MLflow setup ---
    config_str = os.environ.get('KFP_MLFLOW_CONFIG', '')
    mlflow_run_id = None
    if config_str:
        config = json.loads(config_str)
        import mlflow
        os.environ['MLFLOW_TRACKING_URI'] = config['endpoint']
        os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        if os.path.exists(token_path):
            with open(token_path) as f:
                os.environ['MLFLOW_TRACKING_TOKEN'] = f.read().strip()
        mlflow_run_id = config.get('parentRunId')
        print(f"[MLflow] Connected, run_id={mlflow_run_id}")

    # Load embeddings and labels
    with open(embeddings.path, "rb") as f:
        X = pickle.load(f)

    with open(labels.path, "rb") as f:
        y = pickle.load(f)

    with open(label_encoder.path, "rb") as f:
        label_encoder_obj = pickle.load(f)

    class_names = label_encoder_obj.classes_

    print(f"Dataset info:")
    print(f"   - Total samples: {len(X)}")
    print(f"   - Embedding dimension: {X.shape[1]}")
    print(f"   - Number of countries: {len(class_names)}")

    # Split data for training and testing
    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Split data: {len(X_train)} train, {len(X_test)} test")

    # Train KNN classifier
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    print(f"Training KNN classifier with k={k}...")

    t0 = time.time()
    knn.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"KNN training complete in {train_time:.2f}s")

    # Evaluate model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Extended metrics
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Cross-validation
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

    print(f"Model Performance:")
    print(f"   - Accuracy: {accuracy:.1%}")
    print(f"   - F1 (macro): {f1_macro:.4f}")
    print(f"   - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # --- MLflow logging ---
    if mlflow_run_id:
        import mlflow
        from mlflow.models import infer_signature

        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_params({
                "model_type": "KNeighborsClassifier",
                "n_neighbors": k,
                "test_size": test_size,
                "random_state": random_state,
                "cv_folds": 5,
                "embedding_dimension": X.shape[1],
            })

            mlflow.log_metrics({
                "train/num_samples": len(X_train),
                "train/training_time_sec": round(train_time, 4),
                "test/num_samples": len(X_test),
                "test/accuracy": accuracy,
                "test/precision_macro": precision_macro,
                "test/recall_macro": recall_macro,
                "test/f1_macro": f1_macro,
                "test/precision_weighted": precision_weighted,
                "test/recall_weighted": recall_weighted,
                "test/f1_weighted": f1_weighted,
                "cv/accuracy_mean": cv_scores.mean(),
                "cv/accuracy_std": cv_scores.std(),
            })

            for i, cv_acc in enumerate(cv_scores):
                mlflow.log_metric("cv/fold_accuracy", cv_acc, step=i)

            for cls_name in class_names:
                safe = str(cls_name).replace(" ", "_").replace(".", "")[:50]
                if cls_name in report:
                    mlflow.log_metrics({
                        f"class/{safe}/precision": report[cls_name]["precision"],
                        f"class/{safe}/recall": report[cls_name]["recall"],
                        f"class/{safe}/f1": report[cls_name]["f1-score"],
                        f"class/{safe}/support": report[cls_name]["support"],
                    })

            cm = confusion_matrix(y_test, y_pred)
            cm_dict = {"labels": class_names.tolist(), "matrix": cm.tolist()}
            cm_path = "/tmp/confusion_matrix.json"
            with open(cm_path, "w") as f:
                json.dump(cm_dict, f, indent=2)
            mlflow.log_artifact(cm_path, "evaluation")

            report_path = "/tmp/classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred, target_names=class_names))
            mlflow.log_artifact(report_path, "evaluation")

            try:
                signature = infer_signature(X_test[:5], knn.predict(X_test[:5]))
                mlflow.sklearn.log_model(
                    knn, "knn_model",
                    signature=signature,
                    registered_model_name="coo-knn-classifier",
                )
                print("[MLflow] Logged sklearn model with signature")
            except Exception as e:
                print(f"[MLflow] Could not log sklearn model: {e}")
                mlflow.log_artifact(trained_model.path, "model")

            mlflow.set_tags({
                "pipeline_step": "train_model",
                "model_type": "KNeighborsClassifier",
                "num_classes": str(len(class_names)),
            })

        print("[MLflow] Logged training metrics, per-class metrics, CV scores, and model")

    # Save model
    joblib.dump(knn, trained_model.path)

    # Save test data for evaluation
    test_data = {
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "accuracy": accuracy,
    }

    with open(test_results.path, "wb") as f:
        pickle.dump(test_data, f)

    # Log KFP metrics too
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("f1_macro", f1_macro)
    metrics.log_metric("precision_macro", precision_macro)
    metrics.log_metric("recall_macro", recall_macro)
    metrics.log_metric("test_samples", len(X_test))
    metrics.log_metric("num_classes", len(class_names))
    metrics.log_metric("cv_accuracy_mean", float(cv_scores.mean()))

    print("Model training complete")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["matplotlib", "seaborn", "scikit-learn", "numpy", "boto3", "mlflow"],
)
def evaluate_model(
    test_results: Input[Dataset],
    label_encoder: Input[Model],
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    confusion_matrix_plot: Output[Dataset],
):
    """Evaluate model and generate visualizations"""
    import pickle
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    import boto3
    import os
    import json

    # --- MLflow setup ---
    config_str = os.environ.get('KFP_MLFLOW_CONFIG', '')
    mlflow_run_id = None
    if config_str:
        config = json.loads(config_str)
        import mlflow
        os.environ['MLFLOW_TRACKING_URI'] = config['endpoint']
        os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        if os.path.exists(token_path):
            with open(token_path) as f:
                os.environ['MLFLOW_TRACKING_TOKEN'] = f.read().strip()
        mlflow_run_id = config.get('parentRunId')
        print(f"[MLflow] Connected, run_id={mlflow_run_id}")

    # Load test results and label encoder
    with open(test_results.path, "rb") as f:
        test_data = pickle.load(f)

    with open(label_encoder.path, "rb") as f:
        label_encoder_obj = pickle.load(f)

    y_test = np.array(test_data["y_test"])
    y_pred = np.array(test_data["y_pred"])
    accuracy = test_data["accuracy"]

    class_names = label_encoder_obj.classes_

    print(f"Model Performance:")
    print(f"   - Accuracy: {accuracy:.1%}")

    # Show detailed results
    print("\nDetailed Classification Report:")
    report_str = classification_report(y_test, y_pred, target_names=class_names)
    print(report_str)

    # Setup MinIO client
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False,
    )

    artifacts_for_mlflow = []

    # Create confusion matrix
    if len(class_names) > 1:
        cm = confusion_matrix(y_test, y_pred)

        # --- Confusion matrix heatmap ---
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix - Country of Origin Prediction")
        plt.xlabel("Predicted Country")
        plt.ylabel("Actual Country")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        local_plot_path = "/tmp/confusion_matrix.png"
        plt.savefig(local_plot_path, dpi=150, bbox_inches="tight")
        plt.savefig(confusion_matrix_plot.path, dpi=150, bbox_inches="tight")
        plt.close()
        artifacts_for_mlflow.append(local_plot_path)

        # --- Normalized confusion matrix ---
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Country")
        plt.ylabel("Actual Country")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        norm_plot_path = "/tmp/confusion_matrix_normalized.png"
        plt.savefig(norm_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        artifacts_for_mlflow.append(norm_plot_path)

        # --- Per-class accuracy bar chart ---
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        per_class_f1 = [report_dict[c]["f1-score"] for c in class_names]
        per_class_prec = [report_dict[c]["precision"] for c in class_names]
        per_class_rec = [report_dict[c]["recall"] for c in class_names]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.25
        ax.bar(x - width, per_class_prec, width, label='Precision', color='#2196F3')
        ax.bar(x, per_class_rec, width, label='Recall', color='#FF9800')
        ax.bar(x + width, per_class_f1, width, label='F1-Score', color='#4CAF50')
        ax.set_xlabel('Country')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Precision / Recall / F1')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        bar_plot_path = "/tmp/per_class_metrics.png"
        plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        artifacts_for_mlflow.append(bar_plot_path)

        # Upload to MinIO
        s3_client.upload_file(local_plot_path, bucket_name, "outputs/confusion_matrix.png")
        print("Confusion matrix saved to MinIO")

    # Save evaluation report to MinIO
    local_report_path = "/tmp/evaluation_report.txt"
    with open(local_report_path, "w") as f:
        f.write(f"Model Performance:\n")
        f.write(f"Accuracy: {accuracy:.1%}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report_str)
    artifacts_for_mlflow.append(local_report_path)

    s3_client.upload_file(
        local_report_path, bucket_name, "outputs/evaluation_report.txt"
    )

    # --- MLflow logging ---
    if mlflow_run_id:
        import mlflow
        with mlflow.start_run(run_id=mlflow_run_id):
            for artifact_path in artifacts_for_mlflow:
                mlflow.log_artifact(artifact_path, "evaluation")

            mlflow.set_tag("pipeline_step", "evaluate_model")

        print(f"[MLflow] Logged {len(artifacts_for_mlflow)} evaluation artifacts")

    print("Evaluation complete - results saved to MinIO outputs folder")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "requests",
        "boto3",
        "mlflow",
    ],
)
def predict_on_unseen_data(
    trained_model: Input[Model],
    label_encoder: Input[Model],
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    unseen_data_filename: str,
    predictions_output_filename: str,
    endpoint: str,
    embedding_model: str,
    api_key: str,
):
    """Load unseen data, generate embeddings, make predictions, and save results to MinIO"""
    import pandas as pd
    import numpy as np
    import pickle
    import joblib
    import requests
    import json
    import boto3
    import os
    import time

    # --- MLflow setup ---
    config_str = os.environ.get('KFP_MLFLOW_CONFIG', '')
    mlflow_run_id = None
    if config_str:
        config = json.loads(config_str)
        import mlflow
        os.environ['MLFLOW_TRACKING_URI'] = config['endpoint']
        os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        if os.path.exists(token_path):
            with open(token_path) as f:
                os.environ['MLFLOW_TRACKING_TOKEN'] = f.read().strip()
        mlflow_run_id = config.get('parentRunId')
        print(f"[MLflow] Connected, run_id={mlflow_run_id}")

    # Setup MinIO client
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False,
    )

    # Download unseen data from MinIO
    local_data_path = "/tmp/unseen_data.csv"
    s3_client.download_file(
        bucket_name, f"data/{unseen_data_filename}", local_data_path
    )

    # Load the unseen data
    df = pd.read_csv(local_data_path)
    print(f"Loaded {len(df)} unseen records for prediction")
    print(f"Columns: {list(df.columns)}")

    # Extract descriptions for embedding generation
    descriptions_list = df["Part_Description"].tolist()
    print(f"Generating embeddings for {len(descriptions_list)} descriptions...")

    # Generate embeddings via API call
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    embeddings_list = []
    latencies = []

    start_total = time.time()
    for desc in descriptions_list:
        payload = {"model": embedding_model, "input": desc}

        t0 = time.time()
        response = requests.post(
            f"{endpoint}/v1/embeddings", headers=headers, json=payload, verify=False
        )
        latencies.append(time.time() - t0)

        if response.status_code == 200:
            result = response.json()
            embedding = result["data"][0]["embedding"]
            embeddings_list.append(embedding)
        else:
            print(f"Error generating embedding: {response.status_code}")
            raise Exception(f"Failed to generate embedding for: {desc[:50]}...")

    if not embeddings_list:
        raise Exception("Failed to generate embeddings for unseen data")

    inference_time = time.time() - start_total

    X_unseen = np.array(embeddings_list)
    print(f"Generated embeddings with shape: {X_unseen.shape}")

    # Load trained model and label encoder
    knn_model = joblib.load(trained_model.path)

    with open(label_encoder.path, "rb") as f:
        label_encoder_obj = pickle.load(f)

    # Make predictions
    print("Making predictions...")
    t0 = time.time()
    y_pred_encoded = knn_model.predict(X_unseen)
    predict_time = time.time() - t0
    y_pred_labels = label_encoder_obj.inverse_transform(y_pred_encoded)

    # Get prediction probabilities/distances for confidence scoring
    distances, _ = knn_model.kneighbors(X_unseen)
    confidence_scores = 1 / (
        1 + distances.mean(axis=1)
    )

    print(f"Predictions complete for {len(y_pred_labels)} samples")

    # Create predictions dataframe
    predictions_df = df.copy()
    predictions_df["Predicted_Country_Of_Origin"] = y_pred_labels
    predictions_df["Confidence_Score"] = confidence_scores

    # Save predictions locally first
    local_predictions_path = "/tmp/predictions.csv"
    predictions_df.to_csv(local_predictions_path, index=False)

    # Upload predictions to MinIO
    s3_client.upload_file(
        local_predictions_path, bucket_name, f"data/{predictions_output_filename}"
    )

    country_dist = pd.Series(y_pred_labels).value_counts()

    print(f"Predictions saved to MinIO: data/{predictions_output_filename}")
    print(f"Predicted countries distribution:")
    for country, count in country_dist.head(10).items():
        print(f"   - {country}: {count}")

    # --- MLflow logging ---
    if mlflow_run_id:
        import mlflow
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_params({
                "unseen_data_filename": unseen_data_filename,
                "predictions_output_filename": predictions_output_filename,
            })

            mlflow.log_metrics({
                "predict/num_samples": len(y_pred_labels),
                "predict/num_unique_countries": len(country_dist),
                "predict/confidence_mean": round(float(confidence_scores.mean()), 4),
                "predict/confidence_median": round(float(np.median(confidence_scores)), 4),
                "predict/confidence_min": round(float(confidence_scores.min()), 4),
                "predict/confidence_max": round(float(confidence_scores.max()), 4),
                "predict/confidence_std": round(float(confidence_scores.std()), 4),
                "predict/low_confidence_count": int((confidence_scores < 0.5).sum()),
                "predict/high_confidence_count": int((confidence_scores > 0.8).sum()),
                "predict/embedding_time_sec": round(inference_time, 2),
                "predict/avg_embedding_latency_ms": round(np.mean(latencies) * 1000, 2),
                "predict/model_predict_time_sec": round(predict_time, 4),
            })

            for country, count in country_dist.items():
                safe = str(country).replace(" ", "_").replace(".", "")[:50]
                mlflow.log_metric(f"predict/country_{safe}", count)

            mlflow.log_artifact(local_predictions_path, "predictions")

            mlflow.set_tag("pipeline_step", "predict_on_unseen_data")

        print("[MLflow] Logged prediction metrics, confidence stats, and predictions.csv")


@pipeline(
    name="electronics-embedding-pipeline",
    description="Electronics parts country of origin prediction using BGE-Large embeddings",
)
def electronics_embedding_pipeline(
    minio_endpoint: str = "",
    minio_access_key: str = "",
    minio_secret_key: str = "",
    bucket_name: str = "pipeline",
    data_filename: str = "synthetic_electronics_parts_1k.csv",
    endpoint: str = "",
    embedding_model: str = "",
    api_key: str = "",
    unseen_data_filename: str = "unseen_electronics_parts.csv",
    predictions_output_filename: str = "predictions_output.csv",
):
    """Main pipeline for electronics parts classification"""

    # Step 1: Load and prepare data
    load_task = load_and_prepare_data(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        data_filename=data_filename,
    )

    # Step 2: Generate embeddings
    embed_task = generate_embeddings(
        descriptions=load_task.outputs["descriptions"],
        endpoint=endpoint,
        embedding_model=embedding_model,
        api_key=api_key,
    )

    # Step 3: Train model
    train_task = train_model(
        embeddings=embed_task.outputs["embeddings"],
        labels=load_task.outputs["labels"],
        label_encoder=load_task.outputs["label_encoder"],
    )

    # Step 4: Evaluate model
    eval_task = evaluate_model(
        test_results=train_task.outputs["test_results"],
        label_encoder=load_task.outputs["label_encoder"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
    )

    # Step 5: Make predictions on unseen data
    predict_task = predict_on_unseen_data(
        trained_model=train_task.outputs["trained_model"],
        label_encoder=load_task.outputs["label_encoder"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        unseen_data_filename=unseen_data_filename,
        predictions_output_filename=predictions_output_filename,
        endpoint=endpoint,
        embedding_model=embedding_model,
        api_key=api_key,
    )


if __name__ == "__main__":
    # This allows the pipeline to be compiled
    pass
