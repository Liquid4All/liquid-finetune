import sys
from datetime import datetime

import yaml


def _resolve_kuberay_cluster_spec(kuberay_cfg: dict) -> dict:
    gpu_type = kuberay_cfg.get("gpu_type", "nvidia.com/gpu")

    worker_replicas = kuberay_cfg.get("worker_replicas")
    gpus_per_worker = kuberay_cfg.get("gpus_per_worker")
    legacy_gpu_count = kuberay_cfg.get("gpu_count")

    if worker_replicas is None:
        if legacy_gpu_count is None:
            raise ValueError(
                "kuberay.worker_replicas is required. "
                "For backward compatibility, kuberay.gpu_count is treated as the "
                "number of 1-GPU worker pods."
            )
        worker_replicas = int(legacy_gpu_count)

    if gpus_per_worker is None:
        gpus_per_worker = 1

    worker_replicas = int(worker_replicas)
    gpus_per_worker = int(gpus_per_worker)
    head_gpu_count = int(kuberay_cfg.get("head_gpu_count", 0))

    if worker_replicas < 1:
        raise ValueError(f"kuberay.worker_replicas must be >= 1, got {worker_replicas}")
    if gpus_per_worker < 1:
        raise ValueError(f"kuberay.gpus_per_worker must be >= 1, got {gpus_per_worker}")
    if head_gpu_count < 0:
        raise ValueError(f"kuberay.head_gpu_count must be >= 0, got {head_gpu_count}")

    return {
        "gpu_type": gpu_type,
        "worker_replicas": worker_replicas,
        "gpus_per_worker": gpus_per_worker,
        "total_gpus": worker_replicas * gpus_per_worker,
        "head_gpu_count": head_gpu_count,
        "head_cpu": str(kuberay_cfg.get("head_cpu", 4)),
        "head_memory": str(kuberay_cfg.get("head_memory", "16Gi")),
        "worker_cpu": str(kuberay_cfg.get("worker_cpu", kuberay_cfg.get("cpu", 8))),
        "worker_memory": str(
            kuberay_cfg.get("worker_memory", kuberay_cfg.get("memory", "64Gi"))
        ),
    }


def _build_submission_config(config_dict: dict, kuberay_cfg: dict) -> tuple[dict, str]:
    """Prepare the config consumed by liquid-finetune inside the Ray job."""
    cluster_spec = _resolve_kuberay_cluster_spec(kuberay_cfg)

    submission_config = {
        key: value for key, value in config_dict.items() if key != "kuberay"
    }
    output_dir = kuberay_cfg.get("output_dir", "/outputs")
    submission_config.setdefault("training_config", {})["output_dir"] = output_dir

    ray_cfg = dict(submission_config.get("ray") or {})
    ray_cfg.setdefault("address", "auto")
    ray_cfg.setdefault("num_workers", cluster_spec["total_gpus"])
    submission_config["ray"] = ray_cfg
    return submission_config, output_dir


def check_and_handle_kuberay(
    config_path_arg: str | None = None,
    *,
    config_dict: dict | None = None,
) -> bool:
    if config_dict is None:
        if not config_path_arg:
            return False

        try:
            from liquid_finetune.config.parser import resolve_config_path

            config_path = resolve_config_path(config_path_arg)
        except (FileNotFoundError, Exception):
            return False

        with open(config_path) as f:
            config_dict = yaml.safe_load(f) or {}

    try:
        kuberay_cfg = (
            config_dict.get("kuberay") if isinstance(config_dict, dict) else None
        )
        if not kuberay_cfg:
            return False

        if not kuberay_cfg.get("image"):
            print(
                "Error: kuberay.image is required (container image with liquid-finetune)"
            )
            sys.exit(1)

        print("Config contains KubeRay settings - submitting RayJob...\n")
        _print_config_summary(config_dict, kuberay_cfg)
        _submit(config_dict, kuberay_cfg)
        return True
    except SystemExit:
        raise
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"\nError submitting KubeRay job: {exc}")
        sys.exit(1)


def _print_config_summary(config_dict: dict, kuberay_cfg: dict) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    train_cfg = config_dict.get("training_config", {})
    ds_cfg = config_dict.get("dataset", {})
    peft_cfg = config_dict.get("peft_config", {})
    cluster_spec = _resolve_kuberay_cluster_spec(kuberay_cfg)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="bold cyan", min_width=18)
    table.add_column("Value", style="green")

    table.add_row("Model", config_dict.get("model_name", "?"))
    table.add_row("Training Type", config_dict.get("training_type", "?").upper())
    table.add_row("Dataset", ds_cfg.get("path", "?"))
    if ds_cfg.get("limit"):
        table.add_row("Dataset Limit", f"{ds_cfg['limit']:,}")
    table.add_row("Image", kuberay_cfg["image"])
    table.add_row("Worker Replicas", str(cluster_spec["worker_replicas"]))
    table.add_row("GPUs / Worker", str(cluster_spec["gpus_per_worker"]))
    table.add_row("Total GPUs", str(cluster_spec["total_gpus"]))
    table.add_row("Namespace", kuberay_cfg.get("namespace", "default"))

    if train_cfg.get("learning_rate"):
        table.add_row("Learning Rate", f"{float(train_cfg['learning_rate']):.2e}")
    if train_cfg.get("per_device_train_batch_size"):
        table.add_row("Batch Size", str(train_cfg["per_device_train_batch_size"]))
    if train_cfg.get("num_train_epochs"):
        table.add_row("Epochs", str(train_cfg["num_train_epochs"]))
    if peft_cfg.get("use_peft"):
        table.add_row("PEFT", f"Enabled ({peft_cfg.get('extends', 'custom')})")

    panel = Panel(
        table,
        title="[bold blue]KubeRay Training Configuration[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )
    Console().print(panel)


def _load_kubernetes_config(config_module) -> None:
    load_attempts = [
        ("kubeconfig", config_module.load_kube_config),
        ("in-cluster config", config_module.load_incluster_config),
    ]
    failures = []
    for label, loader in load_attempts:
        try:
            loader()
            return
        except Exception as exc:
            failures.append(f"{label}: {exc}")

    print("Error: Could not load Kubernetes client config.")
    for failure in failures:
        print(f"  {failure}")
    sys.exit(1)


def _submit(config_dict: dict, kuberay_cfg: dict) -> None:
    try:
        from kubernetes import client, config
    except ImportError:
        print("Error: 'kubernetes' package is required for KubeRay support.")
        print("  uv sync")
        print("  or: uv add kubernetes")
        sys.exit(1)

    namespace = kuberay_cfg.get("namespace", "default")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = kuberay_cfg.get("job_name", f"liquid-finetune-{timestamp}")
    configmap_name = f"{job_name}-config"

    submission_config, output_dir = _build_submission_config(config_dict, kuberay_cfg)
    config_str = yaml.safe_dump(submission_config, sort_keys=False)

    _load_kubernetes_config(config)

    core_v1 = client.CoreV1Api()
    custom_api = client.CustomObjectsApi()

    configmap = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=configmap_name, namespace=namespace),
        data={"config.yaml": config_str},
    )

    print(f"Submitting RayJob '{job_name}' to namespace '{namespace}'...")

    try:
        core_v1.create_namespaced_config_map(namespace=namespace, body=configmap)
    except client.ApiException as exc:
        print(f"Failed to create ConfigMap: {exc.reason}")
        sys.exit(1)

    rayjob = _generate_rayjob_manifest(
        job_name, configmap_name, kuberay_cfg, output_dir
    )

    try:
        custom_api.create_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=namespace,
            plural="rayjobs",
            body=rayjob,
        )
    except client.ApiException as exc:
        print(f"Failed to create RayJob: {exc.reason}")
        sys.exit(1)

    print(f"RayJob '{job_name}' submitted successfully.")
    print("\nMonitor with:")
    print(f"  kubectl get rayjob {job_name} -n {namespace}")
    print(f"  kubectl logs -f -l ray.io/cluster={job_name} -n {namespace}")


def _generate_rayjob_manifest(
    job_name: str,
    configmap_name: str,
    kuberay_cfg: dict,
    output_dir: str,
) -> dict:
    cluster_spec = _resolve_kuberay_cluster_spec(kuberay_cfg)
    gpu_type = cluster_spec["gpu_type"]
    image = kuberay_cfg["image"]
    ray_version = kuberay_cfg.get("ray_version", "2.48.0")

    env_list = [
        {"name": "OUTPUT_DIR", "value": output_dir},
        {"name": "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "value": "1"},
    ]
    for key, value in kuberay_cfg.get("env", {}).items():
        env_list.append({"name": key, "value": str(value)})

    volume_mounts = [
        {"name": "config", "mountPath": "/tmp/config.yaml", "subPath": "config.yaml"},
        {"name": "dshm", "mountPath": "/dev/shm"},
    ]
    volumes = [
        {"name": "config", "configMap": {"name": configmap_name}},
        {"name": "dshm", "emptyDir": {"medium": "Memory"}},
    ]

    output_pvc = kuberay_cfg.get("output_pvc")
    if output_pvc:
        volume_mounts.append({"name": "output-vol", "mountPath": output_dir})
        volumes.append(
            {"name": "output-vol", "persistentVolumeClaim": {"claimName": output_pvc}}
        )

    head_container = {
        "name": "ray-head",
        "image": image,
        "resources": {
            "limits": {
                "memory": cluster_spec["head_memory"],
                "cpu": cluster_spec["head_cpu"],
            },
            "requests": {
                "memory": cluster_spec["head_memory"],
                "cpu": cluster_spec["head_cpu"],
            },
        },
        "volumeMounts": volume_mounts,
        "env": env_list,
    }
    if cluster_spec["head_gpu_count"] > 0:
        head_container["resources"]["limits"][gpu_type] = str(
            cluster_spec["head_gpu_count"]
        )
        head_container["resources"]["requests"][gpu_type] = str(
            cluster_spec["head_gpu_count"]
        )

    worker_container = {
        "name": "ray-worker",
        "image": image,
        "resources": {
            "limits": {
                gpu_type: str(cluster_spec["gpus_per_worker"]),
                "memory": cluster_spec["worker_memory"],
                "cpu": cluster_spec["worker_cpu"],
            },
            "requests": {
                gpu_type: str(cluster_spec["gpus_per_worker"]),
                "memory": cluster_spec["worker_memory"],
                "cpu": cluster_spec["worker_cpu"],
            },
        },
        "volumeMounts": volume_mounts,
        "env": env_list,
    }

    head_pod_spec = {"containers": [head_container], "volumes": volumes}
    worker_pod_spec = {"containers": [worker_container], "volumes": volumes}

    if kuberay_cfg.get("service_account_name"):
        head_pod_spec["serviceAccountName"] = kuberay_cfg["service_account_name"]
        worker_pod_spec["serviceAccountName"] = kuberay_cfg["service_account_name"]

    common_node_selector = kuberay_cfg.get("node_selector")
    common_tolerations = kuberay_cfg.get("tolerations")
    head_pod_spec["nodeSelector"] = kuberay_cfg.get(
        "head_node_selector", common_node_selector
    )
    worker_pod_spec["nodeSelector"] = kuberay_cfg.get(
        "worker_node_selector", common_node_selector
    )
    head_pod_spec["tolerations"] = kuberay_cfg.get(
        "head_tolerations", common_tolerations
    )
    worker_pod_spec["tolerations"] = kuberay_cfg.get(
        "worker_tolerations", common_tolerations
    )
    if head_pod_spec["nodeSelector"] is None:
        head_pod_spec.pop("nodeSelector")
    if worker_pod_spec["nodeSelector"] is None:
        worker_pod_spec.pop("nodeSelector")
    if head_pod_spec["tolerations"] is None:
        head_pod_spec.pop("tolerations")
    if worker_pod_spec["tolerations"] is None:
        worker_pod_spec.pop("tolerations")

    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": job_name,
            "namespace": kuberay_cfg.get("namespace", "default"),
        },
        "spec": {
            "submissionMode": "K8sJobMode",
            "entrypoint": (
                'python -c "'
                "import sys; "
                "sys.argv=['liquid-finetune', '/tmp/config.yaml']; "
                "from liquid_finetune import main; "
                'main()"'
            ),
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": kuberay_cfg.get(
                "ttl_seconds_after_finished", 300
            ),
            "rayClusterSpec": {
                "rayVersion": ray_version,
                "headGroupSpec": {
                    "serviceType": kuberay_cfg.get("head_service_type", "ClusterIP"),
                    "rayStartParams": {
                        "dashboard-host": "0.0.0.0",
                        **(
                            {"num-gpus": str(cluster_spec["head_gpu_count"])}
                            if cluster_spec["head_gpu_count"] > 0
                            else {}
                        ),
                    },
                    "template": {
                        "metadata": {"labels": {"ray.io/cluster": job_name}},
                        "spec": head_pod_spec,
                    },
                },
                "workerGroupSpecs": [
                    {
                        "groupName": "gpu-workers",
                        "replicas": cluster_spec["worker_replicas"],
                        "minReplicas": cluster_spec["worker_replicas"],
                        "maxReplicas": cluster_spec["worker_replicas"],
                        "rayStartParams": {
                            "num-gpus": str(cluster_spec["gpus_per_worker"])
                        },
                        "template": {
                            "metadata": {"labels": {"ray.io/cluster": job_name}},
                            "spec": worker_pod_spec,
                        },
                    }
                ],
            },
        },
    }
