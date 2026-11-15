"""Kubernetes tools for the churn API deployment.

Every tool works only in the approved namespace. Tools that change the cluster
(restart, scale) are write tools and need explicit user confirmation.
"""
import datetime as dt
import os

from kubernetes import client
from kubernetes import config as kube_config
from kubernetes.client.rest import ApiException
from urllib3.exceptions import HTTPError as Urllib3Error

from agent import config
from agent.logging_utils import redact
from agent.schemas import ToolError, ToolRejected, ToolUnavailable
from agent.tools.base import register

API_TIMEOUT = 8  # seconds per Kubernetes API call
CONFIGMAP = os.getenv("K8S_CONFIGMAP", "churn-api-config")
MAX_REPLICAS = 5

NS_PROP = {"type": "string", "description": "namespace, only the approved one is allowed"}


# ---------- helpers ----------
def _apis():
    """Return (CoreV1Api, AppsV1Api). Tests replace this function."""
    try:
        kube_config.load_kube_config()
    except Exception:
        try:
            kube_config.load_incluster_config()
        except Exception as exc:
            raise ToolUnavailable("no Kubernetes configuration found (%s)" % type(exc).__name__)
    return client.CoreV1Api(), client.AppsV1Api()


def _call(fn, *args, **kwargs):
    """Run one Kubernetes API call and map its errors to tool errors."""
    try:
        return fn(*args, _request_timeout=API_TIMEOUT, **kwargs)
    except ApiException as exc:
        if exc.status == 404:
            raise ToolError("not found in Kubernetes: %s" % exc.reason)
        if exc.status in (401, 403):
            raise ToolRejected("Kubernetes denied the request (HTTP %s)" % exc.status)
        raise ToolUnavailable("Kubernetes API error (HTTP %s %s)" % (exc.status, exc.reason))
    except (Urllib3Error, OSError) as exc:
        raise ToolUnavailable("Kubernetes API is not reachable (%s)" % type(exc).__name__)


def _ns(namespace):
    namespace = namespace or config.K8S_NAMESPACE
    if namespace not in config.K8S_ALLOWED_NAMESPACES:
        raise ToolRejected("namespace '%s' is not approved; allowed: %s"
                           % (namespace, ", ".join(sorted(config.K8S_ALLOWED_NAMESPACES))))
    return namespace


def _g(obj, path, default=None):
    """Safe attribute lookup: _g(pod, 'status.phase')."""
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


def _pod_summary(pod):
    statuses = _g(pod, "status.container_statuses", []) or []
    problems = []
    for c in statuses:
        waiting = _g(c, "state.waiting")
        terminated = _g(c, "state.terminated")
        last = _g(c, "last_state.terminated")
        if waiting and waiting.reason:
            problems.append("%s is waiting: %s" % (c.name, waiting.reason))
        if terminated and terminated.reason:
            problems.append("%s terminated: %s (exit code %s)" % (c.name, terminated.reason, terminated.exit_code))
        if last and last.reason:
            problems.append("%s last restart was caused by: %s (exit code %s)" % (c.name, last.reason, last.exit_code))
    restarts = sum((c.restart_count or 0) for c in statuses)
    ready = bool(statuses) and all(c.ready for c in statuses)
    phase = _g(pod, "status.phase")
    waiting_problem = any("is waiting" in p for p in problems)
    return {
        "name": _g(pod, "metadata.name"),
        "phase": phase,
        "ready": ready,
        "restarts": restarts,
        "healthy": phase == "Running" and ready and not waiting_problem,
        "images": [c.image for c in (_g(pod, "spec.containers", []) or [])],
        "node": _g(pod, "spec.node_name"),
        "started": _iso(_g(pod, "status.start_time")),
        "problems": problems,
    }


def _deployment_summary(dep):
    containers = _g(dep, "spec.template.spec.containers", []) or []
    return {
        "name": _g(dep, "metadata.name"),
        "desired_replicas": _g(dep, "spec.replicas", 0),
        "ready_replicas": _g(dep, "status.ready_replicas", 0) or 0,
        "updated_replicas": _g(dep, "status.updated_replicas", 0) or 0,
        "available_replicas": _g(dep, "status.available_replicas", 0) or 0,
        "images": [c.image for c in containers],
        "version_label": (_g(dep, "spec.template.metadata.labels", {}) or {}).get("version"),
        "conditions": [
            {"type": c.type, "status": c.status, "reason": c.reason}
            for c in (_g(dep, "status.conditions", []) or [])
        ],
    }


def _pods_of(core, ns, name=None):
    pods = _call(core.list_namespaced_pod, ns).items
    if name:
        pods = [p for p in pods if _g(p, "metadata.name") == name]
        if not pods:
            raise ToolError("pod '%s' not found in namespace %s" % (name, ns))
    return pods


def _warning_events(core, ns, limit=10):
    events = _call(core.list_namespaced_event, ns).items
    out = []
    for e in events:
        if e.type != "Warning":
            continue
        when = e.last_timestamp or e.event_time or _g(e, "metadata.creation_timestamp")
        out.append((when.timestamp() if hasattr(when, "timestamp") else 0, {
            "object": _g(e, "involved_object.name"),
            "reason": e.reason,
            "message": redact((e.message or "")[:300]),
            "count": e.count,
            "time": _iso(when),
        }))
    out.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in out[:limit]]


def _pod_logs(core, ns, pod, tail=40, previous=False):
    text = _call(core.read_namespaced_pod_log, pod, ns, tail_lines=int(tail), previous=bool(previous))
    return redact(text)[-4000:]


def _findings(deployment, pods, logs):
    """Plain-language findings that come only from the collected evidence."""
    notes = []
    if deployment and deployment["desired_replicas"] == 0:
        notes.append("The deployment is scaled to 0 replicas, so no pod can serve requests.")
    for p in pods:
        text = " ".join(p["problems"])
        if "ImagePullBackOff" in text or "ErrImagePull" in text or "InvalidImageName" in text:
            notes.append("%s cannot start because its image cannot be pulled (%s). Check the image name and tag."
                         % (p["name"], ", ".join(p["images"])))
        if "CrashLoopBackOff" in text:
            notes.append("%s keeps crashing (CrashLoopBackOff, %s restarts). Read its logs for the error." % (p["name"], p["restarts"]))
        if "OOMKilled" in text:
            notes.append("%s was killed for using too much memory (OOMKilled). Raise the memory limit." % p["name"])
        if p["phase"] == "Running" and not p["ready"]:
            notes.append("%s is running but not ready, so the readiness probe is failing and it gets no traffic." % p["name"])
        if p["phase"] == "Pending":
            notes.append("%s is Pending, it has not been scheduled or started yet." % p["name"])
    for pod, text in logs.items():
        if "model not loaded" in text or "could not load model" in text:
            notes.append("Logs of %s show the model could not be loaded, which keeps /ready failing." % pod)
    if not notes and pods and all(p["healthy"] for p in pods):
        notes.append("All %d pod(s) are Running and Ready. No fault found in the cluster data." % len(pods))
    if not pods:
        notes.append("No pods exist for this deployment.")
    return notes


# ---------- read tools ----------
@register("k8s_list_workloads", "List the Deployments (with replica counts) and Pods in the approved namespace.",
          {"type": "object", "properties": {"namespace": NS_PROP}})
def k8s_list_workloads(namespace=None):
    ns = _ns(namespace)
    core, apps = _apis()
    deployments = [_deployment_summary(d) for d in _call(apps.list_namespaced_deployment, ns).items]
    pods = [_pod_summary(p) for p in _pods_of(core, ns)]
    return {"namespace": ns, "deployments": deployments, "pods": pods}


@register("k8s_get_pod_status", "Get status, readiness, restart count and problems of pods (all pods or one by name).",
          {"type": "object", "properties": {"pod": {"type": "string"}, "namespace": NS_PROP}})
def k8s_get_pod_status(pod=None, namespace=None):
    ns = _ns(namespace)
    core, _ = _apis()
    return {"namespace": ns, "pods": [_pod_summary(p) for p in _pods_of(core, ns, pod)]}


@register("k8s_get_pod_logs", "Read the last log lines of a pod. Set previous=true to read the crashed container.",
          {"type": "object", "properties": {
              "pod": {"type": "string"}, "tail_lines": {"type": "integer", "description": "default 50"},
              "previous": {"type": "boolean"}, "namespace": NS_PROP}, "required": ["pod"]})
def k8s_get_pod_logs(pod, tail_lines=50, previous=False, namespace=None):
    ns = _ns(namespace)
    core, _ = _apis()
    return {"namespace": ns, "pod": pod, "previous": bool(previous),
            "log_tail": _pod_logs(core, ns, pod, min(int(tail_lines), 200), previous)}


@register("k8s_list_services", "List Services with type, ports, selector and the number of ready endpoints.",
          {"type": "object", "properties": {"namespace": NS_PROP}})
def k8s_list_services(namespace=None):
    ns = _ns(namespace)
    core, _ = _apis()
    services = []
    for s in _call(core.list_namespaced_service, ns).items:
        name = _g(s, "metadata.name")
        try:
            ep = _call(core.read_namespaced_endpoints, name, ns)
            ready = sum(len(sub.addresses or []) for sub in (ep.subsets or []))
        except (ToolError, ToolUnavailable):
            ready = None
        services.append({
            "name": name,
            "type": _g(s, "spec.type"),
            "cluster_ip": _g(s, "spec.cluster_ip"),
            "ports": [{"port": p.port, "target_port": str(p.target_port), "protocol": p.protocol}
                      for p in (_g(s, "spec.ports", []) or [])],
            "selector": _g(s, "spec.selector"),
            "ready_endpoints": ready,
        })
    return {"namespace": ns, "services": services}


@register("k8s_find_unhealthy_pods", "Find pods that are not ready or have restarted, plus recent Warning events.",
          {"type": "object", "properties": {"namespace": NS_PROP}})
def k8s_find_unhealthy_pods(namespace=None):
    ns = _ns(namespace)
    core, _ = _apis()
    pods = [_pod_summary(p) for p in _pods_of(core, ns)]
    return {
        "namespace": ns,
        "total_pods": len(pods),
        "unhealthy_pods": [p for p in pods if not p["healthy"]],
        "pods_with_restarts": [p for p in pods if p["restarts"] > 0],
        "warning_events": _warning_events(core, ns),
    }


@register("k8s_get_deployed_version", "Report the deployed application version: image tag, version label, ConfigMap APP_VERSION and rollout status.",
          {"type": "object", "properties": {"deployment": {"type": "string"}, "namespace": NS_PROP}})
def k8s_get_deployed_version(deployment=None, namespace=None):
    ns = _ns(namespace)
    core, apps = _apis()
    name = deployment or config.K8S_DEPLOYMENT
    summary = _deployment_summary(_call(apps.read_namespaced_deployment, name, ns))
    try:
        cm = _call(core.read_namespaced_config_map, CONFIGMAP, ns)
        app_version = (cm.data or {}).get("APP_VERSION")
    except (ToolError, ToolUnavailable):
        app_version = None
    return {
        "namespace": ns,
        "deployment": name,
        "images": summary["images"],
        "version_label": summary["version_label"],
        "configmap_app_version": app_version,
        "replicas": {"desired": summary["desired_replicas"], "ready": summary["ready_replicas"],
                     "updated": summary["updated_replicas"], "available": summary["available_replicas"]},
        "rollout_complete": summary["desired_replicas"] == summary["ready_replicas"] == summary["updated_replicas"],
    }


@register("k8s_diagnose_deployment",
          "Investigate why the prediction API may be unavailable: deployment status, pod status, probes, Warning events and logs, with findings.",
          {"type": "object", "properties": {"deployment": {"type": "string"}, "namespace": NS_PROP}})
def k8s_diagnose_deployment(deployment=None, namespace=None):
    ns = _ns(namespace)
    core, apps = _apis()
    name = deployment or config.K8S_DEPLOYMENT
    dep = _deployment_summary(_call(apps.read_namespaced_deployment, name, ns))
    pods = [_pod_summary(p) for p in _pods_of(core, ns) if (_g(p, "metadata.name") or "").startswith(name)]
    logs = {}
    for p in pods:
        if not p["healthy"] or p["restarts"] > 0:
            try:
                logs[p["name"]] = _pod_logs(core, ns, p["name"], 30, previous=p["restarts"] > 0 and not p["ready"])
            except (ToolError, ToolUnavailable) as exc:
                logs[p["name"]] = "log not available: %s" % exc
    return {  # findings first, so they survive when long results are shortened
        "findings": _findings(dep, pods, logs),
        "namespace": ns, "deployment": dep, "pods": pods,
        "warning_events": _warning_events(core, ns), "logs": logs,
    }


# ---------- write tools (need confirmation) ----------
def _need_confirmation(confirm, action):
    if confirm is not True:
        raise ToolRejected("%s changes the cluster and needs explicit confirmation from the user (confirm=true)" % action)


@register("k8s_restart_deployment", "Rolling restart of a deployment. WRITE action: needs explicit user confirmation.",
          {"type": "object", "properties": {"deployment": {"type": "string"}, "namespace": NS_PROP}},
          permission="write")
def k8s_restart_deployment(deployment=None, namespace=None, confirm=False):
    ns = _ns(namespace)  # namespace is checked first
    _need_confirmation(confirm, "restart")
    _, apps = _apis()
    name = deployment or config.K8S_DEPLOYMENT
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": stamp}}}}}
    _call(apps.patch_namespaced_deployment, name, ns, body)
    return {"namespace": ns, "deployment": name, "restarted_at": stamp}


@register("k8s_scale_deployment", "Change the replica count of a deployment (0 to 5). WRITE action: needs explicit user confirmation.",
          {"type": "object", "properties": {
              "replicas": {"type": "integer"}, "deployment": {"type": "string"}, "namespace": NS_PROP},
           "required": ["replicas"]},
          permission="write")
def k8s_scale_deployment(replicas, deployment=None, namespace=None, confirm=False):
    ns = _ns(namespace)
    _need_confirmation(confirm, "scaling")
    if isinstance(replicas, bool) or not isinstance(replicas, int) or not 0 <= replicas <= MAX_REPLICAS:
        raise ToolError("replicas must be a whole number between 0 and %d" % MAX_REPLICAS)
    _, apps = _apis()
    name = deployment or config.K8S_DEPLOYMENT
    _call(apps.patch_namespaced_deployment, name, ns, {"spec": {"replicas": replicas}})
    return {"namespace": ns, "deployment": name, "replicas": replicas}
