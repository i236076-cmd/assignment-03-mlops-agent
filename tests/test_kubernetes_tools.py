from types import SimpleNamespace as NS

from agent.tools import kubernetes_tools as k8s
from agent.tools.base import call_tool


def pod(name, phase="Running", ready=True, restarts=0, waiting=None, last_reason=None):
    state = NS(waiting=NS(reason=waiting) if waiting else None, terminated=None)
    last = NS(terminated=NS(reason=last_reason, exit_code=137) if last_reason else None)
    status = NS(name="churn-api", ready=ready, restart_count=restarts, state=state, last_state=last, image="churn-api:1.0.0")
    return NS(metadata=NS(name=name), spec=NS(containers=[NS(image="churn-api:1.0.0")], node_name="minikube"),
              status=NS(phase=phase, container_statuses=[status], start_time=None))


class FakeCore:
    def __init__(self, pods, log=""):
        self.pods, self.log = pods, log

    def list_namespaced_pod(self, ns, **kw):
        return NS(items=self.pods)

    def list_namespaced_event(self, ns, **kw):
        return NS(items=[NS(type="Warning", reason="BackOff", message="Back-off restarting", count=3,
                            last_timestamp=None, event_time=None, metadata=NS(creation_timestamp=None),
                            involved_object=NS(name="churn-api-x"))])

    def read_namespaced_pod_log(self, name, ns, **kw):
        return self.log


class FakeApps:
    def __init__(self):
        self.patches = []

    def read_namespaced_deployment(self, name, ns, **kw):
        return NS(metadata=NS(name=name), spec=NS(replicas=2, template=NS(
            metadata=NS(labels={"version": "1.0.0"}), spec=NS(containers=[NS(image="churn-api:1.0.0")]))),
            status=NS(ready_replicas=1, updated_replicas=2, available_replicas=1, conditions=[]))

    def patch_namespaced_deployment(self, name, ns, body, **kw):
        self.patches.append((name, ns, body))


def use_fakes(monkeypatch, core, apps):
    monkeypatch.setattr(k8s, "_apis", lambda: (core, apps))


def test_unhealthy_pod_and_restart_events_are_found(monkeypatch):
    pods = [pod("churn-api-a"), pod("churn-api-b", ready=False, restarts=4, waiting="CrashLoopBackOff", last_reason="OOMKilled")]
    use_fakes(monkeypatch, FakeCore(pods), FakeApps())
    result = call_tool("k8s_find_unhealthy_pods")
    assert result["status"] == "ok"
    assert [p["name"] for p in result["data"]["unhealthy_pods"]] == ["churn-api-b"]
    assert result["data"]["pods_with_restarts"][0]["restarts"] == 4
    assert result["data"]["warning_events"][0]["reason"] == "BackOff"


def test_diagnosis_uses_logs_as_evidence(monkeypatch):
    pods = [pod("churn-api-a", ready=False)]
    core = FakeCore(pods, log="ERROR could not load model from model: no such file")
    use_fakes(monkeypatch, core, FakeApps())
    result = call_tool("k8s_diagnose_deployment")
    text = " ".join(result["data"]["findings"])
    assert "not ready" in text and "model could not be loaded" in text


def test_deployed_version_report(monkeypatch):
    core = FakeCore([])
    core.read_namespaced_config_map = lambda name, ns, **kw: NS(data={"APP_VERSION": "1.0.0"})
    use_fakes(monkeypatch, core, FakeApps())
    data = call_tool("k8s_get_deployed_version")["data"]
    assert data["images"] == ["churn-api:1.0.0"] and data["configmap_app_version"] == "1.0.0"
    assert data["rollout_complete"] is False          # 1 of 2 replicas ready


def test_unauthorized_modifications_are_rejected(monkeypatch):
    apps = FakeApps()
    use_fakes(monkeypatch, FakeCore([]), apps)
    wrong_namespace = call_tool("k8s_restart_deployment", namespace="kube-system", confirm=True)
    no_confirmation = call_tool("k8s_restart_deployment")
    scale_other_ns = call_tool("k8s_scale_deployment", replicas=0, namespace="default", confirm=True)
    assert wrong_namespace["status"] == "rejected" and "not approved" in wrong_namespace["error"]
    assert no_confirmation["status"] == "rejected" and "confirmation" in no_confirmation["error"]
    assert scale_other_ns["status"] == "rejected"
    assert apps.patches == []                          # the cluster was never touched


def test_confirmed_restart_in_approved_namespace_works(monkeypatch):
    apps = FakeApps()
    use_fakes(monkeypatch, FakeCore([]), apps)
    assert call_tool("k8s_restart_deployment", confirm=True)["status"] == "ok"
    assert len(apps.patches) == 1
    assert call_tool("k8s_scale_deployment", replicas=99, confirm=True)["status"] == "error"   # too many replicas


def test_kubernetes_unreachable_is_unavailable(monkeypatch):
    def no_config():
        from agent.schemas import ToolUnavailable
        raise ToolUnavailable("no Kubernetes configuration found")
    monkeypatch.setattr(k8s, "_apis", no_config)
    assert call_tool("k8s_list_workloads")["status"] == "unavailable"
