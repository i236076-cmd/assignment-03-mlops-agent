#!/usr/bin/env bash
# Generates all evidence, adds it to REPORT.md and commits. Push is done by hand with a token.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
: "${AIRFLOW_PASSWORD:?run: export AIRFLOW_PASSWORD=... first}"
pytest -q || { echo "tests failed, stop"; exit 1; }
OUT=screenshots/logs; mkdir -p "$OUT"

echo "== demos"; ./scripts/run_demo.sh > /dev/null 2>&1
echo "== failure 1 (quality gate)"
python -m agent.main "Why did the last training run fail?" 2>/dev/null > "$OUT/failure1.txt"
echo "== failure 2 (bad image)"
kubectl -n mlops-demo set image deployment/churn-api churn-api=churn-api:does-not-exist
sleep 25
python -m agent.main "Why is my prediction API unavailable?" 2>/dev/null > "$OUT/failure2.txt"
kubectl -n mlops-demo get pods > "$OUT/failure2_pods.txt"
kubectl -n mlops-demo rollout undo deployment/churn-api 2>/dev/null
kubectl -n mlops-demo rollout status deployment/churn-api --timeout=120s
kubectl -n mlops-demo get pods,svc,deploy > "$OUT/k8s_final.txt"
{ dvc status; git tag; } > "$OUT/dvc_final.txt" 2>&1

python3 - <<'PY'
from pathlib import Path
import glob
r = Path("REPORT.md")
marker = "\n---\n## Appendix: actual agent outputs"
text = r.read_text().split(marker)[0]
def block(title, path):
    return "\n### %s\n```\n%s\n```\n" % (title, Path(path).read_text().strip())
out = text + marker + "\n"
for f in sorted(glob.glob("screenshots/logs/demo*.txt")):
    out += block(Path(f).stem, f)
out += block("Failure 1: quality gate failed (root cause: min_f1=0.95 is above the model F1 of about 0.74; fix: use a realistic threshold such as 0.70)", "screenshots/logs/failure1.txt")
out += block("Failure 2: bad image tag (root cause: churn-api:does-not-exist cannot be pulled, so the new pod stays in ImagePullBackOff while the old pods keep serving; fix: kubectl rollout undo)", "screenshots/logs/failure2.txt")
out += block("Failure 2: pods during the failure", "screenshots/logs/failure2_pods.txt")
out += block("Kubernetes state after recovery", "screenshots/logs/k8s_final.txt")
out += block("DVC status and dataset tags", "screenshots/logs/dvc_final.txt")
r.write_text(out)
print("REPORT.md updated")
PY
git add -A && git commit -qm "Add demo evidence, failure investigations and report appendix" && echo "== committed. Now push with your token."
grep -ci "pull" "$OUT/failure2.txt" | sed 's/^/failure2 mentions image pull problem (count): /'
