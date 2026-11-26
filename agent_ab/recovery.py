import json
from pathlib import Path

from .util import now_iso, write_json
from .fastrelax_naming import canonical_stage_name, fastrelax_stage


TERMINAL = {"complete", "failed", "skipped"}


class RecoveryState:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "orchestration_state.json"
        self.value = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"stages": {}, "history": []}

    def stage(self, name):
        return fastrelax_stage(self.value["stages"], name)

    def update(self, name, status, detail=None, **extra):
        name = canonical_stage_name(name)
        previous = self.stage(name)
        attempt = int(previous.get("attempt", 0)) + (1 if status == "running" else 0)
        value = {"status": status, "updated_at": now_iso(), "attempt": attempt}
        if detail:
            value["detail"] = detail
        value.update(extra)
        self.value["stages"][name] = value
        legacy_name = name.replace("_fastrelax", "_rosetta_fix")
        if legacy_name != name:
            self.value["stages"].pop(legacy_name, None)
        self.value["history"].append({"stage": name, "status": status, "at": value["updated_at"], "detail": detail})
        write_json(self.path, self.value)
        return value

    def should_run(self, name, retry_failed=False):
        status = self.stage(name).get("status")
        return status not in ("complete", "running") and (status != "failed" or retry_failed)
