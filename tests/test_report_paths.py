"""Reports are committed to a public repo: no DATA_ROOT, repo or home path, and no Drive account name, in them."""
from pathlib import Path

from analytics import report_paths as RP

HOME = "/Users/someone"
DATA = HOME + "/Library/CloudStorage/GoogleDrive-someone@example.com/My Drive/nba_harvest"
REPO = HOME + "/Documents/proj"


def test_redact_names_data_root_repo_and_home_without_the_account():
    text = ("Written by `python -m analytics.ghosting.train --data %s --report %s/reports/ghosting.md`.\n"
            "log %s/logs/x.log; cache %s/.cache; other %s/Library/CloudStorage/GoogleDrive-a@b.com/other\n"
            % (DATA, REPO, REPO, HOME, HOME))
    out = RP.redact(text, data_root=DATA, repo_root=REPO, home=HOME)
    assert "--data $DATA_ROOT --report reports/ghosting.md" in out
    assert "log logs/x.log" in out and "cache ~/.cache" in out
    assert "~/Library/CloudStorage/GoogleDrive/other" in out
    for leak in ("@", HOME, "someone"):
        assert leak not in out


def test_redact_without_data_root_still_drops_the_account():
    out = RP.redact("--data %s x" % DATA, data_root=None, repo_root=REPO, home=HOME)
    assert out == "--data ~/Library/CloudStorage/GoogleDrive/My Drive/nba_harvest x"


def test_redact_file_rewrites_once(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("cmd --data %s\n" % DATA, encoding="utf-8")
    assert RP.redact_file(p, data_root=DATA, repo_root=REPO, home=HOME) is True
    assert p.read_text(encoding="utf-8") == "cmd --data $DATA_ROOT\n"
    assert RP.redact_file(p, data_root=DATA, repo_root=REPO, home=HOME) is False
    assert RP.redact_file(tmp_path / "absent.md", data_root=DATA) is False


def test_run_all_redacts_lane_reports(tmp_path, monkeypatch):
    import run_all
    seen = []
    monkeypatch.setattr(run_all, "redact_file", lambda path, data_root=None: seen.append(Path(path)) or False)
    run_all.main(["--fixtures", "--ext", "xfg", "--out", str(tmp_path), "--skip-heldout"])
    assert tmp_path / "reports" / "xfg.md" in seen
