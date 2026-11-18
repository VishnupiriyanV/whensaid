"""`whensaid doctor`: what works, what doesn't, and what to do about it. Makes no network
calls and never loads a model onto the GPU."""
import os
import platform
import shutil
import subprocess
import sys
import time

OK, WARN, FAIL = "ok", "warn", "fail"


def _free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def checks() -> list[tuple[str, str, str, str]]:
    """(status, name, detail, how to fix)"""
    from . import cudalibs, db, models, paths
    from .indexer import indexer_running

    out = []
    add = lambda status, name, detail, fix="": out.append((status, name, detail, fix))

    add(OK, "python", f"{platform.python_version()} on {platform.system()} {platform.machine()}")

    # compute
    cudalibs.preload()  # the pip wheels are found by path, not by the system loader
    try:
        import ctranslate2

        gpus = ctranslate2.get_cuda_device_count()
    except Exception as e:
        add(FAIL, "ctranslate2", f"cannot be imported: {e}", "pip install --force-reinstall whensaid")
        gpus = 0
    if gpus and cudalibs.libs_load():
        add(OK, "gpu", f"{gpus} NVIDIA GPU, CUDA 12 libraries found")
    elif gpus:
        add(WARN, "gpu", "an NVIDIA GPU is there but the CUDA 12 libraries are not: transcribing on the CPU (about 3x slower)",
            "pip install 'whensaid[cuda]'   (with uvx: uvx --from 'whensaid[cuda]' whensaid ...)")
    else:
        add(OK, "gpu", f"no NVIDIA GPU, transcribing on the CPU ({os.cpu_count()} cores)")

    # models
    free = _free_gb(paths.models_dir())
    for kind in ("whisper", "embedder", "reranker"):
        if models.present(kind):
            add(OK, f"{kind} model", "on disk")
        else:
            need = {"whisper": 1.5, "embedder": 0.1, "reranker": 0.1}[kind]
            if free < need * 1.5:
                add(FAIL, f"{kind} model", f"missing, and only {free:.1f} GB free in {paths.models_dir()}", "free some disk space")
            else:
                add(WARN, f"{kind} model", f"not downloaded yet ({models.SIZES[kind]}), fetched on first use",
                    "whensaid models pull" if not models.offline() else "offline mode is on: run `whensaid models pull` once online")
    if (paths.home() / "ipv4").exists():
        add(OK, "network", "downloads use IPv4 only (a stalled IPv6 route was found earlier)")

    # yt-dlp, only needed for add-url
    from .fetch import FetchError, ytdlp_cmd

    try:
        cmd = ytdlp_cmd()
        v = subprocess.run(cmd + ["--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        age = None
        try:
            y, m, d = (int(x) for x in v.split(".")[:3])
            age = (time.time() - time.mktime((y, m, d, 0, 0, 0, 0, 0, -1))) / 86400
        except ValueError:
            pass
        if age is not None and age > 120:
            add(WARN, "yt-dlp", f"{v}, {age:.0f} days old; old versions are the usual reason YouTube fails", "pip install -U yt-dlp")
        else:
            add(OK, "yt-dlp", v or "installed")
        if not (shutil.which("deno") or shutil.which("node")):
            add(WARN, "yt-dlp JS runtime", "no deno or node: some YouTube formats may be missing (captions usually still work)",
                "install deno (https://deno.com)")
    except FetchError:
        add(OK, "yt-dlp", "not installed (only needed for `whensaid add-url`)", "pip install 'whensaid[url]'")
    except Exception as e:
        add(WARN, "yt-dlp", f"installed but did not run: {e}")

    # the index
    dbp = paths.db_path()
    if not dbp.exists():
        add(OK, "index", f"none yet at {dbp}", "whensaid add <folder>  (or add-url <url>), then whensaid index")
    else:
        try:
            conn = db.connect_readonly(dbp)
            n = conn.execute("select count(*) from sources").fetchone()[0]
            chunks = conn.execute("select count(*) from chunks").fetchone()[0]
            size = dbp.stat().st_size / 1e6
            add(OK, "index", f"{n} sources, {chunks} chunks, {size:.0f} MB at {dbp}")
            built = db.get_meta(conn, "embed_model")
            from .embed import MODEL

            if built and built != MODEL:
                add(FAIL, "embedding model", f"the index was built with {built}, this version uses {MODEL}", "whensaid rechunk")
            bad = {r[0]: r[1] for r in conn.execute(
                "select status, count(*) from sources where status in ('failed', 'skipped') group by status")} \
                if n else {}
            for status, count in bad.items():
                add(WARN if status == "failed" else OK, f"{status} sources", f"{count}", "whensaid status  (shows the reasons)")
        except Exception as e:
            add(FAIL, "index", f"cannot be read: {e}", f"move {dbp} aside and re-index")
    if indexer_running():
        add(OK, "indexer", "running right now (search keeps working meanwhile)")
    return out


def run(out=sys.stdout) -> int:
    marks = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}
    results = checks()
    for status, name, detail, fix in results:
        print(f"[{marks[status]}] {name:18} {detail}", file=out)
        if fix and status != OK or (fix and status == OK and "none yet" in detail):
            print(f"       -> {fix}", file=out)
    bad = sum(1 for r in results if r[0] == FAIL)
    warn = sum(1 for r in results if r[0] == WARN)
    print(f"\n{bad} problems, {warn} warnings" if bad or warn else "\neverything looks fine", file=out)
    return 1 if bad else 0
