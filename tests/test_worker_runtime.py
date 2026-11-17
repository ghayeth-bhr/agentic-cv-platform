from agentic_cv.worker_runtime import create_export_worker


def test_export_worker_runtime_uses_shared_data_directory(tmp_path):
    runtime = create_export_worker(tmp_path / "data")

    assert runtime.worker.run_once() == ()
    assert (tmp_path / "data" / "metadata" / "jobs.sqlite3").exists()
    runtime.close()