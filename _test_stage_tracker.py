import time
import json

from aml_fraud_detector.utils.stage_tracker import StageTracker, StageRecord


def test_stage_record_basic():
    print("=== Test 1: StageRecord basic ===")
    rec = StageRecord(stage_name="TestStage")
    assert rec.stage_name == "TestStage"
    assert rec.status == "pending"
    assert rec.error is None
    assert rec.input_paths == []
    assert rec.output_paths == []
    d = rec.to_dict()
    assert d["stage_name"] == "TestStage"
    assert d["status"] == "pending"
    print("  ✓ StageRecord basic works")


def test_tracker_complete_stage():
    print("\n=== Test 2: Tracker complete stage ===")
    t = StageTracker()
    t.start_stage("StageA", input_paths=["/in/a.csv"])
    time.sleep(0.05)
    t.set_output_paths(["/out/b.csv", "/out/c.pkl"])
    t.complete_stage()

    records = t.records
    assert len(records) == 1
    r = records[0]
    assert r.stage_name == "StageA"
    assert r.status == "completed"
    assert r.start_time is not None
    assert r.end_time is not None
    assert r.duration_seconds is not None
    assert r.duration_seconds >= 0.04
    assert r.input_paths == ["/in/a.csv"]
    assert r.output_paths == ["/out/b.csv", "/out/c.pkl"]
    assert r.error is None
    print(f"  ✓ completed: duration={r.duration_seconds:.3f}s")


def test_tracker_fail_stage():
    print("\n=== Test 3: Tracker fail stage ===")
    t = StageTracker()
    t.start_stage("StageB", input_paths=["/in/x.csv"])
    time.sleep(0.02)
    try:
        raise ValueError("something went wrong")
    except Exception as e:
        t.fail_stage(e)

    records = t.records
    assert len(records) == 1
    r = records[0]
    assert r.stage_name == "StageB"
    assert r.status == "failed"
    assert r.duration_seconds is not None
    assert r.duration_seconds >= 0.01
    assert r.error is not None
    assert r.error["error_type"] == "ValueError"
    assert r.error["error_message"] == "something went wrong"
    print(f"  ✓ failed: error_type={r.error['error_type']}, duration={r.duration_seconds:.3f}s")


def test_context_manager_success():
    print("\n=== Test 4: Context manager success ===")
    t = StageTracker()
    with t.track_stage("StageC", input_paths=["/in.csv"], output_paths=["/out.csv"]):
        time.sleep(0.01)

    assert len(t.records) == 1
    r = t.records[0]
    assert r.status == "completed"
    assert r.input_paths == ["/in.csv"]
    assert r.output_paths == ["/out.csv"]
    assert r.duration_seconds >= 0.005
    print("  ✓ context manager success")


def test_context_manager_failure():
    print("\n=== Test 5: Context manager failure ===")
    t = StageTracker()
    try:
        with t.track_stage("StageD", input_paths=["/bad.csv"]):
            raise RuntimeError("boom")
    except RuntimeError as e:
        pass

    assert len(t.records) == 1
    r = t.records[0]
    assert r.status == "failed"
    assert r.error["error_type"] == "RuntimeError"
    assert r.error["error_message"] == "boom"
    assert r.input_paths == ["/bad.csv"]
    assert r.duration_seconds is not None
    print("  ✓ context manager re-raises and records failure")


def test_multiple_stages_mixed():
    print("\n=== Test 6: Multiple stages (success + failure) ===")
    t = StageTracker()

    with t.track_stage("DataIngestion", input_paths=["src.csv"]):
        t.set_output_paths(["train.csv", "test.csv"])
        time.sleep(0.01)

    with t.track_stage("DataValidation", input_paths=["train.csv", "test.csv"],
                       output_paths=["train.csv", "test.csv"]):
        time.sleep(0.005)

    try:
        with t.track_stage("DataTransformation", input_paths=["train.csv"]):
            time.sleep(0.02)
            raise ValueError("column missing")
    except ValueError:
        pass

    assert len(t.records) == 3

    r0, r1, r2 = t.records
    assert r0.status == "completed"
    assert r0.output_paths == ["train.csv", "test.csv"]
    assert r1.status == "completed"
    assert r2.status == "failed"
    assert r2.error["error_type"] == "ValueError"
    assert r2.error["error_message"] == "column missing"

    print(f"  ✓ 3 stages: completed={sum(1 for r in t.records if r.status == 'completed')}, "
          f"failed={sum(1 for r in t.records if r.status == 'failed')}")


def test_to_dict_list_serializable():
    print("\n=== Test 7: to_dict_list() JSON serializable ===")
    t = StageTracker()
    with t.track_stage("S1", input_paths=["/a"]):
        t.set_output_paths(["/b"])
    try:
        with t.track_stage("S2"):
            raise TypeError("bad type")
    except TypeError:
        pass

    dlist = t.to_dict_list()
    json_str = json.dumps(dlist, indent=2)
    data = json.loads(json_str)

    assert len(data) == 2
    assert data[0]["status"] == "completed"
    assert data[1]["status"] == "failed"
    assert "duration_seconds" in data[0]
    assert "error" in data[1]
    assert data[1]["error"]["error_type"] == "TypeError"
    print("  ✓ JSON round-trip works")


def test_fail_stage_preserves_output_paths():
    print("\n=== Test 8: Failed stage preserves input/output paths ===")
    t = StageTracker()
    t.start_stage("StageF", input_paths=["/in.csv"])
    t.set_output_paths(["/partial.pkl"])
    try:
        raise OSError("disk full")
    except Exception as e:
        t.fail_stage(e)

    r = t.records[0]
    assert r.status == "failed"
    assert r.input_paths == ["/in.csv"]
    assert r.output_paths == ["/partial.pkl"]
    assert r.error["error_type"] == "OSError"
    print("  ✓ failed stage has input_paths, output_paths, and error")


if __name__ == "__main__":
    test_stage_record_basic()
    test_tracker_complete_stage()
    test_tracker_fail_stage()
    test_context_manager_success()
    test_context_manager_failure()
    test_multiple_stages_mixed()
    test_to_dict_list_serializable()
    test_fail_stage_preserves_output_paths()
    print("\n" + "=" * 60)
    print("ALL 8 TESTS PASSED ✓")
    print("=" * 60)
