import json

import pytest

from stfr.agent import NOTICE, run_episode
from stfr.faults import Fault
from stfr.tasks import Task
from tests.test_svc import UPD, mk
from stfr.triggers import TRIGGERS
from stfr.llm import Reply


class ScriptedChat:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, *, seed, temperature=0.0):
        self.calls.append([dict(m) for m in messages])
        return Reply(self.replies.pop(0) if self.replies else "{}", 10, 5, 0.01, cached=len(self.calls) == 1)


def call(tool, args, conf=None):
    d = {"thought": "t", "tool": tool, "args": args}
    if conf is not None:
        d["confidence"] = conf
    return json.dumps(d)


DONE = json.dumps({"done": True, "summary": "ok"})
GET = {"collection": "orders", "id": "o1"}


def task(initial=None):
    return Task("t0", "Ship order o1.", initial or mk().state, {"orders/o1/status": "shipped"},
                [["orders", "o1"]], [], 1, 0)


def ep(replies, trigger="never", fault=Fault.CLEAN, response="recover", t=None, **kw):
    chat = ScriptedChat(replies)
    return run_episode(t or task(), chat, fault=fault, slot=0, trigger=trigger, response=response,
                       threshold=0.5, seed=1, **kw), chat


def test_clean_episode_reaches_done():
    e, chat = ep([call("update_field", UPD), DONE])
    assert e.status == "done" and e.final_state["orders"]["o1"]["status"] == "shipped"
    assert (e.tokens_in, e.tokens_out) == (20, 10) and e.cost_usd == 0.01  # cost skips the cached reply
    assert e.fault_applied is False and e.flags == []
    assert chat.calls[1][-1] == {"role": "user", "content": 'RESULT step 1: {"ok": true, "record": {"customer": "ann", "item": "pen", "qty": 2, "status": "shipped"}}'}
    json.dumps(e.steps)


def test_notice_text_identical_across_triggers():
    bad = mk().state
    bad["orders"]["o1"]["status"] = "lost"  # contract fires on get_record
    e1, c1 = ep([call("get_record", GET), DONE], "on_contract", t=task(bad))
    e2, c2 = ep([call("update_field", UPD), DONE], "on_readback", Fault.SILENT_DROP)
    assert e1.flags == [1] and e2.flags == [1] and e2.fault_applied and e2.fault_applied_call == 0
    assert c1.calls[1][-1] == {"role": "user", "content": NOTICE.format(k=1, tool="get_record")}
    assert c2.calls[1][-1] == {"role": "user", "content": NOTICE.format(k=1, tool="update_field")}
    assert e2.readback_calls == 1


def test_notice_byte_equal_for_different_triggers_same_tool_and_step():
    runs = [ep([call("update_field", UPD), DONE], trig, Fault.SILENT_DROP) for trig in ("on_readback", "always")]
    assert all(e.flags == [1] for e, _ in runs)
    assert runs[0][1].calls[1][-1] == runs[1][1].calls[1][-1]
    assert runs[0][1].calls[1][-1]["content"] == NOTICE.format(k=1, tool="update_field")


def test_abstain_stops_on_first_flag():
    e, chat = ep([call("update_field", UPD), DONE], "always", response="abstain")
    assert e.status == "abstained" and e.flags == [1] and len(chat.calls) == 1


def test_never_trigger_appends_no_notice():
    e, chat = ep([call("update_field", UPD), DONE])
    assert not any("NOTICE" in m["content"] for m in chat.calls[-1])


def test_malformed_json_is_an_error_step_and_three_end_episode():
    e, chat = ep(["not json", "{bad", call("get_record", GET), "x", "y", "z"])
    assert e.status == "parse_failure" and len(e.steps) == 6 and len(chat.calls) == 6  # reset by the valid call
    assert chat.calls[1][-1]["content"].startswith("ERROR step 1: invalid action:")
    assert e.flags == []  # invalid steps never reach triggers


def test_unknown_tool_is_an_error_step():
    e, chat = ep([call("drop_db", {}), call("update_field", UPD), DONE], "always")
    assert chat.calls[1][-1]["content"].startswith("ERROR step 1: invalid action:") and e.status == "done"
    assert e.flags == [2]


def test_step_budget_exhausted():
    e, _ = ep([call("get_record", GET)] * 5, max_steps=3)
    assert e.status == "budget_exhausted" and len(e.steps) == 3


def test_missing_confidence_defaults_to_one():
    e, _ = ep([call("get_record", GET), call("get_record", GET, 0.1), DONE], "on_uncertainty")
    assert e.steps[0]["confidence"] == 1.0 and e.flags == [2]


@pytest.mark.parametrize("reply", [
    '{"thought": "t", "tool": ["get_record"], "args": {}}', '{"tool": {}, "args": {}}',
    '[1, 2]', '42', '"get_record"', '{"tool": "get_record", "args": []}'])
def test_odd_json_shapes_are_invalid_actions_not_crashes(reply):
    e, chat = ep([reply, DONE])
    assert e.status == "done" and e.flags == []
    assert chat.calls[1][-1]["content"].startswith("ERROR step 1: invalid action:")


@pytest.mark.parametrize("conf,want", [('"high"', 1.0), ("NaN", 1.0), ("Infinity", 1.0), ("-Infinity", 1.0),
                                       ("5", 1.0), ("-3", 0.0), ("0.25", 0.25), ("true", 1.0)])
def test_confidence_sanitised(conf, want):
    r = '{"tool": "get_record", "args": %s, "confidence": %s}' % (json.dumps(GET), conf)
    e, _ = ep([r, DONE])
    assert e.steps[0]["confidence"] == want


def test_nan_confidence_does_not_silently_skip_uncertainty():
    r = '{"tool": "get_record", "args": %s, "confidence": NaN}' % json.dumps(GET)
    e, _ = ep([r, DONE], "on_uncertainty")
    assert e.flags == []  # NaN -> 1.0, a confident step, by design (documented, not a NaN comparison accident)


def test_step_before_is_dict_for_writes_and_none_for_reads(monkeypatch):
    seen = []
    monkeypatch.setitem(TRIGGERS, "never", lambda s, c: seen.append(s) or False)
    ep([call("get_record", GET), call("update_field", UPD), DONE])
    assert [s.tool for s in seen] == ["get_record", "update_field"]
    assert seen[0].before is None and seen[1].before["orders"]["o1"]["status"] == "pending"
    e, _ = ep([call("update_field", UPD), DONE], "on_readback")  # clean write: no ValueError, no flag
    assert e.flags == []
