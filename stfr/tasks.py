"""Task generator (6 templates), deterministic fault slots, sealed split files."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .faults import Fault, _eligible
from .svc import TOOLS, Service

DATA = Path(__file__).parent / "data"
SPLITS = {"dev": range(0, 30), "testA": range(10000, 10060), "testB": range(20000, 20060)}
NAMES = ["ann", "bob", "cy", "dee", "eli", "fay"]
SKUS = ["pen", "ink", "pad", "cap", "tin", "mug", "jar", "key"]


class SealedSplit(Exception):
    pass


class SealBroken(Exception):
    pass


@dataclass
class Task:
    id: str
    prompt: str
    initial: dict
    goal: dict
    allowed: list
    ref_steps: list
    n_writes: int
    n_reads: int


def _step(tool, **args):
    return {"tool": tool, "args": args}


def _base(rng):
    skus = rng.sample(SKUS, 4)
    inv = {f"i{n}": {"sku": k, "stock": rng.randint(8, 50), "price": 10 * rng.randint(1, 10)}
           for n, k in enumerate(skus, 1)}
    orders = {f"o{n}": {"customer": rng.choice(NAMES), "item": rng.choice(skus), "qty": rng.randint(1, 5),
                        "status": rng.choice(["pending", "shipped", "cancelled"])} for n in range(1, 6)}
    accs = {f"a{n}": {"owner": o, "balance": rng.randint(200, 1000), "tier": rng.choice(["basic", "gold"])}
            for n, o in enumerate(rng.sample(NAMES, 3), 1)}
    return {"orders": orders, "inventory": inv, "accounts": accs}


def _iid(st, sku):
    return next(i for i, r in st["inventory"].items() if r["sku"] == sku)


def generate(seed: int) -> Task:
    rng = random.Random(seed)
    st = _base(rng)
    o, inv, acc = st["orders"], st["inventory"], st["accounts"]
    oid = f"o{rng.randint(1, 5)}"
    iid = f"i{rng.randint(1, 4)}"
    t = seed % 6  # round-robin so every template (incl. transfer/create) appears equally often
    if t == 0:  # restock an item, then mark the order ready
        o[oid]["status"] = "pending"
        new = inv[iid]["stock"] + rng.randint(10, 30)
        prompt = (f"Inventory record {iid} (item {inv[iid]['sku']}) is running low. Set its stock to {new}, "
                  f"then mark order {oid} as ready.")
        steps = [_step("get_record", collection="inventory", id=iid),
                 _step("update_field", collection="inventory", id=iid, field="stock", value=new),
                 _step("update_field", collection="orders", id=oid, field="status", value="ready")]
        goal = {f"inventory/{iid}/stock": new, f"orders/{oid}/status": "ready"}
        allowed = [["inventory", iid], ["orders", oid]]
    elif t == 1:  # move funds, then mark the order shipped
        o[oid]["status"] = "pending"
        a, b = rng.sample(sorted(acc), 2)
        amt = 10 * rng.randint(1, 10)
        prompt = (f"Order {oid} has been paid for. Transfer {amt} from account {a} to account {b}, "
                  f"then mark order {oid} as shipped.")
        steps = [_step("get_record", collection="accounts", id=a),
                 _step("transfer", from_id=a, to_id=b, amount=amt),
                 _step("update_field", collection="orders", id=oid, field="status", value="shipped")]
        goal = {f"accounts/{a}/balance": acc[a]["balance"] - amt, f"accounts/{b}/balance": acc[b]["balance"] + amt,
                f"orders/{oid}/status": "shipped"}
        allowed = [["accounts", a], ["accounts", b], ["orders", oid]]
    elif t == 2:  # cancel an order and return its stock
        o[oid]["status"] = "pending"
        i = _iid(st, o[oid]["item"])
        prompt = (f"Cancel order {oid} and return its quantity to the stock of its item "
                  f"(find the item's inventory record by sku).")
        steps = [_step("get_record", collection="orders", id=oid),
                 _step("list_records", collection="inventory", where={"sku": o[oid]["item"]}),
                 _step("update_field", collection="orders", id=oid, field="status", value="cancelled"),
                 _step("update_field", collection="inventory", id=i, field="stock",
                       value=inv[i]["stock"] + o[oid]["qty"])]
        goal = {f"orders/{oid}/status": "cancelled", f"inventory/{i}/stock": inv[i]["stock"] + o[oid]["qty"]}
        allowed = [["orders", oid], ["inventory", i]]
    elif t == 3:  # create an order for an in-stock item and decrement stock
        cust, qty = rng.choice(NAMES), rng.randint(1, 5)
        sku = inv[iid]["sku"]
        new = f"o{max(int(k[1:]) for k in o) + 1}"
        prompt = (f"Create a new pending order for customer {cust}: {qty} units of item {sku} "
                  f"(find the item's inventory record by sku). Then decrease that item's stock by {qty}.")
        fields = {"customer": cust, "item": sku, "qty": qty, "status": "pending"}
        steps = [_step("list_records", collection="inventory", where={"sku": sku}),
                 _step("create_record", collection="orders", fields=fields),
                 _step("update_field", collection="inventory", id=iid, field="stock", value=inv[iid]["stock"] - qty)]
        goal = {**{f"orders/{new}/{k}": v for k, v in fields.items()}, f"inventory/{iid}/stock": inv[iid]["stock"] - qty}
        allowed = [["orders", new], ["inventory", iid]]
    elif t == 4:  # find the one 'ready' order, ship it, decrement its item's stock
        for v in o.values():
            if v["status"] == "ready":
                v["status"] = "pending"
        o[oid]["status"] = "ready"
        i = _iid(st, o[oid]["item"])
        q = o[oid]["qty"]
        prompt = (f"Find the order whose status is 'ready' (use list_records on orders) and mark it shipped. "
                  f"Then reduce the stock of its item by the order's quantity (find the inventory record by sku).")
        steps = [_step("list_records", collection="orders", where={"status": "ready"}),
                 _step("list_records", collection="inventory", where={"sku": o[oid]["item"]}),
                 _step("update_field", collection="orders", id=oid, field="status", value="shipped"),
                 _step("update_field", collection="inventory", id=i, field="stock", value=inv[i]["stock"] - q)]
        goal = {f"orders/{oid}/status": "shipped", f"inventory/{i}/stock": inv[i]["stock"] - q}
        allowed = [["orders", oid], ["inventory", i]]
    else:  # 10% discount on the items of a customer's two pending orders
        cust = rng.choice(NAMES)
        s1, s2 = rng.sample([v["sku"] for v in inv.values()], 2)
        for n, v in o.items():
            v["customer"] = rng.choice([x for x in NAMES if x != cust])
            if v["status"] == "pending":
                v["status"] = "shipped"
        for n, s in (("o1", s1), ("o2", s2)):
            o[n].update(customer=cust, item=s, status="pending")
        prompt = (f"Customer {cust} gets a 10% discount. Find all of {cust}'s pending orders (use list_records "
                  f"on orders). For the item on each of those orders, set the item's inventory price to 90% "
                  f"of its current price (find the inventory record by sku).")
        steps, goal, allowed = [_step("list_records", collection="orders", where={"customer": cust, "status": "pending"})], {}, []
        for s in (s1, s2):
            i = _iid(st, s)
            steps += [_step("list_records", collection="inventory", where={"sku": s}),
                      _step("update_field", collection="inventory", id=i, field="price", value=inv[i]["price"] * 9 // 10)]
            goal[f"inventory/{i}/price"] = inv[i]["price"] * 9 // 10
            allowed.append(["inventory", i])
    w = sum(TOOLS[s["tool"]].kind == "write" for s in steps)
    return Task(f"t{seed}", prompt, st, goal, allowed, steps, w, len(steps) - w)


def fault_slot(task: Task, fault: Fault, seed: int) -> int | None:
    """Index-of-kind of an eligible ref_steps call (reads for SILENT_READ, else writes), or None."""
    if fault == Fault.CLEAN:
        return 0
    kind = "read" if fault == Fault.SILENT_READ else "write"
    s, n, ok = Service(task.initial), 0, []
    for st in task.ref_steps:
        res = s.call(st["tool"], st["args"])
        if TOOLS[st["tool"]].kind == kind:
            if _eligible(fault, st["tool"], res):
                ok.append(n)
            n += 1
    return random.Random(f"{task.id}|{fault.value}|{seed}").choice(ok) if ok else None


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_splits(dir: Path) -> None:
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    for name, seeds in SPLITS.items():
        (dir / f"{name}.jsonl").write_text("".join(json.dumps(asdict(generate(s)), sort_keys=True) + "\n" for s in seeds))
    (dir / "SEALED.json").write_text(json.dumps({n: _sha(dir / f"{n}.jsonl") for n in ("testA", "testB")}, indent=1))


def load_split(name: str, *, allow_sealed: bool = False) -> list[Task]:
    sealed = name in ("testA", "testB")
    if sealed and not allow_sealed:
        raise SealedSplit(f"{name} is sealed")
    p = DATA / f"{name}.jsonl"
    if sealed and _sha(p) != json.loads((DATA / "SEALED.json").read_text())[name]:
        raise SealBroken(f"{name} does not match SEALED.json")
    return [Task(**json.loads(l)) for l in p.read_text().splitlines() if l]
