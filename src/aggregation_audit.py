"""Local hash-chain audit for MPC rounds; no DP accounting or blockchain consensus."""
from __future__ import annotations

import json
import os
from pathlib import Path

from privacy_core import canonical, digest, vector
from secure_aggregation import PRIME, SCHEMA


class AggregationAudit:
    def __init__(self, path, namespace, clients, dimension, clip, scale):
        self.path = Path(path)
        self.header = {"kind": "policy", "version": 3, "namespace": namespace,
                       "clients": sorted(clients), "dimension": dimension, "clip": clip,
                       "scale": scale, "mechanism": "two_server_additive_secure_sum",
                       "differential_privacy": False}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._append(self.header, "0" * 64, 0)
        self.verify()

    def _append(self, event, previous, index):
        block = {"index": index, "previous": previous, "event": event}
        block["hash"] = digest(block)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(block).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return block["hash"]

    def verify(self, expected_head=None):
        pending, finished, head, size = {}, set(), "0" * 64, 0
        sessions = {}
        with self.path.open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                block = json.loads(line)
                if (set(block) != {"index", "previous", "event", "hash"} or block["index"] != i or
                        block["previous"] != head or digest({k: block[k] for k in ("index", "previous", "event")}) != block["hash"]):
                    raise ValueError("MPC audit hash chain changed")
                event = block["event"]
                if i == 0:
                    if event != self.header:
                        raise ValueError("MPC audit policy changed")
                else:
                    key = event.get("contract_hash")
                    if not isinstance(key, str) or len(key) != 64:
                        raise ValueError("Invalid audit contract")
                    if event.get("kind") == "begin":
                        if set(event) != {"kind", "contract_hash", "contract"} or key in pending or key in finished or digest(event["contract"]) != key:
                            raise ValueError("Changed or replayed audit contract")
                        data = event["contract"]
                        self._validate_contract(data)
                        state = sessions.setdefault(data["session"], self._initial_session())
                        if (state["pending"] is not None or data["round_index"] != state["last_round"] + 1 or
                                data["parent_model"] != state["last_model"]):
                            raise ValueError("Out-of-order or stale MPC audit contract")
                        state["pending"] = key
                        pending[key] = event["contract"]
                    elif event.get("kind") == "aggregate":
                        if set(event) != {"kind", "contract_hash", "model_hash", "receipt_hash"} or key not in pending or key in finished:
                            raise ValueError("Unreserved or replayed audit aggregate")
                        for field in ("model_hash", "receipt_hash"):
                            if not isinstance(event[field], str) or len(event[field]) != 64 or any(c not in "0123456789abcdef" for c in event[field]):
                                raise ValueError("Invalid aggregate commitment")
                        data = pending[key]
                        state = sessions[data["session"]]
                        if state["pending"] != key:
                            raise ValueError("MPC aggregate contract is not active")
                        state.update(last_round=data["round_index"], last_model=event["model_hash"], pending=None)
                        finished.add(key)
                    else:
                        raise ValueError("Unknown MPC audit event")
                head, size = block["hash"], i + 1
        if not size or expected_head is not None and expected_head != head:
            raise ValueError("MPC audit missing history or pinned head differs")
        self.head, self.size, self.pending, self.finished, self.sessions = head, size, pending, finished, sessions
        return {"head": head, "completed_rounds": len(finished), "reserved_rounds": len(pending)}

    def _initial_session(self):
        return {"last_round": 0, "last_model": digest([0.] * self.header["dimension"]), "pending": None}

    def _validate_contract(self, data):
        required = {"schema", "session", "round_index", "parent_model", "clients", "dimension", "clip", "scale", "prime"}
        if (not isinstance(data, dict) or set(data) != required or data["schema"] != SCHEMA or data["prime"] != PRIME or
                not isinstance(data["clients"], list) or sorted(data["clients"]) != self.header["clients"] or
                len(set(data["clients"])) != len(data["clients"]) or
                type(data["dimension"]) is not int or data["dimension"] != self.header["dimension"] or
                data["clip"] != self.header["clip"] or type(data["scale"]) is not int or data["scale"] != self.header["scale"] or
                not isinstance(data["session"], str) or not data["session"] or
                type(data["round_index"]) is not int or data["round_index"] < 1 or
                not isinstance(data["parent_model"], str) or len(data["parent_model"]) != 64 or
                any(c not in "0123456789abcdef" for c in data["parent_model"])):
            raise ValueError("MPC contract differs from fixed audit policy")

    def reserve(self, contract):
        self.verify()
        data = contract.as_dict()
        self._validate_contract(data)
        key = digest(data)
        state = self.sessions.get(data["session"], self._initial_session())
        if (key != contract.contract_hash or key in self.pending or state["pending"] is not None or
                data["round_index"] != state["last_round"] + 1 or data["parent_model"] != state["last_model"]):
            raise ValueError("Changed or replayed MPC contract")
        self._append({"kind": "begin", "contract_hash": key, "contract": data}, self.head, self.size)
        self.verify()

    def commit(self, contract, model, receipt):
        self.verify()
        key = contract.contract_hash
        model = vector(model, self.header["dimension"])
        model_hash = digest(model.tolist())
        if (key not in self.pending or key in self.finished or contract.as_dict() != self.pending[key] or
                receipt.get("contract_hash") != key or receipt.get("parent_model") != contract.parent_model or
                receipt.get("model_hash") != model_hash):
            raise ValueError("MPC round was not reserved or is already committed")
        self._append({"kind": "aggregate", "contract_hash": key,
                      "model_hash": model_hash, "receipt_hash": digest(receipt)}, self.head, self.size)
        return self.verify()
