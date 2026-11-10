"""QC-15: standalone proof-witness verifier (standard-library only).

Full-witness records encode the stored binary64 operands of every altered MPO
core. This checker recomputes each update residual *exactly* as a rational,
checks exterior-context norms and polar defects, validates the causal-cone
operation schedule and core continuity, then replays the accumulated bound.

The separate 'ledger-v1' mode checks scalar inequalities only and is marked
conditional: supplied local residuals are NOT validated without witnesses.

No imports from QC-12..QC-14, NumPy, SciPy, BLAS or the simulator. The checker
is ordinary Python, with mathematical proof obligations documented separately.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
from fractions import Fraction as F
from dataclasses import dataclass
from math import isfinite, isqrt
from pathlib import Path
import struct
from typing import Any

SCHEMA = "qc15-full-witness-v1"
LEDGER_SCHEMA = "qc15-conditional-ledger-v1"
MAX_JSON_BYTES = 40_000_000
MAX_EVENTS = 5000
MAX_TENSOR_ENTRIES = 150_000
MAX_TOTAL_WITNESS_ENTRIES = 3_000_000
SQRT_BITS = 160
ZERO = (F(0), F(0))
ONE = (F(1), F(0))
I2 = (ONE, ZERO, ZERO, ONE)
Z2 = (ONE, ZERO, ZERO, (F(-1), F(0)))
X2 = (ZERO, ONE, ONE, ZERO)
Y2 = (ZERO, (F(0), F(-1)), (F(0), F(1)), ZERO)
PAULIS = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

class CertificateError(ValueError):
    pass


def require(cond: bool, msg: str):
    if not cond:
        raise CertificateError(msg)


def cint(x: Any, low: int = 0, high: int = 10**9) -> int:
    require(type(x) is int and low <= x <= high, "integer out of bounds")
    return x


def realhex(s: str, *, nonnegative=False) -> F:
    require(isinstance(s, str) and 0 < len(s) < 100, "missing binary64 hexadecimal scalar")
    try:
        x = float.fromhex(s)
    except (ValueError, OverflowError) as e:
        raise CertificateError("invalid binary64 hexadecimal scalar") from e
    require(isfinite(x) and (not nonnegative or x >= 0), "invalid or negative scalar")
    return F(x)


def plus(a, b): return a[0] + b[0], a[1] + b[1]
def minus(a, b): return a[0] - b[0], a[1] - b[1]
def times(a, b): return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]
def conj(a): return a[0], -a[1]
def norm2(a): return a[0]*a[0]+a[1]*a[1]


def sqrt_up(x: F, bits: int = SQRT_BITS) -> F:
    require(x >= 0, "negative squared norm")
    if x == 0:
        return F(0)
    scaled_n = x.numerator << (2 * bits)
    d = x.denominator
    candidate = isqrt(scaled_n // d)
    if candidate*candidate*d < scaled_n:
        candidate += 1
    return F(candidate, 1 << bits)


def _json_no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise CertificateError("duplicate JSON key")
        out[k] = v
    return out


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def read_proof(path: Path):
    path = Path(path)
    with path.open("rb") as h:
        first = h.read(2)
    if first == b"\x1f\x8b":
        with gzip.open(path, "rb") as h:
            raw = h.read(MAX_JSON_BYTES + 1)
    else:
        raw = path.read_bytes()
    require(len(raw) <= MAX_JSON_BYTES, "certificate exceeds decompressed byte limit")
    try:
        obj = json.loads(raw, object_pairs_hook=_json_no_duplicates,
                         parse_constant=lambda s: (_ for _ in ()).throw(CertificateError("nonfinite JSON")))
    except (UnicodeError, json.JSONDecodeError, OSError) as e:
        raise CertificateError("unreadable certificate") from e
    require(isinstance(obj, dict), "certificate must be an object")
    digest = obj.pop("sha256", None)
    require(isinstance(digest, str) and len(digest) == 64, "missing payload checksum")
    require(hashlib.sha256(_canonical_bytes(obj)).hexdigest() == digest, "payload checksum mismatch")
    return obj


def write_proof(obj: dict, path: Path):
    require("sha256" not in obj, "checksum already supplied")
    digest = hashlib.sha256(_canonical_bytes(obj)).hexdigest()
    data = _canonical_bytes({**obj, "sha256": digest})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with path.open("wb") as out:
            with gzip.GzipFile(filename="", fileobj=out, mode="wb", compresslevel=9, mtime=0) as gz:
                gz.write(data)
    else:
        path.write_bytes(data)


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    vals: tuple[tuple[F, F], ...]

    def __post_init__(self):
        require(len(self.shape) in (2, 4), "matrix or MPO core required")
        elements = 1
        for d in self.shape:
            elements *= cint(d, 1, 4096)
        require(elements <= MAX_TENSOR_ENTRIES and len(self.vals) == elements,
                "wrong tensor element count")
        if len(self.shape) == 4:
            require(self.shape[1:3] == (2, 2), "invalid MPO physical indices")

    def get4(self, l, a, b, r):
        return self.vals[((l * 2 + a)*2 + b)*self.shape[3] + r]

    def get2(self, i, j):
        return self.vals[i * self.shape[1] + j]


def read_tensor(obj: dict) -> Tensor:
    require(isinstance(obj, dict) and set(obj) == {"shape", "binary64_le"}, "bad tensor encoding")
    require(isinstance(obj["shape"], list) and len(obj["shape"]) in (2, 4), "tensor shape")
    shape = tuple(cint(x, 1, 4096) for x in obj["shape"])
    entries = 1
    for n in shape: entries *= n
    require(entries <= MAX_TENSOR_ENTRIES, "tensor entry cap")
    data = obj["binary64_le"]
    require(isinstance(data, str) and len(data) <= MAX_TENSOR_ENTRIES * 32, "tensor payload cap")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise CertificateError("invalid tensor base64") from e
    require(len(raw) == 16 * entries, "tensor byte-size mismatch")
    floats = struct.iter_unpack("<dd", raw)
    values = []
    for re, im in floats:
        require(isfinite(re) and isfinite(im), "nonfinite tensor coefficient")
        values.append((F(re), F(im)))
    return Tensor(shape, tuple(values))


def initial_core(label: str) -> Tensor:
    return Tensor((1,2,2,1), PAULIS[label])


def checked_chain(cores):
    require(bool(cores), "empty chain")
    require(cores[0].shape[0] == 1 and cores[-1].shape[-1] == 1, "open MPO boundary")
    for a,b in zip(cores,cores[1:]):
        require(a.shape[-1] == b.shape[0], "broken MPO bond")


def contracted_pair(a: Tensor, b: Tensor):
    require(len(a.shape) == len(b.shape) == 4, "MPO cores required")
    require(a.shape[-1] == b.shape[0], "pair bond mismatch")
    dl, dr, inner = a.shape[0], b.shape[-1], a.shape[-1]
    table = []
    for l in range(dl):
        block = []
        for r in range(dr):
            op = []
            for x in range(4):
                ax, bx = divmod(x, 2)
                row = []
                for y in range(4):
                    ay, by = divmod(y, 2)
                    v = ZERO
                    for bond in range(inner):
                        v = plus(v, times(a.get4(l,ax,ay,bond), b.get4(bond,bx,by,r)))
                    row.append(v)
                op.append(row)
            block.append(op)
        table.append(block)
    return table


def pair_error2(a,b,c,d,gate=None):
    require(a.shape[0] == c.shape[0] and b.shape[-1] == d.shape[-1],
            "pair outer bonds changed")
    orig = contracted_pair(a,b)
    newer = contracted_pair(c,d)
    if gate is not None:
        require(gate.shape == (4,4), "invalid gate matrix")
    sq=F(0)
    for l in range(a.shape[0]):
        for r in range(b.shape[-1]):
            for i in range(4):
                for j in range(4):
                    expected = orig[l][r][i][j]
                    if gate is not None:
                        expected = ZERO
                        for x in range(4):
                            for y in range(4):
                                expected=plus(expected,times(times(conj(gate.get2(x,i)),orig[l][r][x][y]),gate.get2(y,j)))
                    sq += norm2(minus(expected,newer[l][r][i][j]))
    return sq


def channel_error2(a,b,operators):
    require(a.shape == b.shape, "channel changed bond shape")
    sq=F(0)
    for l in range(a.shape[0]):
        for r in range(a.shape[-1]):
            for i in range(2):
                for j in range(2):
                    expected=ZERO
                    for k in operators:
                        for x in range(2):
                            for y in range(2):
                                expected=plus(expected,times(times(conj(k.get2(x,i)),a.get4(l,x,y,r)),k.get2(y,j)))
                    sq+=norm2(minus(expected,b.get4(l,i,j,r)))
    return sq


def gram_defect2(a: Tensor, orientation: str):
    require(len(a.shape) == 4, "Gram needs an MPO core")
    dl, dr = a.shape[0], a.shape[-1]
    dim = dr if orientation == "L" else dl
    require(orientation in ("L", "R"), "bad isometry orientation")
    sq=F(0)
    for i in range(dim):
        for j in range(dim):
            element=ZERO
            if orientation == "L":
                for l in range(dl):
                    for ao in range(2):
                        for ai in range(2):
                            element=plus(element,times(conj(a.get4(l,ao,ai,i)),a.get4(l,ao,ai,j)))
            else:
                for r in range(dr):
                    for ao in range(2):
                        for ai in range(2):
                            element=plus(element,times(a.get4(i,ao,ai,r),conj(a.get4(j,ao,ai,r))))
            if i==j:element=minus(element,ONE)
            sq+=norm2(element)
    return sq


def gate_defect2(u: Tensor):
    require(u.shape == (4,4), "expected 4x4 gate")
    return _stacked_defect2((u,), 4)


def kraus_defect2(ks):
    require(ks and all(k.shape == (2,2) for k in ks), "bad Kraus list")
    return _stacked_defect2(ks, 2)


def _stacked_defect2(mats, dim):
    sq=F(0)
    for i in range(dim):
        for j in range(dim):
            element=ZERO
            for u in mats:
                for x in range(dim):
                    element=plus(element,times(conj(u.get2(x,i)),u.get2(x,j)))
            if i==j:element=minus(element,ONE)
            sq+=norm2(element)
    return sq


def frob_norm2(a: Tensor):
    return sum((norm2(v) for v in a.vals), F(0))


def context_bound(cores, excluded, *, gate=False):
    lo=min(excluded);hi=max(excluded)
    product=F(1)
    for idx, core in enumerate(cores):
        if idx in excluded:continue
        if gate and idx<lo:
            product *= 1 + sqrt_up(gram_defect2(core,"L"))
        elif gate and idx>hi:
            product *= 1 + sqrt_up(gram_defect2(core,"R"))
        else:
            product *= sqrt_up(frob_norm2(core))
    return product


def exact_readout(cores):
    vector=[ONE]
    for core in cores:
        out=[]
        for j in range(core.shape[-1]):
            v=ZERO
            for i in range(core.shape[0]):
                v=plus(v,times(vector[i],core.get4(i,0,0,j)))
            out.append(v)
        vector=out
    require(len(vector)==1, "unclosed readout")
    return vector[0]


def expected_cone(qubits, layers, targets):
    current=set(targets)
    states=[None]*(len(layers)+1)
    states[-1]=tuple(sorted(current))
    for t in range(len(layers)-1,-1,-1):
        seen=set()
        for j,_ in layers[t]:
            require(0<=j<qubits-1 and j not in seen and j+1 not in seen,
                    "invalid or overlapping nearest-neighbor gate")
            seen.update((j,j+1))
        for j,_ in layers[t]:
            if j in current or j+1 in current:
                current.update((j,j+1))
        states[t]=tuple(sorted(current))
        require(states[t]==tuple(range(min(current),max(current)+1)),
                "noncontiguous backward cone")
    return states


def check_witness(proof, *, kernel=None, event_source=None, on_boundary=None):
    require(proof.get("format") == SCHEMA, "full-witness format required")
    circuit=proof.get("circuit")
    require(isinstance(circuit,dict),"missing circuit")
    n=cint(circuit.get("qubits"),2,64)
    targets=circuit.get("targets")
    label=circuit.get("pauli")
    require(isinstance(targets,list) and targets and targets==list(range(targets[0],targets[-1]+1))
            and all(type(j) is int and 0<=j<n for j in targets),"invalid contiguous targets")
    require(isinstance(label,str) and len(label)==len(targets) and set(label)<=set(PAULIS),
            "invalid observable labels")
    layers=[]
    require(isinstance(circuit.get("layers"),list) and len(circuit["layers"])<=64,
            "invalid layers")
    for layer in circuit["layers"]:
        require(isinstance(layer,list),"layer is not a list")
        out=[]
        for item in layer:
            require(isinstance(item,dict) and set(item)=={"site","gate"},"bad gate entry")
            out.append((cint(item["site"],0,n-2),read_tensor(item["gate"])))
        layers.append(out)
    ops=circuit.get("kraus")
    require(isinstance(ops,list) and 1<=len(ops)<=10,"Kraus operator list size")
    ops=[read_tensor(x) for x in ops]
    require(all(k.shape==(2,2) for k in ops),"Kraus shape")
    states=expected_cone(n,layers,targets)
    require(len(states[0])<=64,"cone too wide")
    transcript=proof.get("events") if event_source is None else event_source
    require((isinstance(transcript,list) or event_source is not None)
            and len(transcript)<=MAX_EVENTS+64,"event limit")
    claimed=proof.get("result")
    require(isinstance(claimed,dict),"missing result")
    require(cint(claimed.get("cone_width"),1,64)==len(states[0]),"cone width mismatch")
    channel_defect=sqrt_up((kernel.kraus_defect2 if kernel is not None else kraus_defect2)(ops))
    require(channel_defect<F(1,2),"Kraus map lacks polar-isometry correction")
    ch_alpha=channel_defect*(2+channel_defect)
    gate_alpha={}
    for layer in layers:
        for _,u in layer:
            d=sqrt_up((kernel.gate_defect2 if kernel is not None else gate_defect2)(u))
            require(d<F(1,2),"gate lacks polar-unitary correction")
            gate_alpha[id(u)]=d*(2+d)
    first,last=targets[0],targets[-1]
    cores=[initial_core(ch) for ch in label]
    index=0
    accumulated=F(0)
    whole=0
    tally={"channel":0,"left_qr":0,"right_qr":0,"gate_svd":0}
    if on_boundary is not None:
        on_boundary(0,first,last,cores,accumulated,tally)
    def apply_event(kind,t,site,indices,gate=None):
        nonlocal index, accumulated, whole
        require(index<len(transcript),"missing event")
        ev=transcript[index]
        index+=1
        require(isinstance(ev,dict) and ev.get("kind")==kind
                and ev.get("layer")==t and ev.get("site")==site,
                "event kind/layer/site mismatch")
        require(ev.get("indices")==indices and isinstance(ev.get("new_cores"),list)
                and len(ev["new_cores"])==len(indices),"event changed-core set mismatch")
        after=list(cores)
        for pos,obj in zip(indices,ev["new_cores"]):
            after[pos]=read_tensor(obj)
            whole+=len(after[pos].vals)
            require(whole<=MAX_TOTAL_WITNESS_ENTRIES,"witness entry budget")
        checked_chain(after)
        if kind=="channel":
            squared=(kernel.channel_error2 if kernel is not None else channel_error2)(
                cores[indices[0]],after[indices[0]],ops)
            needed_alpha=ch_alpha
        elif kind=="gate_svd":
            squared=(kernel.pair_error2 if kernel is not None else pair_error2)(
                cores[indices[0]],cores[indices[1]],after[indices[0]],after[indices[1]],gate)
            needed_alpha=gate_alpha[id(gate)]
        else:
            squared=(kernel.pair_error2 if kernel is not None else pair_error2)(
                cores[indices[0]],cores[indices[1]],after[indices[0]],after[indices[1]])
            needed_alpha=F(0)
        r=realhex(ev.get("residual_upper"),nonnegative=True)
        g=realhex(ev.get("context_upper"),nonnegative=True)
        alpha=realhex(ev.get("polar_upper"),nonnegative=True)
        require(r*r>=squared, "invalid exact local residual upper")
        require(g>=(kernel.context_bound if kernel is not None else context_bound)(
                    after,indices,gate=(kind=="gate_svd")),
                "invalid external-context norm upper")
        require(alpha>=needed_alpha,"invalid gate/channel polar allowance")
        require(alpha==0 if kind not in ("gate_svd","channel") else True,
                "QR cannot have polar correction")
        accumulated += r*g+alpha*(1+accumulated)
        cores[:] = after
        tally[kind] += 1
        if on_boundary is not None:
            on_boundary(index,first,last,cores,accumulated,tally)
    for t in range(len(layers)-1,-1,-1):
        after_state=states[t+1]
        for site in range(first,last+1):
            apply_event("channel",t,site,[site-first])
        require(index<len(transcript),"missing grow event")
        ev=transcript[index]
        index+=1
        low,high=states[t][0],states[t][-1]
        require(ev=={"kind":"grow","layer":t,"first":low,"last":high},"grow event mismatch")
        require(low<=first and high>=last,"cone support shrank")
        cores[:0]=[initial_core("I") for _ in range(first-low)]
        cores.extend(initial_core("I") for _ in range(high-last))
        first,last=low,high
        checked_chain(cores)
        if on_boundary is not None:
            on_boundary(index,first,last,cores,accumulated,tally)
        for site,u in layers[t]:
            if site in after_state or site+1 in after_state:
                j=site-first
                require(0<=j<len(cores)-1,"gate outside cone")
                for k in range(j):
                    apply_event("left_qr",t,first+k,[k,k+1])
                for k in range(len(cores)-1,j+1,-1):
                    apply_event("right_qr",t,first+k,[k-1,k])
                apply_event("gate_svd",t,site,[j,j+1],u)
    require(index==len(transcript),"unused trailing events")
    checked_chain(cores)
    value=(kernel.exact_readout if kernel is not None else exact_readout)(cores)
    reported=realhex(claimed.get("expectation"))
    error=realhex(claimed.get("error_upper"),nonnegative=True)
    require(error>=accumulated+abs(value[0]-reported),"end-to-end bound understated")
    require(abs(value[1])<=error,"unaccounted imaginary readout")
    require(claimed.get("event_counts")==tally,"event counts mismatch")
    return {"proof_level":"full-witness", "events_checked":sum(tally.values()),
            "event_counts":tally,"cone_width":len(states[0]),
            "reported_expectation":float(reported),"verified_bound":float(error),
            "arithmetic_lower_requirement":float(accumulated),
            "witness_complex_entries":whole,
            "target":"polar-corrected binary64 gate/Kraus circuit"}


def check_ledger(proof):
    require(proof.get("format")==LEDGER_SCHEMA,"invalid conditional-ledger format")
    events=proof.get("events")
    require(isinstance(events,list) and len(events)<=MAX_EVENTS,"conditional ledger length")
    acc=F(0)
    count=0
    for ev in events:
        require(isinstance(ev,dict) and ev.get("kind") in
                ("channel","left_qr","right_qr","gate_svd"),"bad ledger event")
        r=realhex(ev.get("residual_upper"),nonnegative=True)
        g=realhex(ev.get("context_upper"),nonnegative=True)
        alpha=realhex(ev.get("polar_upper"),nonnegative=True)
        if ev["kind"] in ("left_qr","right_qr"):
            require(alpha==0,"QR allowance must be zero")
        count+=1
        acc+=r*g+alpha*(1+acc)
    result=proof.get("result")
    require(isinstance(result,dict),"missing result")
    numerator=result.get("readout_real_numerator_hex")
    require(isinstance(numerator,str) and 0<len(numerator)<=100_000
            and numerator.lstrip("-").startswith("0x")
            and all(ch in "0123456789abcdef" for ch in numerator.lstrip("-")[2:]),
            "invalid dyadic numerator encoding")
    exact=F(int(numerator,16),1<<cint(result.get("readout_denominator_exp"),0,100_000))
    reported=realhex(result.get("expectation"))
    error=realhex(result.get("error_upper"),nonnegative=True)
    require(error>=acc+abs(exact-reported),"conditional ledger bound understated")
    require(count==cint(result.get("event_count"),0,MAX_EVENTS),"ledger count mismatch")
    return {"proof_level":"conditional-ledger-only","events_checked":count,
            "reported_expectation":float(reported),"verified_bound":float(error),
            "warning":"local residual and context claims are supplied by producer; no operand witnesses"}


def check_certificate(path: Path, *, require_witness=False):
    obj=read_proof(Path(path))
    if obj.get("format")==SCHEMA:
        return check_witness(obj)
    if obj.get("format")==LEDGER_SCHEMA and not require_witness:
        return check_ledger(obj)
    raise CertificateError("full operand witness required")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate",type=Path)
    parser.add_argument("--require-witness",action="store_true")
    args=parser.parse_args()
    try:
        result=check_certificate(args.certificate,require_witness=args.require_witness)
    except (CertificateError,ValueError,OverflowError,MemoryError,KeyError,TypeError) as ex:
        parser.exit(2, "FAIL: "+str(ex)+"\n")
    print(json.dumps({"status":"PASS",**result},sort_keys=True))


if __name__=="__main__":
    main()
