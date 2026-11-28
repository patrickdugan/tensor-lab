"""Frozen, CPU-only HAD-QFT-QPU04 design on two saved Kingston regions."""
from pathlib import Path
import hashlib,json,time
from selective import core,audit,bound,count
from route_dp import Router,verify

HERE=Path(__file__).resolve().parent
REGIONS=((71,72,73,58,70,74,79),(123,136,143,122,124,142,144))
EDGES=((0,1),(1,2),(0,3),(0,4),(2,5),(2,6))
ORDERS={6:[5,6,4,1,3,2]}
PARENT='9547024409ef7594cfea5230e26125d176dbf8db'


def route_hash(g):return hashlib.sha256(json.dumps(g,separators=(',',':')).encode()).hexdigest()


def candidates():
    # Both arms receive the same routing grammar and four control-order choices.
    for style in ('direct','commutator'):
        for order in ('rev','forward'):
            for modified in (False,True):
                yield style,order,modified


def select():
    r=Router(7,EDGES);rows=[];best={};t=time.monotonic()
    for style,order,modified in candidates():
        g=core(7,style,order,level_orders=ORDERS if modified else None)
        z=r.route(g);v=audit(g,7,bound(7));error=verify(g,z,7)
        row=dict(style=style,order=order,modified=modified,**v,routed_cx=count(z['gates']),dp_cx=z['cost'],route_error=error,initial=z['initial'],final=z['final'],swaps=z['swaps'],fusions=z['fusions'],local_fusions=z['local_fusions'])
        rows.append(row)
        key=(row['routed_cx'],row['dp_cx'],len(z['gates']),order,modified)
        if style not in best or key<best[style][0]:best[style]=(key,row)
    return dict(scope='CPU_ONLY',parent=PARENT,regions=[list(x) for x in REGIONS],edges=[list(x) for x in EDGES],candidates=rows,best={s:v[1] for s,v in best.items()},seconds=time.monotonic()-t)


def frozen_design(region,report):
    assert 0<=region<len(REGIONS)
    arms=[]
    for name,style in (('A','direct'),('D','commutator')):
        x=report['best'][style]
        arms.append(dict(arm=name,style=style,order=x['order'],modified=x['modified'],phys=list(REGIONS[region]),routed_cx=x['routed_cx'],dp_cx=x['dp_cx'],logical_cx=x['cx'],initial=x['initial'],final=x['final'],m=0))
    return dict(scope='FROZEN_QPU04_PHASE_CONTRAST',stage='b7',b=7,replicate=region,backend='ibm_kingston',shots=1024,parent=PARENT,edges=[list(x) for x in EDGES],arms=arms,families=dict(signal=['A','D'],difference=[['D','A']]),primary='edge contrast D-A, matched physical region; shot-only Hoeffding',note='Same node set, independently optimized input/output permutation per compiler. Two disjoint regions. Secondary probes exploratory.')


def regenerate(design):
    assert design['b']==7 and design['shots']==1024 and design['parent']==PARENT
    assert design['edges']==[list(x) for x in EDGES] and design['replicate'] in (0,1)
    assert [x['arm'] for x in design['arms']]==['A','D']
    assert design['families']==dict(signal=['A','D'],difference=[['D','A']])
    r=Router(7,EDGES);out={}
    for arm in design['arms']:
        assert arm['phys']==list(REGIONS[design['replicate']]) and arm['m']==0
        assert arm['style']==('direct' if arm['arm']=='A' else 'commutator')
        g=core(7,arm['style'],arm['order'],level_orders=ORDERS if arm['modified'] else None)
        z=r.route(g);v=audit(g,7,bound(7));error=verify(g,z,7)
        assert count(g)==arm['logical_cx'] and count(z['gates'])==arm['routed_cx'] and z['cost']==arm['dp_cx']
        assert z['initial']==arm['initial'] and z['final']==arm['final']
        out[arm['arm']]=(g,z,{**v,'max_routing_error':error})
    return out

if __name__=='__main__':
    report=select();print(json.dumps(report,indent=2),flush=True)
    (HERE/'QPU04_ROUTING_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    for i in range(len(REGIONS)):
        d=frozen_design(i,report);regenerate(d);(HERE/f'qpu04_b7_r{i}.json').write_text(json.dumps(d,indent=2)+'\n')
