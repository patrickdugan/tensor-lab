import json,sys
a,b=(json.load(open(p)) for p in sys.argv[1:])
def close(x,y,tol=1e-12):
    if isinstance(x,dict): return x.keys()==y.keys() and all(close(x[k],y[k],tol) for k in x)
    if isinstance(x,list): return len(x)==len(y) and all(close(u,v,tol) for u,v in zip(x,y))
    if isinstance(x,float) or isinstance(y,float): return abs(x-y)<=tol*max(1.0,abs(x),abs(y))
    return x==y
# Fitted offsets, evaluation counts and bounds must match exactly; floating mean-square errors
# differ in the last digits across Node/V8 versions, so those use a 1e-12 relative tolerance.
assert close(a['fit'],b['fit'])
assert len(a['rows'])==len(b['rows'])
for x,y in zip(a['rows'],b['rows']):
    assert all(x[k]==y[k] for k in ('id','bit','physical','measured'))
    assert all(abs(x[k]-y[k])<1e-9 for k in ('ideal','twirled','nonunital','twirled_sx','nonunital_sx'))
print('QC25C receipt replay passed',len(a['rows']))
