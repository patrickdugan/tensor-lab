import json,sys
a,b=(json.load(open(p)) for p in sys.argv[1:])
assert a['fit']==b['fit']
assert len(a['rows'])==len(b['rows'])
for x,y in zip(a['rows'],b['rows']):
    assert all(x[k]==y[k] for k in ('id','bit','physical','measured'))
    assert all(abs(x[k]-y[k])<1e-9 for k in ('ideal','twirled','nonunital','twirled_sx','nonunital_sx'))
print('QC25C receipt replay passed',len(a['rows']))
