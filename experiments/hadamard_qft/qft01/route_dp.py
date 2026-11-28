"""Exact layout DP within a restricted routing grammar.
Free initial and final permutations; all-pairs shortest-path bridges, adjacent
SWAPs, and fused CX followed by SWAP. Gate order fixed, no extra wires.
Not a lower bound on arbitrary hardware synthesis. Cost is number of CNOTs.
"""
from itertools import permutations
from collections import deque
from functools import lru_cache
import numpy as np
from recursive import simulate
from compact import simplify


class Router:
    def __init__(self,n,edges,local3=True):
        self.n=n; self.edges=tuple(sorted({tuple(sorted(e)) for e in edges})); assert all(0<=a<b<n for a,b in self.edges)
        assert n<=8
        self.perms=np.array(list(permutations(range(n))),dtype=np.int16) # logical -> physical
        self.pidx={tuple(p):i for i,p in enumerate(self.perms)}
        self.states=len(self.perms); self.ids=np.arange(self.states,dtype=np.int32)
        adj=[[] for _ in range(n)]
        for a,b in self.edges:adj[a].append(b);adj[b].append(a)
        self.paths={}
        for a in range(n):
            paths={a:[a]}; q=deque([a])
            while q:
                b=q.popleft()
                for c in adj[b]:
                    if c not in paths:paths[c]=paths[b]+[c];q.append(c)
            assert len(paths)==n,'disconnected graph'
            self.paths.update({(a,b):p for b,p in paths.items()})
        self.dist=np.array([[len(self.paths[a,b])-1 for b in range(n)] for a in range(n)],dtype=np.int16)
        # edge SWAP is its own inverse, hence transition table symmetric.
        self.sw=[];self.swap_by_pair={}
        for a,b in self.edges:
            p=self.perms.copy();p[self.perms==a]=b;p[self.perms==b]=a
            j=np.array([self.pidx[tuple(row)] for row in p],dtype=np.int32)
            self.sw.append(j);self.swap_by_pair[a,b]=j
        self.sw=np.stack(self.sw)
        self.bridge={}
        for a in range(n):
            for b in range(n):
                if a==b:continue
                p=self.paths[a,b];e=[('cx',x,y) for x,y in zip(p,p[1:])]
                self.bridge[a,b]=e if len(e)==1 else e+e[-2::-1]+e[1:]+e[-2:0:-1]
        self._trans={};self.local3=local3;self.templates=[]
        if local3:
            # Enumerate GL(3,2) by BFS in adjacent directed CNOT generators.
            # Each word implements a linear reversible map, no relative phases.
            I=(1,2,4); words={I:[]}; queue=deque([I]);gens=((0,1),(1,0),(1,2),(2,1))
            while queue:
                u=queue.popleft()
                for a,b in gens:
                    v=list(u);v[b]^=v[a];v=tuple(v)
                    if v not in words:words[v]=words[u]+[(a,b)];queue.append(v)
            assert len(words)==168
            for mid in range(n):
                for a in adj[mid]:
                    for b in adj[mid]:
                        if a>=b:continue
                        path=(a,mid,b)
                        for p in permutations(range(3)):
                            if p==(0,1,2):continue
                            # after layout -> before layout for the candidate
                            # output permutation p, moving every token in path.
                            inv=[p.index(i) for i in range(3)]
                            q=self.perms.copy()
                            for j in range(3):q[self.perms==path[j]]=path[inv[j]]
                            trans=np.array([self.pidx[tuple(row)] for row in q],dtype=np.int32)
                            costs=[];gates=[]
                            for x,y in ((0,2),(2,0),(0,1),(1,0),(1,2),(2,1)):
                                u=list(I);u[y]^=u[x];v=[0]*3
                                for i,j in enumerate(p):v[j]=u[i]
                                word=words[tuple(v)]
                                costs.append(len(word));gates.append([('cx',path[i],path[j]) for i,j in word])
                            self.templates.append((path,p,trans,costs,gates))

    def _transitions(self,a,b):
        if (a,b) not in self._trans:
            aa=self.perms[:,a];bb=self.perms[:,b];d=self.dist[aa,bb]
            price=np.where(d==1,1,4*d-4).astype(np.int32)
            # Fusion: adjacent CX followed by SWAP -> reverse CX then CX.
            j=self.ids.copy();valid=d==1
            for x,y in self.edges:
                mask=valid & (((aa==x)&(bb==y))|((aa==y)&(bb==x)))
                j[mask]=self.swap_by_pair[x,y][mask]
            extras=[]
            for ti,(path,p,trans,costs,_) in enumerate(self.templates):
                # Candidate templates include adjacent and endpoint CNOTs.
                pre=self.perms[trans];aa=pre[:,a];bb=pre[:,b]
                for k,(x,y) in enumerate(((path[0],path[2]),(path[2],path[0]),(path[0],path[1]),(path[1],path[0]),(path[1],path[2]),(path[2],path[1]))):
                    after=self.ids[(aa==x)&(bb==y)]
                    extras.append((after,trans[after],costs[k],2+6*ti+k))
            self._trans[a,b]=(price,valid,j,extras)
        return self._trans[a,b]

    def route(self,gates,materialize=True):
        cost=np.zeros(self.states,dtype=np.int32);history=[]
        for g in gates:
            if g[0]!='cx':continue
            _,a,b=g
            # Multi-source shortest path via synchronous Bellman relaxation.
            v=cost.copy();prev=self.ids.copy()
            while True:
                new=v.copy();pred=prev.copy()
                for j in self.sw:
                    cand=v[j]+3;hit=cand<new;new[hit]=cand[hit];pred[hit]=j[hit]
                if np.array_equal(v,new):break
                v,prev=new,pred
            price,valid,j,extras=self._transitions(a,b)
            after=v+price;pre=self.ids.copy();fused=np.zeros(self.states,dtype=np.int16)
            cand=v[j]+2;hit=valid&(cand<after);after[hit]=cand[hit];pre[hit]=j[hit];fused[hit]=1
            for dst,src,c,tag in extras:
                cand=v[src]+c;hit=cand<after[dst];ix=dst[hit]
                after[ix]=cand[hit];pre[ix]=src[hit];fused[ix]=tag
            if materialize:history.append((prev,pre,fused))
            cost=after
        end=int(np.argmin(cost));opt=int(cost[end])
        if not materialize:return opt
        last=end;rev=[]
        for prev,pre,fused in reversed(history):
            start=int(pre[last]);j=start;sw=[]
            while prev[j]!=j:
                k=int(prev[j]);diff=np.flatnonzero(self.perms[k]!=self.perms[j]);assert len(diff)==2
                sw.append(tuple(int(self.perms[k,i]) for i in diff));j=k
            rev.append((sw[::-1],int(fused[last]),start,last));last=j
        initial=list(map(int,self.perms[last]));pos=initial.copy();idx=0;out=[];swaps=0;fusions=0;local_fusions=0
        def swap(a,b):
            ia,ib=pos.index(a),pos.index(b);pos[ia],pos[ib]=b,a
        for g in gates:
            if g[0]!='cx':out.append((g[0],pos[g[1]],*g[2:]));continue
            sw,fu,pre,post=rev[-1-idx];idx+=1
            for a,b in sw:
                out += [('cx',a,b),('cx',b,a),('cx',a,b)];swap(a,b);swaps+=1
            assert pos==list(self.perms[pre]);a,b=pos[g[1]],pos[g[2]]
            if fu==1:
                out += [('cx',b,a),('cx',a,b)];swap(a,b);fusions+=1
            elif fu>=2:
                path,p,_,_,words=self.templates[(fu-2)//6];out+=words[(fu-2)%6]
                pos=[path[p[path.index(x)]] if x in path else x for x in pos];local_fusions+=1
            else:out += self.bridge[a,b]
            assert pos==list(self.perms[post])
        assert sum(g[0]=='cx' for g in out)==opt
        assert all(g[0]!='cx' or tuple(sorted(g[1:])) in self.edges for g in out)
        return dict(cost=opt,gates=simplify(out),initial=initial,final=pos,swaps=swaps,fusions=fusions,local_fusions=local_fusions)


def index(n,pos):
    i=np.arange(1<<n,dtype=np.int64);out=np.zeros_like(i)
    for l,p in enumerate(pos):out|=((i>>(n-1-l))&1)<<(n-1-p)
    return out


def verify(g,route,n):
    d=1<<n;I=np.eye(d,dtype=complex);ix=index(n,route['initial']);ox=index(n,route['final']);
    a=simulate(g,n,I);b=simulate(route['gates'],n,I[:,ix])[ox,:]
    err=float(np.max(abs(a-b)));assert err<1e-10,err
    return err
