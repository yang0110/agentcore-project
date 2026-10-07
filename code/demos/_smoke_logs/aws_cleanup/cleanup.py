import boto3, time, sys
R='us-west-2'; c=boto3.client('bedrock-agentcore-control',region_name=R); e=boto3.client('ecr',region_name=R)
D='/home/ubuntu/workspace/agentcore-project/code/demos/_smoke_logs/aws_cleanup/'
want=set(open(D+'del_rt.txt').read().split())
rts=[];tok=None
while True:
    kw={'maxResults':100}; kw.update({'nextToken':tok} if tok else {})
    r=c.list_agent_runtimes(**kw); rts+=r['agentRuntimes']; tok=r.get('nextToken')
    if not tok: break
for r in rts:
    if r['agentRuntimeName'] in want:
        try: c.delete_agent_runtime(agentRuntimeId=r['agentRuntimeId']); print('del runtime',r['agentRuntimeName'],flush=True)
        except Exception as x: print('FAIL runtime',r['agentRuntimeName'],x,flush=True)
# wait until gone, then capacity providers
for i in range(60):
    left=[r['agentRuntimeName'] for r in c.list_agent_runtimes(maxResults=100)['agentRuntimes']]
    tok=None; names=[]
    while True:
        kw={'maxResults':100}; kw.update({'nextToken':tok} if tok else {})
        r=c.list_agent_runtimes(**kw); names+=[x['agentRuntimeName'] for x in r['agentRuntimes']]; tok=r.get('nextToken')
        if not tok: break
    if not want & set(names): break
    time.sleep(20)
print('runtimes remaining from list:',sorted(want&set(names)),flush=True)
for cp in c.list_capacity_providers().get('capacityProviders',[]):
    if cp['name'] in ('acs_cpu_cp','acs_gpu_cp','acs_swe_cp'):
        for k in range(20):
            try: c.delete_capacity_provider(capacityProviderId=cp['capacityProviderId']); print('del cp',cp['name'],flush=True); break
            except Exception as x: print('retry cp',cp['name'],str(x)[:120],flush=True); time.sleep(30)
for n in open(D+'del_ecr.txt').read().split():
    try: e.delete_repository(repositoryName=n,force=True); print('del ecr',n,flush=True)
    except Exception as x: print('FAIL ecr',n,x,flush=True)
print('DONE',flush=True)
