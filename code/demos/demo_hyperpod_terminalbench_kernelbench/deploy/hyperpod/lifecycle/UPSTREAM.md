# Vendored HyperPod Slurm lifecycle scripts

Copied unmodified from aws-samples/awsome-distributed-training (MIT-0, `LICENSE.upstream`),
`1.architectures/5.sagemaker-hyperpod/LifecycleScripts/base-config/` at commit
`7bca2cf40e5cd886b3342e8e0598b59f7b176604` (2026-10-12).

`hyperpod.py up` uploads this folder to `s3://<bucket>/lifecycle/<cluster>/` and adds the generated
`provisioning_parameters.json` (controller group, the `gpu` worker group in partition `rl`, the FSx DNS
name + mount name). `config.py` defaults are kept: Docker + enroot + pyxis on, observability off.
