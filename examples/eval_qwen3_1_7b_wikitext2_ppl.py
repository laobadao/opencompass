from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.wikitext.wikitext_2_raw_ppl_752e2a import wikitext_2_raw_datasets
    from opencompass.configs.models.qwen3.hf_qwen3_1_7b import models as qwen3_1_7b

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

datasets = [*wikitext_2_raw_datasets]
workdir = 'outputs/qwen3_1_7b_wikitext2_ppl'

models = [*qwen3_1_7b]

model_cfg = dict(batch_size=4, run_cfg=dict(num_gpus=1, num_procs=1))

for mdl in models:
    mdl.update(model_cfg)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
        max_num_workers=16,
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask),
        max_num_workers=16,
    ),
)
