import os.path as osp

runtime = dict(prefetch_factor=2, num_workers=16, find_unused_parameters=False)

NLVL_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
paths = dict(root=osp.join(NLVL_DIR, "benchmark"),
             data=osp.join(NLVL_DIR, "data-bin"),
             work_dir=osp.join(NLVL_DIR, "benchmark", "work_dir", "${args.exp}"),
             dataset=osp.join(NLVL_DIR, "data-bin", "${data.dataset}"))
