import json, os, numpy, random, torch


'''[Feb-17-2022] https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/trainer_utils.py#L50'''
def set_seed(seed=52):
    """Fix the random seed for reproducibility"""
    if seed < 0: return
    # os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cuda.matmul.allow_tf32 = False


'''[20211231]'''
class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJsonEncoder, self).default(obj)

'''[20210818]'''
def make_sure_parent_dir_exists(filepath):
    parent_dir = os.path.dirname(filepath)
    os.makedirs(parent_dir, exist_ok=True)


'''[20210818]'''
def write_list_to_json_file(data, filepath):
    make_sure_parent_dir_exists(filepath)
    with open(filepath, "w") as f:
        for i in data:
            f.write(f"{json.dumps(i, cls=NumpyJsonEncoder)}\n")


'''[20210810]'''
def write_object_to_json_file(data, filepath, sort_keys=False):
    make_sure_parent_dir_exists(filepath)
    json.dump(data, open(filepath, "w"), indent=2, sort_keys=sort_keys, default=lambda o: "Unknown")