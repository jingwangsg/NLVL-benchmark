import copy
from evaluate import EvaluaterNLVLEval, EvaluaterScaler
from kn_util.config import LazyCall as L


def build_evaluater_nlvl(losses):
    train_evaluater = L(EvaluaterScaler)(scalers=losses, namespace="train")
    eval_evaluater = L(EvaluaterNLVLEval)(ms=[1, 5], ns=[0.7, 0.5, 0.3], namespace="eval")
    test_evaluater = copy.deepcopy(eval_evaluater)
    test_evaluater.namespace = "test"

    return train_evaluater, eval_evaluater, test_evaluater
