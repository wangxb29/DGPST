from options import TestOptions
import data
import models
from evaluation import GroupEvaluator


opt = TestOptions().parse()
model = models.create_model(opt)
dataset = data.create_dataset(opt)
evaluators = GroupEvaluator(opt)
evaluators.evaluate(model, dataset, opt.resume_iter)
