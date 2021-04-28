import sys

from train_od_ex import TrainingPipeline

if __name__ == '__main__':
    t = TrainingPipeline(export=True)
    model = t.get_model_from_checkpoint(sys.argv[1])
    # Make experement here
    # Watch test method
    pass