import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import argparse
from scripts.train import train
from utils.explainable_utils.gradcam import gradcam_image_gen
from utils.explainable_utils.shap import shap_explainer
from utils.data.confusionMatrixGenerator import confusion_matrix_plotter

def evaluate(load_trained, pretrained_model_path=None):
    """Evaluation of model

    Args:
        pretrained_model_path (str): Path of pretrained model
        load_trained (bool): Load pretrained model or not
    """
    model, train_iterator, valid_iterator, test_iterator, test_gradcam_iterator = train(load_trained, pretrained_model_path)
    confusion_matrix_plotter(model, test_iterator)
    gradcam_image_gen(test_gradcam_iterator, image_num = 7) # image number must be less than size of test loader
    shap_explainer(model, test_iterator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pretrained_model_path', type=str, required=False)
    args = parser.parse_args()
    if args.pretrained_model_path is not None:
        load_trained = True
        evaluate(load_trained, args.pretrained_model_path)
    else:
        load_trained = False
        evaluate(load_trained, args.pretrained_model_path)