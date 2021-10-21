"""Getting params from the command line. 
https://github.com/benedekrozemberczki/SimGNN
"""
import argparse
import keras


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run SimGNN.")

    parser.add_argument("--training-graphs",
                        nargs="?",
                        default="C:/Users/91876/Downloads/SimGNN-main/dataset/A01/Training/A01T_3/",
	                help="Folder with training graph pair jsons.")

    parser.add_argument("--testing-graphs",
                        nargs="?",
                        default="C:/Users/91876/Downloads/SimGNN-main/dataset/A01/Testing/A01E_3/",
	                help="Folder with testing graph pair jsons.")
            

    parser.add_argument("--epochs",
                        type=int,
                        default=50,  #100
	                help="Number of training epochs. Default is 5.")

    """
    Will add the units into main model. Check (gcn1)model built using functional at main.py
    """
    parser.add_argument("--filters-1",
                        type=int,
                        default=64,
	                help="Filters (neurons) in 1st convolution. Default is 128.")

    """
    Will add the units into main model. Check (gcn2)model built using functional at main.py
    """
    parser.add_argument("--filters-2",
                        type=int,
                        default=32,
	                help="Filters (neurons) in 2nd convolution. Default is 64.")

    """
    Will add the units into main model. Check (gcn3)model built using functional at main.py
    """
    parser.add_argument("--filters-3",
                        type=int,
                        default=16,
	                help="Filters (neurons) in 3rd convolution. Default is 32.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--saveafter",
                        type=int,
                        default=30,
	                help="Saves model after every argument epochs")

    return parser.parse_args()
