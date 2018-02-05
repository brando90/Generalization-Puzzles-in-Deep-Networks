import matplotlib as mpl
import matplotlib.pyplot as plt

import pdb
from pdb import set_trace as st

def plot_loss_errors(iterations,stats_collector):
    fig=plt.figure()
    train_line, = plt.plot(iterations,stats_collector.train_losses,label='Train Loss')
    # plt.plot(iterations,stats_collector.val_losses)
    test_line, = plt.plot(iterations,stats_collector.test_losses,label='Test Loss')
    plt.legend(handles=[train_line,test_line])
