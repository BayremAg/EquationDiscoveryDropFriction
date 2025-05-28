import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import ticker
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import manifold
from torch.utils.data import DataLoader, TensorDataset

from definitions import ROOT_DIR
from src.equation_discovery.fit_constant import fit_constants
from src.preprocess_data.config_load_dataset import ConfigLoadData
from src.preprocess_data.preprocess_data import load_Sajjad
from src.SyntaxTree.src.syntax_tree.config_syntax_tree import ConfigSyntaxTree
from src.analyse_equations.config_analyse_equations import ConfigPlotBestEquation
from src.equation_discovery.config_equations_for_each_dataset import ConfigEquationDiscovery
from src.utils.config_hyperparameter import ConfigHyperparameter


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.activation_1 = torch.nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation_2 = torch.nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        # Initialize weights to identity matrix
        #nn.init.eye_(self.fc1.weight)

        # Initialize biases to zero
        #nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation_1(out)
        out = self.fc2(out)
        out = self.activation_2(out)
        out = self.fc3(out)
        return out.squeeze()

def train_NN(x, y):
    # Hyperparameters
    hidden_size = 6  # Example hidden layer size
    output_size = 1  # Example output size
    learning_rate = 0.1
    batch_size = 16
    num_epochs = 10000
    x = torch.tensor(x)
    y = torch.tensor(y[:, 0])

    dataset = TensorDataset(x.to(torch.float32), y.to(torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(x.shape[1], hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4e}   {model.fc1.weight}')
    print("Training complete!")
    return model


def add_tsne_plot(args,fig, gs,D_train):
    x, y =  D_train.loc[:,  ['drop_length', 'adv', 'rec','avg_vel', 'width'] ].to_numpy(), D_train.loc[:, ['y']].to_numpy()
    perplexities = [3, 10, 20, 30]
    for i, perplexity in enumerate(perplexities):
        ax = fig.add_subplot(gs[1, i])
        if i == 3:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

        tsne = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=0,
            perplexity=perplexity,
            max_iter=300,
        )
        Y = tsne.fit_transform(x)

        print("circles, perplexity=%d in sec" % (perplexity))
        ax.set_title("Perplexity=%d" % perplexity)

        sc = ax.scatter(Y[:, 0], Y[:, 1], c=y)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis("tight")
        if i == 0:
            ax.set_ylabel('TSNE of Dataset')
        if i == 3:
            cbar = fig.colorbar(sc, cax=cax, orientation='vertical', format=ticker.FuncFormatter(fmt))
            cbar.set_label('Friction Force [mF]', rotation=0, labelpad=-30, y=1.05)


def add_NN_prediction(args, fig, gs, dataset_name, D_train, model):
    x, y =  D_train.loc[:,  ['drop_length', 'adv', 'rec','avg_vel', 'width'] ].to_numpy(), D_train.loc[:, ['y']].to_numpy()
    predict = model(torch.tensor(x).to(torch.float32)).detach().numpy()
    ax = fig.add_subplot(gs[0, 0:4])
    ax.set_title(dataset_name)
    ax.plot(range(len(predict)), D_train.loc[:, ['y']].to_numpy() ,
            label='true',  # s=1,
            marker='o', linestyle='-')
    ax.plot(range(len(predict)), predict, label='prediction NN',  marker='p',  linestyle='-')
    # ax.bar(range(len(predict)), (D_train.loc[:,['y']].to_numpy() - predict)[:,0], label='difference' )
    ax.set_ylabel('Friction Force [mN]')
    return ax



def add_equation_prediction(ax, equation, D_train):
    parser = ConfigHyperparameter.arguments_parser()
    parser = ConfigLoadData.arguments_parser(parser)
    parser = ConfigEquationDiscovery.arguments_parser(parser)
    parser = ConfigPlotBestEquation.arguments_parser(parser)
    parser = ConfigSyntaxTree.arguments_parser(parser)
    args = parser.parse_args()
    tree = fit_constants(args, equation, D_train)
    y_pred_train = tree.evaluate_subtree(-1, D_train)
    ax.plot(range(len(y_pred_train)), y_pred_train, label=equation,
    marker = 'v', linestyle='--'
    )

def run():
    parser = ConfigLoadData.arguments_parser()
    args = parser.parse_args()


    #D_test =  load_Sajjad(args,f'{ROOT_DIR}/data/Sajjad/full_dataset500_RoboSci_V3_no-defect_97.csv')
    D_train = load_Sajjad(args, f'{ROOT_DIR}/data/Sajjad/full_dataset500_RoboSci_V3_no-defect_97.csv')
    D_train.loc[:, ['y']] = D_train.loc[:, ['y']] *1000
    D_train.loc[:, ['drop_length']] = D_train.loc[:, ['drop_length']] * 100
    D_train.loc[:, ['y_center']] = D_train.loc[:, ['y_center']] * 100
    D_train.loc[:, ['width']] = D_train.loc[:, ['width']] * 100

    x, y = D_train.loc[:,  ['drop_length', 'adv', 'rec','avg_vel', 'width'] ].to_numpy(), D_train.loc[:,['y']].to_numpy()
    #x, y = D_train.loc[:, ['y']].to_numpy(), D_train.loc[:, ['y']].to_numpy()
    #X_test, y_test = D_test.loc[:,  ['y']].to_numpy(), D_test.loc[:,['y']].to_numpy()
    dataset_name = (f"{int(D_train.loc[0, 'Video ID'])} "
              f"{round(np.rad2deg(D_train.loc[0, 'tilt_angle']))}° "
              f"{D_train.loc[0, 'excel_name']}")

    fig = plt.figure( figsize=(15, 8))
    gs = fig.add_gridspec(2,4)

    model = train_NN(x,y)
    ax = add_NN_prediction(args, fig, gs, dataset_name, D_train, model)
    add_equation_prediction(ax, equation = ' * c sin / rec / 1 mid', D_train=D_train)
    add_equation_prediction(ax, equation=' * c * width - cos rec  cos adv ', D_train=D_train)
    ax.legend(loc='upper right',ncol=4)
    add_tsne_plot(args, fig, gs,D_train)
    save_string = ROOT_DIR / f"plots/one_dataset/{dataset_name}"
    save_string.parent.mkdir(exist_ok=True, parents= True)
    fig.savefig(save_string)
    plt.show()

if __name__ == '__main__':
    run()

