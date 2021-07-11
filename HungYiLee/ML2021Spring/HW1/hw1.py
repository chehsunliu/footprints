import logging
from typing import List, NamedTuple, Callable, Dict, Any, Tuple, Union

import torch
import pandas
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from torch import nn
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(format="[%(asctime)s] %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class HW1Config(NamedTuple):
    batch_size: int
    device: str
    epochs: int
    model_saving_path: str
    optimizer: Callable
    optimizer_params: Dict[str, Any]


available_features = [
    "id",

    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "FL", "GA", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "MD", "MA", "MI", "MN",
    "MS", "MO", "NE", "NV", "NJ", "NM", "NY", "NC", "OH", "OK",
    "OR", "PA", "RI", "SC", "TX", "UT", "VA", "WA", "WV", "WI",

    "cli", "ili", "hh_cmnty_cli", "nohh_cmnty_cli", "wearing_mask", "travel_outside_state", "work_outside_home", "shop",
    "restaurant", "spent_time",
    "large_event", "public_transit", "anxious", "depressed", "felt_isolated", "worried_become_ill", "worried_finances",
    "tested_positive",

    "cli.1", "ili.1", "hh_cmnty_cli.1", "nohh_cmnty_cli.1", "wearing_mask.1", "travel_outside_state.1",
    "work_outside_home.1", "shop.1", "restaurant.1", "spent_time.1",
    "large_event.1", "public_transit.1", "anxious.1", "depressed.1", "felt_isolated.1", "worried_become_ill.1",
    "worried_finances.1", "tested_positive.1",

    "cli.2", "ili.2", "hh_cmnty_cli.2", "nohh_cmnty_cli.2", "wearing_mask.2", "travel_outside_state.2",
    "work_outside_home.2", "shop.2", "restaurant.2", "spent_time.2",
    "large_event.2", "public_transit.2", "anxious.2", "depressed.2", "felt_isolated.2", "worried_become_ill.2",
    "worried_finances.2",

    "tested_positive.2"  # label, only available in training data
]

features = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "FL", "GA", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "MD", "MA", "MI", "MN",
    "MS", "MO", "NE", "NV", "NJ", "NM", "NY", "NC", "OH", "OK",
    "OR", "PA", "RI", "SC", "TX", "UT", "VA", "WA", "WV", "WI",

    # "cli",
    # "ili",
    # "hh_cmnty_cli",
    # "nohh_cmnty_cli",
    # "wearing_mask",
    # "travel_outside_state",
    # "work_outside_home",
    # "shop",
    # "restaurant",
    # "spent_time",
    # "large_event",
    # "public_transit",
    # "anxious",
    # "depressed",
    # "felt_isolated",
    # "worried_become_ill",
    # "worried_finances",
    "tested_positive",

    # "cli.1",
    # "ili.1",
    # "hh_cmnty_cli.1",
    # "nohh_cmnty_cli.1",
    # "wearing_mask.1",
    # "travel_outside_state.1",
    # "work_outside_home.1",
    # "shop.1",
    # "restaurant.1",
    # "spent_time.1",
    # "large_event.1",
    # "public_transit.1",
    # "anxious.1",
    # "depressed.1",
    # "felt_isolated.1",
    # "worried_become_ill.1",
    # "worried_finances.1",
    "tested_positive.1",

    # "cli.2",
    # "ili.2",
    # "hh_cmnty_cli.2",
    # "nohh_cmnty_cli.2",
    # "wearing_mask.2",
    # "travel_outside_state.2",
    # "work_outside_home.2",
    # "shop.2",
    # "restaurant.2",
    # "spent_time.2",
    # "large_event.2",
    # "public_transit.2",
    # "anxious.2",
    # "depressed.2",
    # "felt_isolated.2",
    # "worried_become_ill.2",
    # "worried_finances.2",
]


class Covid19Dataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

        self.data_mean = data.mean(dim=0, keepdim=True)
        self.data_std = data.std(dim=0, keepdim=True)

        logger.info("data: %s", self.data.shape)
        if self.labels is not None:
            logger.info("data: %s", self.labels.shape)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

    def normalize(self) -> None:
        self.data[:, 40:] = (self.data[:, 40:] - self.data_mean[:, 40:]) / self.data_std[:, 40:]

    def normalize_by(self, other: 'Covid19Dataset') -> None:
        self.data[:, 40:] = (self.data[:, 40:] - other.data_mean[:, 40:]) / other.data_std[:, 40:]

    @property
    def dimension(self) -> int:
        return self.data.shape[1]

    @classmethod
    def create_dataset(cls, filepath: str,
                       is_training: bool) -> 'Union[Covid19Dataset, Tuple[Covid19Dataset, Covid19Dataset]]':
        logger.info("Reading data from %s", filepath)
        original_data = pandas.read_csv(filepath)

        data = torch.FloatTensor(original_data[features].values)

        if not is_training:
            return cls(data=data)

        labels = torch.FloatTensor(original_data["tested_positive.2"].values)

        total_rows = len(data)
        indexes1 = [i for i in range(total_rows) if i % 10 == 0]
        indexes0 = [i for i in range(total_rows) if i % 10 != 0]

        return cls(data=data[indexes0], labels=labels[indexes0]), cls(data=data[indexes1], labels=labels[indexes1])

    @classmethod
    def create_training_dataset(cls, filepath: str) -> 'Tuple[Covid19Dataset, Covid19Dataset]':
        return cls.create_dataset(filepath, is_training=True)

    @classmethod
    def create_testing_dataset(cls, filepath: str) -> 'Covid19Dataset':
        return cls.create_dataset(filepath, is_training=False)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dimension, 42),
            nn.ReLU(),
            nn.Linear(42, 1),
        )

        self.loss_function = nn.MSELoss(reduction="mean")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X).squeeze(1)

    def calculate_loss(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        return self.loss_function(y1, y2)


class LossRecords(NamedTuple):
    training_loss: List[float] = []
    validation_loss: List[float] = []


def train_model(training_dataloader: DataLoader, validation_dataloader: DataLoader, model: NeuralNetwork,
                config: HW1Config) -> LossRecords:
    optimizer = config.optimizer(model.parameters(), **config.optimizer_params)
    minimal_validation_loss = 1e10
    loss_records = LossRecords()

    for epoch in range(config.epochs):
        model.train()

        for X, y in training_dataloader:
            X, y = X.to(config.device), y.to(config.device)

            optimizer.zero_grad()
            prediction = model(X)
            loss = model.calculate_loss(prediction, y)
            loss.backward()
            optimizer.step()

            loss_records.training_loss.append(loss.detach().cpu().item())

        model.eval()
        validation_loss = validate_model(validation_dataloader, model, config=config)
        loss_records.validation_loss.append(validation_loss)

        if validation_loss < minimal_validation_loss:
            minimal_validation_loss = validation_loss
            torch.save(model.state_dict(), config.model_saving_path)

        if epoch % (config.epochs // 10) == 0 or epoch == config.epochs - 1:
            logger.info(f">> Finish (epoch: {epoch + 1:4d}, min_loss: {minimal_validation_loss:.4f})")

    return loss_records


def validate_model(dataloader: DataLoader, model: NeuralNetwork, config: HW1Config):
    total_loss = 0

    for X, y in dataloader:
        X, y = X.to(config.device), y.to(config.device)
        with torch.no_grad():
            prediction = model(X)
            loss = model.calculate_loss(prediction, y)
            total_loss += loss.detach().cpu().item() * len(X)

    return total_loss / len(dataloader.dataset)


def test_model(dataloader: DataLoader, model: NeuralNetwork, config: HW1Config) -> torch.Tensor:
    predictions = []

    for X in dataloader:
        X = X.to(config.device)
        with torch.no_grad():
            predictions.append(model(X).detach().cpu())

    return torch.cat(predictions, dim=0)


def plot_learning_curve(loss_records: LossRecords) -> None:
    total_steps = len(loss_records.training_loss)
    x1 = range(total_steps)
    x2 = x1[::len(loss_records.training_loss) // len(loss_records.validation_loss)]

    figure(figsize=(6, 4))
    plt.plot(x1, loss_records.training_loss, c="tab:red", label="training")
    plt.plot(x2, loss_records.validation_loss, c="tab:cyan", label="validation")

    plt.ylim(0.0, 5.0)
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.title("Learning curve")
    plt.legend()
    plt.show()
