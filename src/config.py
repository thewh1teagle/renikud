from pathlib import Path
from tap import Tap
from typing import Literal

BASE_PATH = Path(__file__) / '../../'

class TrainArgs(Tap):
    model: str = "dicta-il/dictabert-large-char-menaked" # thewh1teagle/renikud

    device: Literal["cuda", "cuda:1", "cpu", "mps"] = "cuda:1"

    data_dir: Path = BASE_PATH / "data/"

    output_dir: Path = BASE_PATH / "ckpt/"

    log_dir: Path = BASE_PATH / "logs/"

    batch_size: int = 32

    epochs: int = 20

    learning_rate: float = 5e-3

    early_stopping_patience: int = 3

    num_workers: int = 16

    checkpoint_interval: int = 9000

    val_split_num: float = 250 # 250 lines for validation

    split_seed: int = 0

    wandb_entity: str = "Renikud"

    wandb_project: str = "renikud"
    
    wandb_mode: str = "offline"


def get_args():
    return TrainArgs().parse_args()