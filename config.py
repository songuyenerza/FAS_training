from dotenv import load_dotenv
import os

# env = load_dotenv()
env = load_dotenv('env.example')

def str2bool(value):
    return value.lower() in ("true", "1", "t", "y", "yes")

class ApplicationConfig:
    # Đọc các biến từ file .env vào trong config
    url_data = os.environ.get('URL_DATA')
    nproc_per_node = int(os.environ.get('NPROC_PER_NODE'))

    root = os.environ.get('ROOT')
    train_list = os.environ.get('TRAIN_LIST')
    val_list = os.environ.get('VAL_LIST')
    
    cos = True
    syncbn = True
    arch = os.environ.get('ARCH')

    single_center_loss = str2bool(os.environ.get('SINGLE_CENTER_LOSS'))
    single_center_loss_weight = float(os.environ.get('SINGLE_CENTER_LOSS_WEIGHT'))

    num_classes = int(os.environ.get('NUM_CLASS'))
    input_size = int(os.environ.get('INPUT_SIZE'))
    batch_size = int(os.environ.get('BATCH_SIZE'))
    workers = int(os.environ.get('WORKERS'))

    optimizer = os.environ.get('OPTIMIZER')
    learning_rate = float(os.environ.get('LEARNING_RATE'))
    weight_decay = float(os.environ.get('WEIGHT_DECAY'))
    epochs = int(os.environ.get('EPOCHS'))
    print_freq = int(os.environ.get('PRINT_FREQ'))
    save_freq = int(os.environ.get('SAVE_FREQ'))
    accumulate_step = int(os.environ.get('ACCUMULATE_STEP'))

    pretrain = os.environ.get('PRETRAIN')
    saved_model_dir = os.environ.get('SAVED_MODEL_DIR')

    imbalanced_sampler = str2bool(os.environ.get('IMBALANCED_SAMPLER'))
    fp16 = str2bool(os.environ.get('FP16'))  # Sử dụng hàm str2bool để chuyển đổi

    live_weight = float(os.environ.get('LIVE_WEIGHT'))

    # logging wandb
    using_wandb = str2bool(os.environ.get('USING_WANDB'))
    wandb_entity = os.environ.get('WANDB_ENTITY')
    wandb_key = os.environ.get('WANDB_KEY')
    wandb_project = os.environ.get('WANDB_PROJECT')
    notes = os.environ.get('NOTES')
    suffix_run_name = os.environ.get('SUFFIX_RUN_NAME')
    save_artifacts = str2bool(os.environ.get('SAVE_ARTIFACTS'))

    total_steps = int(os.environ.get('TOTAL_STEPS'))
    warmup_epochs = int(os.environ.get('WARMUP_EPOCHS'))

    resume = os.environ.get('RESUME')


config = ApplicationConfig()
