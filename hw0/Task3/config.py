import argparse

#Config for training model
def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="number per batch")
    parser.add_argument("--use_log", action='store_true', help="Transform mel-spec to DB")
    parser.add_argument("--data_path", type=str, default="/mnt/gestalt/home/dcn2001/NSynth_pre_task3")
    parser.add_argument("--model", type=str, default="CNNSA", help="model name")
    parser.add_argument("--model_save_path", type=str, default="./model_state/CNNSA/use_log/best_model.pth")
    parser.add_argument("--curve_save_path", type=str, default="./curve/CNNSA/use_log/")
    parser.add_argument("--init_lr", type=float, default=1e-5)   
    parser.add_argument("--decay_epoch", type=int, default=2)
    parser.add_argument("--l2_lambda", type=float, default=1e-4)

    args = parser.parse_args()
    return args