import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="number per batch")
    parser.add_argument("--data_path", type=str, default="/home/data1/dcn2001/MUSDBHQ_HW")
    parser.add_argument("--pretrained_model_path", type=str, default="")
    parser.add_argument("--model_save_path", type=str, default="./model_state/OpenUnmix/best_aug_6.ckpt")
    parser.add_argument("--curve_save_path", type=str, default="./curve/OpenUnmix_aux_6.jpg")
    parser.add_argument("--init_lr", type=float, default=1e-4)   
    #parser.add_argument("--decay_epoch", type=int, default=10)
    parser.add_argument("--l2_lambda", type=float, default=1e-4)

    args = parser.parse_args()
    return args