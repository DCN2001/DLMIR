import argparse

#Config for training model
def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="number per batch")
    parser.add_argument("--num_workers", type=int, default=8, help="number of worker")
    parser.add_argument("--freeze", action='store_true', help="Freeze feature extractor or not")
    parser.add_argument("--data_path", type=str, default="/mnt/gestalt/home/dcn2001/hw1/slakh")
    parser.add_argument("--model_save_path", type=str, default="./model_state/best_model_finetune.pth", help="Path to save ckpt")
    parser.add_argument("--curve_save_path", type=str, default="./curve/", help="Path to save training curve")  
    parser.add_argument("--init_lr", type=float, default=1e-4)   
    parser.add_argument("--decay_epoch", type=int, default=2)   #Epoch which decay lr if validation loss doesm't decrease 
    parser.add_argument("--l2_lambda", type=float, default=1e-4)    #L2 regularization

    args = parser.parse_args()
    return args
