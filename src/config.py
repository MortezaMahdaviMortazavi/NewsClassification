embed_size = 512
embed_proj_size = 512
max_seq_length = 512
hidden_size = 512
dropout_rate = 0.2
num_layers = 2
is_bidirectional = False
cross_val = 4
data_path = '../dataset/cleaned_train_length.csv'
train_path = f'../dataset/fold_{cross_val}/train.csv'
val_path = f'../dataset/fold_{cross_val}/validation.csv'
batch_size = 8  # Set your desired batch size
num_epochs = 5
lr = 1e-3
logfile = '../logs/logfile4.log'
checkpoint_path = f'../checkpoints/model_checkpoint_{cross_val}.pth'
checkpoint_target_model = '../checkpoints/model_checkpoint_3.pth'
num_classes = 7
LABEL2IDX = {}
IDX2LABEL = {idx:label for label,idx in LABEL2IDX.items()}

CLASSES = [
    "Miscellaneous",        
    "Economy",                
    "Politics",               
    "Sport",                
    "Science and Culture",    
    "Social",                  
    "Literature and Art"
]