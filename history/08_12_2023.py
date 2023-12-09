# Let's train!

# ====================================================
#  RGB model trained with imagenet_rgb_avg stats
# ====================================================

./train.sh alexnet in1k_rgb \
    --train_dataset imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv \
    --val_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --test_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12154739
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_rgb/alexnet/20231208_144920
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_144920
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_144920/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_144920/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_144920/log_test.txt

# ====================================================
#  RGB model trained with imagenet_rgb_avg stats lr 0.00001
# ====================================================

./train.sh alexnet in1k_rgb \
    --train_dataset imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv \
    --val_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --test_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --lr 0.00001

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12156209
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_rgb/alexnet/20231208_151752
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_151752
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_151752/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_151752/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_151752/log_test.txt

# ====================================================
#  RGB model trained with imagenet_rgb_avg stats lr 0.00005
# ====================================================

./train.sh alexnet in1k_rgb \
    --train_dataset imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv \
    --val_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --test_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --lr 0.00005

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12169718
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_rgb/alexnet/20231208_165145
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_165145
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_165145/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_165145/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_165145/log_test.txt

# ====================================================
#  RGB model trained with imagenet_rgb_avg stats sgd lr 0.01
# ====================================================

./train.sh alexnet in1k_rgb \
    --train_dataset imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv \
    --val_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --test_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --optimizer sgd \
    --lr 0.01

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12189068
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_rgb/alexnet/20231208_181051
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_181051
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_181051/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_181051/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_181051/log_test.txt

# ====================================================
#  RGB model trained with imagenet_rgb_avg stats sgd lr 0.001
# ====================================================

./train.sh alexnet in1k_rgb \
    --train_dataset imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv \
    --val_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --test_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --optimizer sgd \
    --lr 0.001

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12191743
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_rgb/alexnet/20231208_183236
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183236
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183236/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183236/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183236/log_test.txt

# ====================================================
#  RGB model trained with imagenet_rgb_avg stats sgd lr 0.005
# ====================================================

./train.sh alexnet in1k_rgb \
    --train_dataset imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv \
    --val_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --test_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --optimizer sgd \
    --lr 0.005

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12192049
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_rgb/alexnet/20231208_183615
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183615
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183615/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183615/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183615/log_test.txt

# ====================================================
#  Line-Drawing model trained with imagenet_line_stdonly stats
# ====================================================

./train.sh alexnet in1k_anime_style \
    --train_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_train_jpg_q100_s256_lmax512_crop-2b7bdbda.ffcv \
    --val_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --test_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12154923
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_anime_style/alexnet/20231208_145131
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_145131
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_145131/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_145131/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_145131/log_test.txt

# ====================================================
#  Line-Drawing model trained with imagenet_line_stdonly stats, sgd lr 0.005
# ====================================================

./train.sh alexnet in1k_anime_style \
    --train_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_train_jpg_q100_s256_lmax512_crop-2b7bdbda.ffcv \
    --val_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --test_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --optimizer sgd \
    --lr 0.005

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12193258
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_anime_style/alexnet/20231208_184601
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_184601
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_184601/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_184601/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_184601/log_test.txt

# ====================================================
#  Line-Drawing model trained with imagenet_rgb_avg stats, sgd lr 0.005
# ====================================================

./train.sh alexnet in1k_anime_style \
    --train_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_train_jpg_q100_s256_lmax512_crop-2b7bdbda.ffcv \
    --val_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --test_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --image_stats_line imagenet_rgb_avg \
    --optimizer sgd \
    --lr 0.005

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12193399
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_anime_style/alexnet/20231208_184720
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_184720
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_184720/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_184720/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_184720/log_test.txt
