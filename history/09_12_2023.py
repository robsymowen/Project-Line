# ====================================================
#  Line-Drawing model trained with imagenet_line_stdonly stats, sgd lr 0.01
# ====================================================

./train.sh alexnet in1k_anime_style \
    --train_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_train_jpg_q100_s256_lmax512_crop-2b7bdbda.ffcv \
    --val_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --test_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --optimizer sgd \
    --lr 0.01

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12234824
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_anime_style/alexnet/20231209_043147
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231209_043147
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231209_043147/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231209_043147/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231209_043147/log_test.txt

# ====================================================
#  Line-Drawing model trained with imagenet_rgb_avg_stdonly stats, sgd lr 0.005
# ====================================================

./train.sh alexnet in1k_anime_style \
    --train_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_train_jpg_q100_s256_lmax512_crop-2b7bdbda.ffcv \
    --val_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv \
    --test_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv \
    --image_stats_line imagenet_rgb_avg_stdonly \
    --optimizer sgd \
    --lr 0.005

checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
checkpoint_path:  /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/checkpoint True
Submitted job_id: 12235701
Logs and checkpoints will be saved at local_dir: /n/holylabs/LABS/alvarez_lab/Users/alvarez/Projects/Project-Line/logs/models/in1k_anime_style/alexnet/20231209_045059
Remote Storage Location: s3://visionlab-members/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231209_045059
train_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231209_045059/log_train.txt
val_log = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231209_045059/log_val.txt
test_url = https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231209_045059/log_test.txt
