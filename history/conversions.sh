python line_drawing_conversion_redux.py anime_style $SHARED_DATA_DIR/imagenet1k-256/val $SHARED_DATA_DIR/imagenet1k-line/anime_style/val --subfolders

python line_drawing_conversion_redux.py contour_style $SHARED_DATA_DIR/imagenet1k-256/val $SHARED_DATA_DIR/imagenet1k-line/contour_style/val

python line_drawing_conversion_redux.py opensketch_style $SHARED_DATA_DIR/imagenet1k-256/val $SHARED_DATA_DIR/imagenet1k-line/opensketch_style/val


# to ffcv datasets
python write_ffcv_dataset.py \
        --cfg.dataset=imagenet1k \
        --cfg.split=val \
        --cfg.data_dir=$SHARED_DATA_DIR/imagenet1k-line/imagenet1k-anime_style \
        --cfg.write_path=$SHARED_DATA_DIR/imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop.ffcv \
        --cfg.write_mode=jpg \
        --cfg.min_resolution=256 \
        --cfg.max_resolution=512 \
        --cfg.max_enforced_with=center_crop \
        --cfg.num_workers=16 \
        --cfg.chunk_size=100 \
        --cfg.shuffle_indices=1 \
        --cfg.jpeg_quality=100 \
        --cfg.compress_probability=1.0

sha256sum imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop.ffcv

python write_ffcv_dataset.py \
        --cfg.dataset=imagenet1k \
        --cfg.split=val \
        --cfg.data_dir=$SHARED_DATA_DIR/imagenet1k-line/imagenet1k-contour_style \
        --cfg.write_path=$SHARED_DATA_DIR/imagenet1k-line-ffcv/imagenet1k-contour_style_val_jpg_q100_s256_lmax512_crop.ffcv \
        --cfg.write_mode=jpg \
        --cfg.min_resolution=256 \
        --cfg.max_resolution=512 \
        --cfg.max_enforced_with=center_crop \
        --cfg.num_workers=16 \
        --cfg.chunk_size=100 \
        --cfg.shuffle_indices=1 \
        --cfg.jpeg_quality=100 \
        --cfg.compress_probability=1.0   
        
sha256sum imagenet1k-contour_style_val_jpg_q100_s256_lmax512_crop.ffcv

python write_ffcv_dataset.py \
        --cfg.dataset=imagenet1k \
        --cfg.split=val \
        --cfg.data_dir=$SHARED_DATA_DIR/imagenet1k-line/imagenet1k-opensketch_style \
        --cfg.write_path=$SHARED_DATA_DIR/imagenet1k-line-ffcv/imagenet1k-opensketch_style_val_jpg_q100_s256_lmax512_crop.ffcv \
        --cfg.write_mode=jpg \
        --cfg.min_resolution=256 \
        --cfg.max_resolution=512 \
        --cfg.max_enforced_with=center_crop \
        --cfg.num_workers=16 \
        --cfg.chunk_size=100 \
        --cfg.shuffle_indices=1 \
        --cfg.jpeg_quality=100 \
        --cfg.compress_probability=1.0           
        
sha256sum imagenet1k-opensketch_style_val_jpg_q100_s256_lmax512_crop.ffcv        