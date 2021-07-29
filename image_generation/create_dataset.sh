blender --background --python render_images.py -- --use_gpu 1 --num_images 10000 --split train --start_idx 0 --min_objects 1 --max_objects 1
blender --background --python render_images.py -- --use_gpu 1 --num_images 1000 --split test --start_idx 0 --min_objects 3 --max_objects 1
blender --background --python render_images.py -- --use_gpu 1 --num_images 100 --split val --start_idx 0 --min_objects 1 --max_objects 1
