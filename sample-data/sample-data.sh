wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
unzip fiqa.zip

export dataset="fiqa"
python -m gpl.train \
    --path_to_generated_data "generated/$dataset" \
    --batch_size_gpl 4 \
    --gpl_steps 5 \
    --output_dir "output/$dataset" \
    --evaluation_data "./$dataset" \
    --qgen_prefix "qgen" \
    --new_size 10 \

# One can run `python -m gpl.train --help` for the information of all the arguments