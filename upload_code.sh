from="."
#to="hermes-ner.southamerica-east1-a.poc-mj-474517:~/triton-info-extraction/testing_bench"
to="hermes-asr.southamerica-east1-a.poc-mj-474517:/home/pita/triton-whisperer"

rsync -zz -zarv --prune-empty-dirs \
    --exclude="testing_bench/results/*" --exclude="logs/*" --exclude="input/*" \
    --exclude="*.pyc" --exclude=".git/*" --exclude=".env" \
    --exclude=".vllm_cache/*" --exclude="*.log" --exclude="hf_cache/*" "$from" "$to"