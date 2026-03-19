from="."
#to="hermes-asr.southamerica-east1-a.poc-mj-474517:~/triton-info-extraction/testing_bench"
to="hermes-asr-b.southamerica-east1-b.poc-mj-474517:/home/pita/triton-whisperer"

rsync -zz -zarv --prune-empty-dirs \
    --exclude="testing_bench/results/*" --exclude="*.csv" --exclude="*.xlsx" \
    --exclude="logs/*" --exclude="input/*" --exclude="*.pyc" \
    --exclude=".git/*" --exclude=".vllm_cache/*" \
    --exclude="*.log" --exclude="hf_cache/*" "$from" "$to"