export OPENAI_API_KEY='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'
export ANTHROPIC_API_KEY='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'
export TRANSFORMERS_CACHE='/mnt/raid0/zekun/.cache/huggingface/transformers'

cd ..

devices=3,4,5

# task type
for model in falcon-40b-instruct
do
    for n_shot in 4 
    do
        for template in QCA
        do
            for position in end
                do
                for attack_type in direct # first_suffix order_prefix first_prefix
                do
                    for task_type in irrelevant # irrelevant
                    do
                        for test_mode in injected # original relevant # injected original
                        do
                            for dataset in NaturalQuestions # TriviaQA
                            do
                                CUDA_VISIBLE_DEVICES=$devices python -m run_evaluation \
                                                                    --dataset $dataset \
                                                                    --split dev \
                                                                    --n_samples 500 \
                                                                    --template $template \
                                                                    --position $position \
                                                                    --attack_type $attack_type \
                                                                    --task_type $task_type \
                                                                    --model $model \
                                                                    --n_shot $n_shot \
                                                                    --test_mode $test_mode \
                                                                    --generate \
                                                                    --debug \
                                                                    --verbose
                            done
                        done
                    done
                done
            done
        done
    done
done

# position 
for model in falcon-40b-instruct
do
    for n_shot in 4 
    do
        for template in QCA
        do
            for position in end start middle
                do
                for attack_type in direct # first_suffix order_prefix first_prefix
                do
                    for task_type in relevant # irrelevant
                    do
                        for test_mode in injected # original relevant # injected original
                        do
                            for dataset in NaturalQuestions # TriviaQA
                            do
                                CUDA_VISIBLE_DEVICES=$devices python -m run_evaluation \
                                                                    --dataset $dataset \
                                                                    --split dev \
                                                                    --n_samples 500 \
                                                                    --template $template \
                                                                    --position $position \
                                                                    --attack_type $attack_type \
                                                                    --task_type $task_type \
                                                                    --model $model \
                                                                    --n_shot $n_shot \
                                                                    --test_mode $test_mode \
                                                                    --generate \
                                                                    # --debug \
                                                                    # --verbose
                            done
                        done
                    done
                done
            done
        done
    done
done