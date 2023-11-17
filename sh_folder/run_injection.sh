cd ..

for position in end
do
    for attack_type in direct # ignore_previous ignore_next order_prefix
    do
        for task_type in relevant # irrelevant
        do
            for test_mode in injected original relevant
            do
                for dataset in TriviaQA NaturalQuestions SQuAD HotpotQA
                do
                    CUDA_VISIBLE_DEVICES=$devices python -m run_injection \
                                                        --dataset $dataset  \
                                                        --split dev \
                                                        --n_samples 1000 \
                                                        --position $position \
                                                        --attack_type $attack_type \
                                                        --task_type $task_type \
                                                        --test_mode $test_mode
                done
            done
        done                                  
    done
done