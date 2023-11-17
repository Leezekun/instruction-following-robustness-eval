export OPENAI_API_KEY='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'
export ANTHROPIC_API_KEY='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'
export TRANSFORMERS_CACHE='/mnt/raid0/zekun/.cache/huggingface/transformers'

cd ..

devices=0,1,2

# run 2: main results
for model in vicuna-33b-v1.3
do
    for n_shot in 4
    do
        for template in QCA
        do
            for position in end
                do
                for attack_type in direct
                do
                    for task_type in relevant
                    do
                        for test_mode in injected original relevant
                        do
                            for dataset in NaturalQuestions TriviaQA HotpotQA SQuAD
                            do
                                for defense in True
                                do
                                    CUDA_VISIBLE_DEVICES=$devices python -m run_evaluation \
                                                                        --dataset $dataset \
                                                                        --split dev \
                                                                        --n_samples 1000 \
                                                                        --template $template \
                                                                        --position $position \
                                                                        --attack_type $attack_type \
                                                                        --task_type $task_type \
                                                                        --model $model \
                                                                        --n_shot $n_shot \
                                                                        --test_mode $test_mode \
                                                                        --defense $defense \
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
done

# # run 3: QCA: attack and defense on natural questions
# for model in vicuna-33b-v1.3
# do
#     for n_shot in 4
#     do
#         for template in QCA
#         do
#             for position in end
#                 do
#                 for attack_type in direct ignore_previous
#                 do
#                     for task_type in relevant
#                     do
#                         for test_mode in injected # original relevant
#                         do
#                             for dataset in NaturalQuestions # TriviaQA
#                             do
#                                 for defense in True False
#                                 do
#                                     CUDA_VISIBLE_DEVICES=$devices python -m run_evaluation \
#                                                                         --dataset $dataset \
#                                                                         --split dev \
#                                                                         --n_samples 1000 \
#                                                                         --template $template \
#                                                                         --position $position \
#                                                                         --attack_type $attack_type \
#                                                                         --task_type $task_type \
#                                                                         --model $model \
#                                                                         --n_shot $n_shot \
#                                                                         --test_mode $test_mode \
#                                                                         --defense $defense \
#                                                                         --generate \
#                                                                         # --debug \
#                                                                         # --verbose
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# # run 5: CQA: attack and defense on natural questions
# for model in vicuna-33b-v1.3
# do
#     for n_shot in 4
#     do
#         for template in CQA
#         do
#             for position in end
#                 do
#                 for attack_type in direct ignore_next order_prefix
#                 do
#                     for task_type in relevant
#                     do
#                         for test_mode in injected # original relevant
#                         do
#                             for dataset in NaturalQuestions # TriviaQA
#                             do
#                                 for defense in True False
#                                 do
#                                     CUDA_VISIBLE_DEVICES=$devices python -m run_evaluation \
#                                                                         --dataset $dataset \
#                                                                         --split dev \
#                                                                         --n_samples 1000 \
#                                                                         --template $template \
#                                                                         --position $position \
#                                                                         --attack_type $attack_type \
#                                                                         --task_type $task_type \
#                                                                         --model $model \
#                                                                         --n_shot $n_shot \
#                                                                         --test_mode $test_mode \
#                                                                         --defense $defense \
#                                                                         --generate \
#                                                                         # --debug \
#                                                                         # --verbose
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# # run 6: task type
# for model in vicuna-33b-v1.3
# do
#     for n_shot in 4
#     do
#         for template in QCA
#         do
#             for position in end
#                 do
#                 for attack_type in direct
#                 do
#                     for task_type in relevant irrelevant
#                     do
#                         for test_mode in injected # original relevant
#                         do
#                             for dataset in NaturalQuestions # TriviaQA
#                             do
#                                 for defense in True
#                                 do
#                                     CUDA_VISIBLE_DEVICES=$devices python -m run_evaluation \
#                                                                         --dataset $dataset \
#                                                                         --split dev \
#                                                                         --n_samples 1000 \
#                                                                         --template $template \
#                                                                         --position $position \
#                                                                         --attack_type $attack_type \
#                                                                         --task_type $task_type \
#                                                                         --model $model \
#                                                                         --n_shot $n_shot \
#                                                                         --test_mode $test_mode \
#                                                                         --defense $defense \
#                                                                         --generate \
#                                                                         # --debug \
#                                                                         # --verbose
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# # run 7: position
# for model in vicuna-33b-v1.3
# do
#     for n_shot in 4
#     do
#         for template in QCA
#         do
#             for position in end middle start
#                 do
#                 for attack_type in direct
#                 do
#                     for task_type in relevant
#                     do
#                         for test_mode in injected # original relevant
#                         do
#                             for dataset in NaturalQuestions # TriviaQA
#                             do
#                                 for defense in True
#                                 do
#                                     CUDA_VISIBLE_DEVICES=$devices python -m run_evaluation \
#                                                                         --dataset $dataset \
#                                                                         --split dev \
#                                                                         --n_samples 1000 \
#                                                                         --template $template \
#                                                                         --position $position \
#                                                                         --attack_type $attack_type \
#                                                                         --task_type $task_type \
#                                                                         --model $model \
#                                                                         --n_shot $n_shot \
#                                                                         --test_mode $test_mode \
#                                                                         --defense $defense \
#                                                                         --generate \
#                                                                         # --debug \
#                                                                         # --verbose
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done