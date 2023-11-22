
cd ..

devices=1

# run 2: main results
for model in alpaca-7b
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
                            for dataset in TriviaQA HotpotQA SQuAD NaturalQuestions
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

# run 3: QCA: attack and defense on natural questions
# for model in alpaca-7b
# do
#     for n_shot in 4
#     do
#         for template in QCA
#         do
#             for position in end
#                 do
#                 for attack_type in ignore_previous # direct # 
#                 do
#                     for task_type in relevant
#                     do
#                         for test_mode in injected
#                         do
#                             for dataset in NaturalQuestions
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

# run 5: CQA: attack and defense on natural questions
# for model in alpaca-7b
# do
#     for n_shot in 4
#     do
#         for template in CQA
#         do
#             for position in end
#                 do
#                 for attack_type in order_prefix # direct # ignore_next 
#                 do
#                     for task_type in relevant
#                     do
#                         for test_mode in injected
#                         do
#                             for dataset in NaturalQuestions
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

# run 6: task type
# for model in alpaca-7b
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
#                         for test_mode in injected
#                         do
#                             for dataset in NaturalQuestions
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
# for model in alpaca-7b
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

# # run 1: number of shots
# for model in alpaca-7b
# do
#     for n_shot in 0 1 2 3 4 5
#     do
#         for template in QCA
#         do
#             for position in end
#                 do
#                 for attack_type in direct
#                 do
#                     for task_type in relevant
#                     do
#                         for test_mode in injected
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