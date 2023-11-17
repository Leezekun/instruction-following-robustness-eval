cd ..

# run 2: main results
for human_eval_num in 50
do
    for human_eval_group in 5
    do
        python -m prepare_human_evals \
                --dataset NaturalQuestions \
                --split dev \
                --n_samples 1000 \
                --template QCA \
                --position end \
                --attack_type direct \
                --task_type relevant \
                --n_shot 4 \
                --test_mode injected \
                --defense True \
                --human_eval_num $human_eval_num \
                --human_eval_group $human_eval_group
    done
done