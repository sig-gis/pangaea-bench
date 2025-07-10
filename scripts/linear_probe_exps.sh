echo Task ${BATCH_TASK_INDEX}

wandb login 27d91b7cd8af63a77b0f025a0df4e85c3caf7cd5

if [ $BATCH_TASK_INDEX -eq 0 ]; then
    scripts/croma/test_croma_hlsburnscars_linear.sh
elif [ $BATCH_TASK_INDEX -eq 1 ]; then
    scripts/croma/test_croma_ai4smallfarms_linear.sh
elif [ $BATCH_TASK_INDEX -eq 2 ]; then
    scripts/croma/test_croma_mados_linear.sh
elif [ $BATCH_TASK_INDEX -eq 3 ]; then
    scripts/croma/test_croma_spacenet7_linear.sh
elif [ $BATCH_TASK_INDEX -eq 4 ]; then
    scripts/croma/test_croma_sen1floods_linear.sh
elif [ $BATCH_TASK_INDEX -eq 5 ]; then
    scripts/croma/test_croma_mtcropclassification_linear.sh
fi

